"""Build derivatives/vikitraj of the release, except the per-frame tables written by build_frame_tables.py.

Writes dataset_description.json, manifest.tsv/.json, splits.json, cameras/, results/ and checkpoints/ under
<out>/derivatives/vikitraj. Every written file is scanned for study IDs, names, dates, clock times and raw paths; a hit
fails the run.

    python dataset/build_derivatives.py --participant XXXXXXX103 [--out DIR]   cameras of one participant
    python dataset/build_derivatives.py --global [--out DIR]                the global parts only
    python dataset/build_derivatives.py --all [--out DIR]                   everything
"""
import argparse
import ast
import json
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import BENCH, DERIV, VIEWS, bids_name, sub_id, write_json, write_tsv  # noqa: E402

CAMERA_VIEWS = ("top", "side")          # views with calibrated cameras (cameras/, labels2d)
FINETUNE = BENCH / "finetune"
TRUE_FALSE = {True: "true", False: "false"}

# Result tables behind the paper's tables and figures (outputs/benchmark/results/<name>.csv) -> README line.
RESULTS = {
    "E1_2d": "E1, 2D keypoints per view against the projected 6 Hz VICON joints: median / mean / p90 error (mm at "
             "the joint, and px), PCK@20 mm and coverage per split, group, view, model variant, arm role, joint and "
             "label set; mean over participants with a bootstrap 95 % CI",
    "E1_2d_participants": "E1 per participant (frames of all its trials pooled), the values E1_2d averages",
    "E2_3d": "E2, two-view 3D joints (top + side triangulated) against VICON: median / mean / p90 error (mm), "
             "PCK@20 mm and coverage, as E1",
    "E2_3d_participants": "E2 per participant",
    "E3_screen": "E3, end-effector trajectory on the screen from VICON and from video (2D top view or two-view 3D, "
                 "mapped to the screen leave-one-trial-out) against the iPad trajectory: RMSE, median and p90 error "
                 "(mm), coverage, path-length ratio and speed correlation; subset own (each source on its valid "
                 "trials) or common (trials valid for every source)",
    "E3_screen_participants": "E3 per participant (median over its trials)",
    "E4_kin_paths": "E4 per trial and source (vicon or a model): elbow flexion and shoulder elevation RMSE and range "
                    "of motion (deg) with coverage (share of samples), fingertip and wrist peak speed (mm/s), time to "
                    "peak (s), straightness, number of speed peaks, SPARC and LDLJ; `arm_role` of the active arm",
    "E4_kin_agreement": "E4 agreement of each kinematic feature with VICON per source: ICC(2,1), Bland-Altman bias "
                        "and 95 % limits of agreement, MAE, VICON mean and Pearson r",
    "E4_kin_angles": "E4 joint-angle RMSE (deg, bootstrap CI) and range-of-motion error per source and group",
    "E4_kin_effects": "E4 group effects per source and feature: Cohen's d UCP vs TD and paired d_z affected vs "
                      "less-affected arm within UCP and within TD",
    "E5_lift3d_rtmw_x": "E5, 3D from RTMW-X keypoints: triangulation, per-frame two-view lifting (mlp) and the "
                        "temporal network (tcn); error per joint and group (mm, bootstrap CI)",
    "E5_lift3d_rtmw_x_participants": "E5 lifting on RTMW-X per participant",
    "E5_lift3d_sapiens2_1b": "E5, 3D from Sapiens2-1B keypoints, as E5_lift3d_rtmw_x",
    "E5_lift3d_sapiens2_1b_participants": "E5 lifting on Sapiens2-1B per participant",
    "E6a_group_limb": "E6a, E1 / E2 error by group and by the evaluated arm: `limb` is its status (affected | "
                      "less_affected) and `role` whether it played the trial (active | resting); unlike `arm_role`, "
                      "`limb` describes the evaluated arm, which may be the resting one",
    "E6b_scaling": "E6b, the pooled ViTPose+-B fine-tune trained on 4, 8, 12 and all 17 training participants "
                   "(E1, E2 and E3 metrics)",
    "E6c_alignment_shift": "E6c, sensitivity to temporal misalignment: test-time reference shifts and training on "
                           "labels shifted by 67 / 200 ms",
    "E6d_views": "E6d, top 2D, side 2D and two-view 3D errors side by side",
    "E6e_filtering": "E6e, raw keypoints against raw markers vs the 6 Hz filtered default",
    "E6f_selection_bias": "E6f, VICON fingertip reach features of the VICON-valid trials that pass both video tests "
                          "vs those that fail them, per group (Cohen's d)",
    "E6f_selection_paths": "E6f per trial: video validity and the VICON fingertip reach features; `arm_role` of the "
                           "active arm",
}
# Released names: per-trial files are *_trials, trial counts n_trials, and the manifest's column names and values.
RESULT_FILE_NAMES = {"E4_kin_paths": "E4_kin_trials", "E6f_selection_paths": "E6f_selection_trials"}
RESULT_COLUMNS = {"n_paths": "n_trials", "move_duration_s": "movement_time"}
ARM_ROLE_TABLES = ("E4_kin_paths", "E6f_selection_paths")   # their `limb` is the active arm's role (arm_role)

MIN_IN_FRONT = 0.99         # build_frame_tables.MIN_IN_FRONT: share of the marker samples in front of a usable camera

SPLITS_NOTE = ("Participant-level split used in the paper. Values are participant_id labels; folds are listed in order fold-1 ... "
               "fold-5. ViconTraj-MS trains on folds 2-5, stops early on fold 1 and is scored on the test "
               "participants. cross_visit: train on ses-1, test on ses-2 (not used in the paper).")

SEQ_ARCHITECTURE = {
    "type": "temporal convolutional network (benchmark/e5_seq.py, TCN), receptive field 29 frames",
    "layers": "1x1 conv 30 -> 128; 3 residual blocks (conv kernel 5 with dilation 1, 2, 4, GELU, 1x1 conv); "
              "GELU; 1x1 conv 128 -> 20",
    "input": "(30, T) per clip of one view: x, y of the base model's 10 joints (keypoints2d/<base>, image px, "
             "output_channels order) normalised by the unpadded person box as ((x, y) - box centre) / max(box "
             "width, box height), 0 where missing, then 10 presence flags (1 = joint present)",
    "output": "(20, T) correction added to the 20 normalised input coordinates; image px = (input + correction) x "
              "max(box width, box height) + box centre",
    "training": "one network per view (model_top.pt, model_side.pt); random 64-frame windows, batch 64, 40 epochs "
                "x 200 steps, AdamW lr 1e-3, weight decay 1e-4, cosine decay; loss L1 position + L1 velocity + "
                "0.1 x L1 jerk against the projected 6 Hz VICON joints; weights of the best validation L1",
}

# Every checkpoint takes the participant's person box per frame; the boxes of the paper are in boxes/.
PERSON_BOX = ("boxes/<sub>/<ses>/<...>_boxes.tsv (x0, y0, x1, y1 per clip frame, image px). The paper used RT-DETR (Hugging Face "
              "PekingU/rtdetr_r50vd_coco_o365, person class, score >= 0.3) on the source video, 0.5 s beyond each "
              "end of the clip: per frame the person with the highest IoU (> 0.3) with a reference box of the video "
              "(median of the largest confident person, area x score, over up to 40 frames spread across its valid "
              "trials), gaps linearly interpolated, then a 5-frame running median; x0, y0, x1, y1 in image px.")
CROP = ("the person box padded by 1.25 about its centre (cx, cy) and enlarged to the crop's aspect ratio, warped by "
        "the affine [[s, 0, W/2 - s cx], [0, s, H/2 - s cy]] with s = W / padded box width")
FT_VITPOSE_INFERENCE = {
    "person_box": PERSON_BOX,
    "crop": f"W x H = 192 x 256 px RGB: {CROP}; values / 255 normalised with mean (0.485, 0.456, 0.406) and std "
            "(0.229, 0.224, 0.225)",
    "forward": "VitPoseForPoseEstimation(pixel_values=crop, dataset_index=0): dataset_index 0 selects the COCO "
               "expert of ViTPose+ (required); 10 heatmaps of 64 x 48 (height x width)",
    "decoding": "DARK sub-pixel peak (heatmap smoothed with the training Gaussian, sigma 2, then one second-order "
                "Taylor step on its log at the arg-max), x 4 to crop px, score = heatmap maximum; averaged with the "
                "prediction on the horizontally mirrored crop (left / right channels swapped back); crop px to image "
                "px by the inverse crop affine",
}
FT_SAPIENS_INFERENCE = {
    "person_box": PERSON_BOX,
    "crop": f"W x H = 768 x 1024 px RGB: {CROP}; values 0-255 normalised with mean (123.675, 116.28, 103.53) and "
            "std (58.395, 57.12, 57.375)",
    "forward": "the original Sapiens2 pose model of base_model with its decode head replaced by head.pt; 308 "
               "heatmaps of 256 x 192 (height x width), of which output_channels are the 10 joints",
    "decoding": "the model's UDP heatmap codec (sigma 6), no flip test; crop px to image px by the inverse crop "
                "affine",
}
SEQ_INFERENCE = {
    "person_box": PERSON_BOX,
    "views": "model_top.pt for top clips, model_side.pt for side clips",
    "edges": "in the paper the head ran on the base model's keypoints of the clip extended by 0.5 s on each side; "
             "on keypoints2d (clip frames only) the first and last 14 frames see zero padding instead",
}
FT_LOG_UNITS = {
    "train_loss": "heatmap MSE over the labelled joints, mean over the epoch",
    "val_median_px, val_mean_px": "median / mean Euclidean error of the visible joint labels of the fold-1 "
                                  "validation crops, in px of the {} model crop (not image px)",
    "minutes": "training time of the epoch",
}
SEQ_LOG_UNITS = {
    "train_loss": "L1 position + L1 velocity + 0.1 x L1 jerk of the last batch of the epoch, box-normalised units",
    "val_l1": "mean absolute error of the corrected x and y of the joints with VICON data over 512 random 64-frame "
              "windows of the fold-1 validation clips, in box-normalised units (px / max(box width, box height))",
}

# De-identification: no written file may match these (case-insensitive) or a term of release/private/scrub_terms.txt;
# checkpoint weights are searched for FORBIDDEN_BINARY.
DATES_TIMES = [r"\b(?:19|20)\d\d-\d\d-\d\d\b", r"\b\d{1,2}:\d\d:\d\d\b",
               r"\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (?:19|20)\d\d\b"]
FORBIDDEN_TEXT = [r"XXXXXXX", r"/home/|[A-Z]:\\", r"raw_video_data|game_data/|vicon_data/|outputs/"] + DATES_TIMES
FORBIDDEN_BINARY = re.compile(rb"XXXXXXX|/home/")


# ----------------------------------------------------------------------------- trials and identifiers
@lru_cache(None)
def trials():
    """common.manifest() with participant_id, session and run. The four trials without trial_in_visit get their round
    from the dense rank of group_no within the visit, which equals trial_in_visit on every other row."""
    m = common.manifest().copy()
    rank = m.groupby(["pid", "visit"]).group_no.rank(method="dense")
    known = m.trial_in_visit.notna()
    if not (rank[known] == m.trial_in_visit[known]).all():
        raise ValueError("trial_in_visit is not the rank of group_no within the visit")
    m["participant_id"] = m.pid.map(sub_id)
    m["session"] = m.visit.map(common.ses_id)
    m["run"] = [common.run_id(t, p) for t, p in zip(m.trial_in_visit.fillna(rank), m.path)]
    return m


def scrub(text):
    """Free text with study IDs mapped to sub IDs and dates, clock times and file paths removed."""
    text = re.sub(r"XXXXXXX\d{3}", lambda mo: sub_id(mo.group()), str(text))
    for pattern in DATES_TIMES + [r"\S*[/\\]\S*"]:
        text = re.sub(pattern, "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def map_ids(value):
    """scrub() applied to every string of a JSON value (strings, lists, dicts)."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [map_ids(v) for v in value]
    if isinstance(value, dict):
        return {k: map_ids(v) for k, v in value.items()}
    return value


def game_terms(text):
    """Free text in the paper's terms: the trajectory logged by the game is the game trajectory, and every spelling of
    its per-trial test becomes 'game-trajectory test'."""
    text = (text.replace("touch-log", "game-trajectory").replace("touch log", "game trajectory")
            .replace("fingertip vs touch", "fingertip vs game trajectory"))
    return re.sub(r"(?<!game-)trajectory test", "game-trajectory test", text)


def flags(series):
    return series.astype(bool).map(TRUE_FALSE)


# ----------------------------------------------------------------------------- manifest
@lru_cache(None)
def calibration():
    """cameras/calibration.csv: one row per calibrated top / side video (npz files without a row are ignored)."""
    return pd.read_csv(BENCH / "cameras" / "calibration.csv")


@lru_cache(None)
def in_front_share(pid, camera):
    """Share of the video's VICON marker samples in front of the camera (build_frame_tables.camera_checks); NaN for a
    camera without VICON trials."""
    return frame_tables().camera_checks(pid).get(camera, (None, float("nan"), None))[1]


@lru_cache(None)
def frame_tables():
    """build_frame_tables, imported on first use (it loads the benchmark modules)."""
    import build_frame_tables
    return build_frame_tables


def usable(c):
    """A calibration.csv row whose camera labels2d uses: calib_ok and at least MIN_IN_FRONT of the video's marker
    samples in front of it (the frame tables' test; it rejects one camera fitted through the participant)."""
    return bool(c.calib_ok) and in_front_share(c.pid, c.camera) >= MIN_IN_FRONT


@lru_cache(None)
def usable_videos():
    c = calibration()
    return {(r.pid, r.visit, r.view, r.video) for r in c.itertuples() if usable(r)}


@lru_cache(None)
def trajectory_qa_limit():
    """QA_MAX_MM of benchmark/e3_screen.py, read from its source (the module is not imported)."""
    tree = ast.parse((common.ROOT / "benchmark" / "e3_screen.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "QA_MAX_MM" for t in node.targets):
            return float(ast.literal_eval(node.value))
    raise ValueError("QA_MAX_MM not found in benchmark/e3_screen.py")


@lru_cache(None)
def trajectory_qa():
    """{(pid, visit): failed} of the sessions the game-trajectory QA tested; untested sessions are absent (benchlib
    reads them as passed, the release as n/a)."""
    q = pd.read_csv(common.benchlib.QA_FILE)
    if not (q.failed == (q.vicon_trajectory_rmse_mm > trajectory_qa_limit())).all():
        raise ValueError("trajectory QA file disagrees with QA_MAX_MM")
    return {(p, v): bool(f) for p, v, f in zip(q.pid, q.visit, q.failed)}


def released_files(m):
    """Relative paths (to the dataset root) of each trial's files written by the other builders; None = no file."""
    cal = usable_videos()
    rows = []
    for r in m.itertuples():
        sub, ses, run = r.participant_id, r.session, r.run
        motion = Path(sub, ses, "motion")
        has_vicon = r.vicon_frames > 0
        f = {"vicon_motion": motion / bids_name(sub, ses, "motion", ".tsv", tracksys="vicon", run=run) if has_vicon else None,
             "ipad_motion": motion / bids_name(sub, ses, "motion", ".tsv", tracksys="ipad", run=run),
             "events": motion / bids_name(sub, ses, "events", ".tsv", run=run)}
        for v in VIEWS:
            has_clip = isinstance(getattr(r, f"{v}_clip_path"), str)
            f[f"{v}_video"] = Path(common.VIDEOS, sub, ses, bids_name(sub, ses, "video", ".mp4", acq=v, run=run)) if has_clip else None
            if v in CAMERA_VIEWS:
                labelled = has_clip and has_vicon and (r.pid, r.visit, v, getattr(r, f"{v}_source_video")) in cal
                f[f"{v}_labels"] = Path(DERIV, "labels2d", sub, ses, bids_name(sub, ses, "labels", ".tsv", acq=v, run=run)) if labelled else None
        rows.append({k: (str(p) if p else None) for k, p in f.items()})
    return pd.DataFrame(rows, index=m.index)


def manifest_table():
    m = trials()
    t = pd.DataFrame({
        "participant_id": m.participant_id, "session": m.session, "run": m.run, "group": m.group,
        "hand": m.hand.str.lower(), "arm_role": [common.benchlib.limb_of(p, h) for p, h in zip(m.pid, m.hand)],
        "movement_time": m.move_duration_s, "vicon_valid": flags(m.vicon_valid),
        "vicon_invalid_reason": m.vicon_invalid_reason.fillna("").map(scrub).map(game_terms),
        "vicon_align_source": m.vicon_align_source, "vicon_align_support": m.vicon_align_support.astype("Int64"),
        "vicon_align_err": m.vicon_align_err_s, "vicon_coverage": m.vicon_coverage,
        "trajectory_qa_failed": [TRUE_FALSE.get(trajectory_qa().get((p, v))) for p, v in zip(m.pid, m.visit)]})
    files = released_files(m)
    for v in VIEWS:
        t[f"{v}_valid"] = flags(m[f"{v}_valid"])
        t[f"{v}_invalid_reason"] = m[f"{v}_invalid_reason"].fillna("").map(scrub).map(game_terms)
        t[f"{v}_align_err"] = m[f"{v}_align_err_s"]
        t[f"{v}_n_frames"] = m[f"{v}_n_frames"].astype("Int64")
        t[f"{v}_fps"] = m[f"{v}_fps"]
        t[f"{v}_video"] = files[f"{v}_video"]
        t[f"{v}_lead_black_frames"] = pd.array([max(0, -int(s)) if f else None for s, f in
                                                zip(m[f"{v}_start_frame"].fillna(0), files[f"{v}_video"])], "Int64")
    views_ok = [";".join(v for v, o in zip(VIEWS, ok) if o) for ok in zip(*(m[f"{v}_valid"].astype(bool) for v in VIEWS))]
    t["benchmark_valid"] = flags(m.vicon_valid.astype(bool) & (np.array(views_ok) != ""))
    t["valid_views"] = views_ok
    for c in ("vicon_motion", "ipad_motion", "events", "top_labels", "side_labels"):
        t[c] = files[c]
    return t.replace({"": np.nan}).sort_values(["participant_id", "session", "run"])


def manifest_sidecar():
    valid = {"true": "passes every check", "false": "fails at least one check (see the reason column)"}
    reason = "Semicolon-separated reasons why {} is false; n/a when valid."
    qa_mm = trajectory_qa_limit()
    d = {
        "participant_id": {"Description": "Participant label (participants.tsv)."},
        "session": {"Description": "Visit.", "Levels": {"ses-1": "first visit", "ses-2": "second visit"}},
        "run": {"Description": "Trial within the session in game order: run = (round - 1) * 4 + trial of the "
                               "round, four trials per game round (run-01 ... run-12)."},
        "group": {"Description": "Participant group.",
                  "Levels": {"TD": "typically developing", "UCP": "unilateral cerebral palsy"}},
        "hand": {"Description": "Hand that played the trial.", "Levels": {"left": "left hand", "right": "right hand"}},
        "arm_role": {"Description": "Role of the active arm.",
                     "Levels": {"affected": "paretic arm (UCP) or non-dominant arm (TD)",
                                "less_affected": "the other arm"}},
        "movement_time": {"Description": "Duration of the movement window (first to last sample of the iPad "
                                         "end-effector trajectory).", "Units": "s"},
        "vicon_valid": {"Description": "VICON data aligned and usable: review passed (fingertip marker missing "
                                       "<= 10 %), the recording covers the whole movement window, the game-to-VICON "
                                       "offset is backed by >= 2 agreeing estimates and within 0.1 s of the consensus "
                                       "of its recording, and the VICON fingertip reproduces the iPad trajectory "
                                       "(game-trajectory test).", "Levels": valid},
        "vicon_invalid_reason": {"Description": reason.format("vicon_valid")},
        "vicon_align_source": {
            "Description": "Source of the game-to-VICON clock offset of the trial.",
            "Levels": {"manual": "the manually marked movement start: agreeing with the recording's consensus, or "
                                 "taken as-is for participants aligned from manual marks only (see vicon_align_err)",
                       "xcorr": "fingertip / trajectory speed cross-correlation, agreeing with the consensus",
                       "trial_consensus": "the consensus offset of the recording (own estimates missing or disagreeing)",
                       "manual_trial_median": "the median manual offset of the recording (participants aligned from "
                                              "manual marks only; this trial has no mark)",
                       "no_vicon_trial": "no VICON recording of this game round"}},
        "vicon_align_support": {"Description": "Number of offset estimates of the recording (manual marks and, "
                                               "unless aligned from manual marks only, cross-correlations of its four "
                                               "trials) that agree with the consensus; <= 1 = unverified."},
        "vicon_align_err": {"Description": "Absolute difference between the trial's game-to-VICON offset and the "
                                           "consensus offset of its recording (the median of its manual offsets for "
                                           "participants aligned from manual marks only).", "Units": "s"},
        "vicon_coverage": {"Description": "Share of the movement window covered by the VICON recording (0-1)."},
        "trajectory_qa_failed": {
            "Description": "Session-level game-trajectory QA (separate from the per-trial game-trajectory test of "
                           "vicon_valid): "
                           "the VICON fingertip, mapped to the screen leave-one-trial-out, is compared with the iPad "
                           "trajectory on the session's trials that are vicon_valid and top_valid, and the session "
                           f"fails when the median RMSE over them exceeds {qa_mm:g} mm. Failed sessions are excluded "
                           "from every experiment. n/a = not tested: no trial of the session is both vicon_valid and "
                           "top_valid (its VICON-valid trials with a valid side view are still used).",
            "Levels": {"true": f"failed (median RMSE > {qa_mm:g} mm)", "false": f"passed (median RMSE <= {qa_mm:g} mm)"}},
        "benchmark_valid": {"Description": "vicon_valid and at least one valid view (top or side). The paper's "
                                           "selection per view is vicon_valid & <view>_valid (<view>_labels is n/a "
                                           "for a few of these trials).",
                            "Levels": {"true": "vicon_valid and top_valid or side_valid",
                                       "false": "vicon_valid is false or neither view is valid"}},
        "valid_views": {"Description": "Semicolon-separated valid views (top, side); n/a when none."},
        "vicon_motion": {"Description": "VICON motion.tsv of the run, relative to the dataset root; n/a when the "
                                        "trial has no VICON data."},
        "ipad_motion": {"Description": "iPad end-effector motion.tsv of the run, relative to the dataset root."},
        "events": {"Description": "events.tsv of the run, relative to the dataset root."},
    }
    for v, name in (("top", "top camera"), ("side", "side camera")):
        ok = ("The clip is present, the video match is unambiguous, the video covers the whole movement window, "
              "the trial's alignment error is < 0.1 s, the fitted clock drift is plausible (|slope - 1| <= 0.002) "
              "and the clip duration matches the movement window within 2 frames.")
        err = "Absolute residual of the trial's video-to-game offset from the offset line fitted to the video."
        d[f"{v}_valid"] = {"Description": f"Video of the {name} aligned and usable. " + ok, "Levels": valid}
        d[f"{v}_invalid_reason"] = {"Description": reason.format(f"{v}_valid") + " 'no clip': no video of the trial."}
        d[f"{v}_lead_black_frames"] = {"Description": f"Leading black frames of the {name} clip: the source video "
                                                      "started after the movement start, so frames 0 to this number "
                                                      "minus 1 have no image (labels2d and keypoints2d have no rows "
                                                      "for them); 0 for most clips; n/a without a clip."}
        d[f"{v}_align_err"] = {"Description": err, "Units": "s"}
        d[f"{v}_n_frames"] = {"Description": f"Frames of the {name} clip (movement window)."}
        d[f"{v}_fps"] = {"Description": f"Nominal frame rate of the {name} source video: frame k of the clip is at "
                                        "run time k / fps. The MP4 header may round it (29.9968 -> 30).", "Units": "Hz"}
        blurred = ("unblurred (recorded from behind and above the participant)" if v == "top" else
                   "with the heads of all persons and the laptop screens blurred")
        d[f"{v}_video"] = {"Description": f"Clip of the {name}, {blurred}, relative to the dataset root "
                                          f"({common.VIDEOS}/); n/a when there is no clip."}
    for v in CAMERA_VIEWS:
        d[f"{v}_labels"] = {"Description": f"2D labels (labels2d) of the {v} clip, relative to the dataset root; "
                                           "n/a unless the clip exists, its video is calibrated (cameras/) and the "
                                           "trial has VICON data."}
    return d


def write_manifest(root):
    t = manifest_table()
    sidecar = manifest_sidecar()
    missing = set(t.columns) ^ set(sidecar)
    if missing:
        raise ValueError(f"manifest columns without description or vice versa: {sorted(missing)}")
    write_tsv(t, root / "manifest.tsv")
    write_json({c: sidecar[c] for c in t.columns}, root / "manifest.json")
    return [root / "manifest.tsv", root / "manifest.json"]


# ----------------------------------------------------------------------------- splits, description
def write_splits(root):
    s = common.benchlib.splits()
    out = {"note": SPLITS_NOTE, "test": map_ids(s["test"]), "folds": map_ids(s["folds"]),
           "cross_visit": {"train_session": common.ses_id(s["cross_visit"]["train_visit"]),
                           "test_session": common.ses_id(s["cross_visit"]["test_visit"])}}
    write_json(out, root / "splits.json")
    return [root / "splits.json"]


def write_description(root):
    write_json({"Name": f"{common.DATASET_NAME} derivatives", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "ViKi-Traj alignment and benchmark pipeline", "Version": common.VERSION,
                                 "Description": "Trajectory-VICON-video alignment and validity (manifest), camera "
                                                "calibration, projected 2D labels, model keypoints, person boxes, "
                                                "3D tracks, result tables and fine-tuned checkpoints of the "
                                                "ViKi-Traj benchmark."}],
                "SourceDatasets": [{"URL": "../../", "Version": common.VERSION}]},
               root / "dataset_description.json")
    (root / ".bidsignore").write_text("\n".join(BIDSIGNORE) + "\n")
    (root / "README.md").write_text(DERIV_README)
    return [root / "dataset_description.json", root / ".bidsignore", root / "README.md"]


BIDSIGNORE = ["manifest.tsv", "manifest.json", "splits.json", "README.md", "cameras/", "labels2d/", "keypoints2d/",
              "boxes/", "tracks3d/", "results/", "checkpoints/"]
DERIV_README = """# ViKi-Traj derivatives (vikitraj)

A BIDS derivative dataset by its `dataset_description.json` only: the tables below use a custom layout, listed in
`.bidsignore`. Each table folder has a JSON sidecar at its top describing the columns. See `../../README.md`.

- `manifest.tsv/.json`: one row per trial (run): validity of every stream with the reason, alignment residuals,
  clip frames, and the paths of the trial's files.
- `splits.json`: test participants, development folds 1-5, cross-visit split.
- `cameras/`: projection matrix of every calibrated top and side video (VICON lab frame to image pixels).
- `labels2d/`: per clip frame, the VICON joints and markers projected into the image, with visibility.
- `keypoints2d/<model>/`: per clip frame, the 2D keypoints of each pose model of the paper.
- `boxes/`: per clip frame, the person box the benchmark used to crop the participant.
- `tracks3d/`: two-view 3D tracks of the active arm (VICON and models), mm, lab frame.
- `results/`: the paper's result tables (`results/README.md` defines the columns).
- `checkpoints/`: the ViconTraj-MS fine-tuned weights with their configs (`checkpoints/README.md`).

The video clips are in `../../sourcedata/videos/`.
"""


# ----------------------------------------------------------------------------- cameras
CAMERA_NOTE = ("ProjectionMatrix P maps the VICON laboratory frame (mm, the frame of the VICON motion files) to "
               "image pixels of the clips of this view (x right, y down, origin top-left, full image size): "
               "[u w, v w, w]^T = P [X, Y, Z, 1]^T; no lens distortion. Calibration uses the NumValidTrials trials of "
               "the video that are vicon_valid and valid in this view, with VICON markers matched to white marker "
               "blobs. SplitFitReprojectionErrorPx: a camera fitted (DLT and robust refinement) on alternate trials "
               "(alternate frames when the video has one valid trial) and tested on the others: TrainMedian on its "
               "fitting pairs, HeldOutMedian and HeldOutP90 on the NumHeldOutPairs pairs of the other trials. "
               "ProjectionMatrix is then refitted on all NumMarkerBlobPairs pairs, so the held-out errors estimate the "
               "accuracy of the procedure, not of this matrix on unseen data. CalibrationOK = HeldOutMedian <= 8 px "
               "with NumHeldOutPairs >= 30 and at least 99 % of the video's VICON marker samples in front of the "
               "camera (MarkersInFrontShare); labels2d and tracks3d use only cameras with CalibrationOK. "
               "PixelsPerMm: median local image scale of ProjectionMatrix at the "
               "matched markers. Runs: every run (valid or not) whose clip of this view comes from the calibrated "
               "video. A session has one entry per calibrated source video.")


def camera_entry(c, runs):
    P = np.load(BENCH / "cameras" / f"{c.camera}.npz")["P"]
    return {"Runs": runs, "ImageWidth": int(c.width), "ImageHeight": int(c.height), "ProjectionMatrix": P.tolist(),
            "PixelsPerMm": round(c.px_per_mm_median, 4), "CalibrationOK": usable(c),
            "MarkersInFrontShare": round(float(in_front_share(c.pid, c.camera)), 4),
            "NumValidTrials": int(c.n_paths), "NumMarkerBlobPairs": int(c.n_blob_pairs),
            "SplitFitReprojectionErrorPx": {"TrainMedian": round(c.reproj_train_median_px, 2),
                                            "HeldOutMedian": round(c.reproj_test_median_px, 2),
                                            "HeldOutP90": round(c.reproj_test_p90_px, 2)},
            "NumHeldOutPairs": int(c.n_test_pairs)}


def write_cameras(root, pids):
    d = root / "cameras"
    m, cal = trials(), calibration()
    written = []
    for pid in pids:
        for old in d.glob(f"{sub_id(pid)}_*_camera.json"):
            old.unlink()
        for (visit, view), g in cal[cal.pid == pid].groupby(["visit", "view"]):
            entries = []
            for c in g.itertuples():
                runs = m[(m.pid == pid) & (m.visit == visit) & (m[f"{view}_source_video"] == c.video)].run
                entries.append(camera_entry(c, sorted(runs)))
            sub, ses = sub_id(pid), common.ses_id(visit)
            f = d / f"{sub}_{ses}_acq-{view}_camera.json"
            write_json({"View": view, "Description": CAMERA_NOTE, "Cameras": entries}, f)
            written.append(f)
    return written


# ----------------------------------------------------------------------------- results
def released_table(t, name):
    """A result table with pid -> participant_id, (pid, row[, visit]) -> session, run, the manifest's column names
    and values (hand left | right, arm_role, n_trials, movement_time), booleans true | false and '' -> n/a."""
    t = t.rename(columns=RESULT_COLUMNS)
    if name in ARM_ROLE_TABLES:
        t = t.rename(columns={"limb": "arm_role"})
    if "hand" in t:
        t["hand"] = t.hand.str.lower()
    if "row" in t:
        key = trials().set_index(["pid", "aligned_mat_row"])[["session", "run"]]
        sr = key.reindex(pd.MultiIndex.from_arrays([t.pid, t.row.astype(int)]))
        if sr.isna().any().any():
            raise ValueError("result rows without a trial in the manifest")
        t = t.drop(columns=["row", "visit"], errors="ignore")
        t.insert(0, "run", sr.run.values)
        t.insert(0, "session", sr.session.values)
    if "pid" in t:
        t.insert(0, "participant_id", t.pop("pid").map(sub_id))
    return t.replace({"": "n/a", "True": "true", "False": "false"})


# Model identifiers of the result tables without keypoints2d / checkpoints (write_results checks the list is complete).
ABLATIONS = [
    (("vitpose_l", "vitpose_h"), "ViTPose+-L and ViTPose+-H zero-shot (larger backbones of vitpose_b)"),
    (("blazepose_noswap", "blazepose_topswap"), "BlazePose+Hands with its own left / right labels, and with left / "
                                                "right swapped in the top view (`blazepose` takes them from ViTPose+)"),
    (("ft_b_full_pooled",), "ViTPose+-B fine-tuned on the marker labels only, without the trajectory term"),
    (("ft_b_full_pooled_trajr",), "trajectory label replacing the fingertip label on every frame ('replace'; the "
                                  "released `ft_b_full_pooled_traj` uses it only where the fingertip marker is "
                                  "hidden, 'fill')"),
    (("ft_b_full_pooled_hasmask",), "as `ft_b_full_pooled`, trained on every joint with VICON data ('all labels'), "
                                    "not only visible ones"),
    (("ft_b_full_pooled_n4", "ft_b_full_pooled_n8", "ft_b_full_pooled_n12"),
     "as `ft_b_full_pooled`, trained on 4, 8 or 12 of the 17 training participants (E6b)"),
    (("ft_b_full_pooled_shift67", "ft_b_full_pooled_shift200"),
     "as `ft_b_full_pooled`, trained on labels shifted by +67 or +200 ms (E6c)"),
    (("ft_b_full_top", "ft_b_full_side"), "as `ft_b_full_pooled`, trained on one view only (`_pooled`: both views)"),
    (("ft_b_head_pooled", "ft_b_head_top", "ft_b_head_side"),
     "as `ft_b_full_pooled` / `_top` / `_side`, with only the decoder fine-tuned"),
    (("seq_vitpose_b", "seq_vitpose_l", "seq_vitpose_h", "seq_sapiens2_0.4b", "seq_sapiens2_1b",
      "seq_ft_b_full_pooled_traj", "seq_ft_sapiens2_0.4b_head", "seq_ft_sapiens2_1b_head"),
     "the temporal head on other base models (E5 ablation)"),
]


def model_identifiers(name, t):
    """Model identifiers of a source table: `variant`, and `source` in the E4 tables (elsewhere a display name)."""
    cols = ["variant"] + (["source"] if name.startswith("E4") else [])
    return {v for c in cols if c in t for v in t[c]} - {"", "vicon"}


def results_readme():
    released = ", ".join(f"`{m}`" for m in common.ZERO_SHOT)
    method = ", ".join(f"`{m}`" for m in common.WITH_METHOD)
    lines = ["# ViKi-Traj result tables", "",
             "The tables behind the paper's tables and figures, one TSV per experiment output (`n/a` = not computed: "
             "too few samples or not applicable). Errors are in mm unless the column says px or deg.", "",
             "## Files", ""]
    lines += [f"- `{RESULT_FILE_NAMES.get(name, name)}.tsv`: {desc}." for name, desc in RESULTS.items()]
    lines += ["", "## Columns", "",
              "- `participant_id`, `session`, `run`, `group`, `hand`, `arm_role`, `movement_time`: as in "
              "manifest.tsv.",
              "- `split`: `test` (the 8 test participants of splits.json) or `all`; `group` in summaries also `all`. "
              "For the ViconTraj-MS variants (`ft_*`, `seq_*` and their ablations) `all` includes the participants of "
              "the training folds 2-5 and the early-stopping fold 1, whose outputs are not held out; use `test`.",
              "- `view`: `top`, `side` (2D) or `3d` (two-view triangulation).",
              "- `role`: whether the evaluated arm played the trial (`active`) or not (`resting`).",
              "- `labels`: `visible` (frames where every marker of the joint has a white blob within 8 px) or `all` "
              "(every frame with VICON data).",
              "- `n_trials`, `n_participants`, `n_labels`: trials, participants and labelled frames behind a value "
              "(`n_trials` and the `*_trials` files are called paths in the benchmark code).",
              "", "## Model identifiers", "",
              "`variant` (and `source` in the E4 tables, where `vicon` is the VICON reference) is a model "
              "identifier; `model` is its display name. In E3 and E6b `source` is the display name followed by "
              "`, top` (top-view keypoint mapped to the screen by a homography) or `, 2 views` (triangulated, affine "
              "map); `Vicon FIN` is the VICON fingertip marker, and `wrist` marks the ViTPose+ zero-shot models, "
              "which have no fingertip keypoint.", "",
              f"- With keypoints2d/: the zero-shot baselines {released}.",
              f"- With keypoints2d/ and checkpoints/: the ViconTraj-MS models of the paper, {method}.",
              "- Ablations, in the result tables only (no keypoints, no weights):"]
    lines += ["  - " + ", ".join(f"`{i}`" for i in ids) + f": {desc}." for ids, desc in ABLATIONS]
    return "\n".join(lines) + "\n"


def write_results(root):
    d = root / "results"
    shutil.rmtree(d, ignore_errors=True)
    written, ids = [], set()
    for name in RESULTS:
        t = pd.read_csv(BENCH / "results" / f"{name}.csv", dtype=str, keep_default_na=False)
        ids |= model_identifiers(name, t)
        f = d / f"{RESULT_FILE_NAMES.get(name, name)}.tsv"
        write_tsv(released_table(t, name), f)
        written.append(f)
    undocumented = ids - set(common.ZERO_SHOT + common.WITH_METHOD) - {i for a, _ in ABLATIONS for i in a}
    if undocumented:
        raise ValueError(f"model identifiers missing from the results README: {sorted(undocumented)}")
    (d / "README.md").write_text(results_readme())
    return written + [d / "README.md"]


# ----------------------------------------------------------------------------- checkpoints
def checkpoint_sources(run):
    """{released file name: source file} of one ViconTraj-MS model; {} when its weights are missing."""
    if run.startswith("seq_"):
        files = {}
        for v in CAMERA_VIEWS:
            files[f"model_{v}.pt"] = FINETUNE / f"{run}_{v}" / "model.pt"
            files[f"log_{v}.tsv"] = FINETUNE / f"{run}_{v}" / "log.csv"
    else:
        files = {p.name: p for p in sorted((FINETUNE / run).glob("*.pt"))}
        files["log.tsv"] = FINETUNE / run / "log.csv"
    weights = [f for n, f in files.items() if n.endswith(".pt")]
    return files if weights and all(f.exists() for f in weights) else {}


def base_model(run, cfg):
    if run.startswith("seq_"):
        return f"{common.MODEL_NAMES[cfg['base']]} keypoints (keypoints2d/{cfg['base']})"
    if cfg.get("kp_layout") == "sapiens2":
        size = cfg["size"]
        return (f"Sapiens2-{size.upper()} pose model (308 keypoints, config sapiens2_{size}_keypoints308_shutterstock_"
                f"goliath_3po-1024x768); head.pt is the trained decode head, the backbone is the original, frozen")
    import finetune                                       # benchmark/finetune.py: Hugging Face ids of the backbones
    return (f"{finetune.BACKBONES[cfg.get('backbone', 'base')]} (transformers VitPoseForPoseEstimation) with the "
            "decoder's last 1x1 convolution replaced by the 10 joint-set channels; model.pt is the full state dict")


def output_channels(run, cfg):
    if run.startswith("seq_"):
        return {f"{j}_{a}": 2 * i + k for i, j in enumerate(common.JOINT_SET) for k, a in enumerate("xy")}
    kp = common.benchlib.KP["sapiens2_1b"] if cfg.get("kp_layout") == "sapiens2" else common.benchlib.FT_KP
    return dict(zip(common.JOINT_SET, kp["L"] + kp["R"]))


def usage(run, cfg):
    """{inference, log_units} of a checkpoint: its input and decoding, and the units of its training log."""
    if run.startswith("seq_"):
        return {"inference": SEQ_INFERENCE, "log_units": SEQ_LOG_UNITS}
    sapiens = cfg.get("kp_layout") == "sapiens2"
    units = {k: v.format("768 x 1024" if sapiens else "192 x 256") for k, v in FT_LOG_UNITS.items()}
    if not sapiens:
        units["stage"] = "head: decoder only trained; full: the whole network"
    return {"inference": FT_SAPIENS_INFERENCE if sapiens else FT_VITPOSE_INFERENCE, "log_units": units}


def checkpoint_config(run, files):
    cfg = json.loads((FINETUNE / run / "config.json").read_text())
    cfg = {k: map_ids(v) for k, v in cfg.items() if not (isinstance(v, str) and re.search(r"[/\\]", v))}
    out = {"name": common.MODEL_NAMES[run], "base_model": base_model(run, cfg),
           "files": sorted(files), "output_channels": output_channels(run, cfg), **usage(run, cfg)}
    if run.startswith("seq_"):
        folds = common.benchlib.splits()["folds"]
        out["architecture"] = SEQ_ARCHITECTURE
        cfg.update(train_participants=map_ids([p for f in folds[1:] for p in f]), val_participants=map_ids(folds[0]))
    return {**out, **cfg}


def checkpoints_readme(runs):
    lines = ["# ViconTraj-MS checkpoints", "",
             "Fine-tuned weights of the models scored in the paper (PyTorch state dicts, `torch.load(f, "
             "map_location='cpu')`), trained on folds 2-5 of splits.json with fold 1 for early stopping; the test "
             "participants were never seen. Each folder has the weights, the training log and `config.json`:", "",
             "- `base_model`, `output_channels` (joint set order left / right shoulder, elbow, wrist, hand, "
             "fingertip), training settings and participants;",
             "- `inference`: the input crop and its normalisation, the forward call and the decoding. ViTPose+-B "
             "needs `dataset_index=0` (the COCO expert of ViTPose+); `seq_*` heads take keypoints2d/<base> "
             "normalised by the person box (`architecture`);",
             "- `log_units`: the logs' validation errors are in px of the model crop for `ft_*` (192 x 256 for "
             "ViTPose+-B, 768 x 1024 for Sapiens2), not image px, and in box-normalised units for `seq_*`.", "",
             "**Person boxes.** Every checkpoint takes the participant's person box per frame (the crop of the "
             "`ft_*` models, the coordinate normalisation of the `seq_*` heads). The boxes the paper used are in "
             "`../boxes/` (one table per clip, rows aligned with keypoints2d); `inference.person_box` describes how "
             "they were made. The `seq_*` heads also used the frames up to 0.5 s beyond each end of the clip, so "
             "applying them to keypoints2d/ reproduces the released `seq_*` keypoints closely but not exactly.", ""]
    lines += [f"- `{r}/`: {common.MODEL_NAMES[r]}." for r in runs]
    return "\n".join(lines) + "\n"


def write_checkpoints(root):
    d = root / "checkpoints"
    shutil.rmtree(d, ignore_errors=True)
    written, runs = [], []
    for run in common.WITH_METHOD:
        files = checkpoint_sources(run)
        if not files:
            print(f"[checkpoints] {run}: no weights, skipped", flush=True)
            continue
        for name, src in files.items():
            dst = d / run / name
            if dst.suffix == ".tsv":
                write_tsv(pd.read_csv(src, dtype=str, keep_default_na=False).replace("", "n/a"), dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            written.append(dst)
        write_json(checkpoint_config(run, files), d / run / "config.json")
        written.append(d / run / "config.json")
        runs.append(run)
    (d / "README.md").write_text(checkpoints_readme(runs))
    size = sum(f.stat().st_size for f in written)
    print(f"[checkpoints] {len(runs)} models, {size / 1e6:.1f} MB", flush=True)
    return written + [d / "README.md"]


# ----------------------------------------------------------------------------- de-identification check
def forbidden_text():
    terms = []
    f = common.PRIVATE / "scrub_terms.txt"
    if f.exists():
        terms = [t.strip() for t in f.read_text().splitlines() if t.strip() and not t.startswith("#")]
    return re.compile("|".join(FORBIDDEN_TEXT + [rf"\b{re.escape(t)}\b" for t in terms]), re.I)


def binary_hits(path, chunk=1 << 26):
    with open(path, "rb") as fh:
        tail = b""
        while block := fh.read(chunk):
            if FORBIDDEN_BINARY.search(tail + block):
                return True
            tail = block[-16:]
    return False


def check_deidentified(paths, out):
    text = forbidden_text()
    bad = []
    for p in paths:
        hits = text.findall(str(p.relative_to(out)))
        if p.suffix == ".pt":
            hits += ["binary match"] if binary_hits(p) else []
        else:
            hits += text.findall(p.read_text())
        if hits:
            bad.append(f"{p.relative_to(out)}: {sorted(set(hits))[:5]}")
    if bad:
        raise SystemExit("identifying content in released files:\n  " + "\n  ".join(bad))


# ----------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--participant", help="study ID: write the cameras of this participant only")
    g.add_argument("--all", action="store_true", help="write everything")
    g.add_argument("--global", dest="global_only", action="store_true",
                   help="write the global parts only (description, manifest, splits, results, checkpoints)")
    ap.add_argument("--out", type=Path, default=common.RELEASE)
    args = ap.parse_args()
    if args.participant and args.participant not in common.pids():
        ap.error(f"unknown participant {args.participant}")
    return args


def main():
    args = parse_args()
    root = args.out / DERIV
    written = []
    if not args.global_only:
        written += write_cameras(root, common.pids() if args.all else [args.participant])
    if not args.participant:
        written += write_description(root) + write_manifest(root) + write_splits(root) + write_results(root)
        written += write_checkpoints(root)
    check_deidentified(written, args.out)
    print(f"[derivatives] {len(written)} files written under {root}, de-identification check passed", flush=True)


if __name__ == "__main__":
    main()
