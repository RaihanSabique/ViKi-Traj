"""Per-frame derivative tables of the release: labels2d, keypoints2d, boxes and tracks3d.

Writes TSV tables with JSON sidecars under <out>/derivatives/vikitraj/: labels2d, keypoints2d/<model> and boxes per
clip (one row per frame of the released video), tracks3d per trial. A re-run replaces the participants' files.

    python dataset/build_frame_tables.py --participant XXXXXXX103 [--out release/ViKi-Traj]
    python dataset/build_frame_tables.py --all
"""
import shutil
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "benchmark"))     # ahead of src/, which common.py puts first, as benchlib expects
import calibrate as CAL  # noqa: E402
import e4_kinematics as E4  # noqa: E402
import evaluate as E  # noqa: E402
import benchlib as B  # noqa: E402

MODELS = C.ZERO_SHOT + C.WITH_METHOD
SCORELESS = {m for m in MODELS if m.startswith("seq_")}       # temporal heads store a constant score of 1
LABEL_VIEWS = ("top", "side")
TABLES = ("labels2d", "keypoints2d", "boxes", "tracks3d")
MIN_IN_FRONT = 0.99                # share of a video's VICON marker samples that must lie in front of its camera


# ----------------------------------------------------------------------------- formatting
def fmt(a, decimals):
    """Strings with a fixed number of decimals, n/a for NaN (no negative zero)."""
    a = np.round(np.asarray(a, float), decimals) + 0.0
    return np.where(np.isnan(a), "n/a", np.char.mod(f"%.{decimals}f", a))


def point_columns(names, xy, third, suffix, decimals):
    """{<name>_x, <name>_y, <name>_<suffix>} for points (T, K, 2) and a per-point value (T, K)."""
    fxy, f3 = fmt(xy, 2), fmt(third, decimals)
    cols = {}
    for k, name in enumerate(names):
        cols[f"{name}_x"], cols[f"{name}_y"], cols[f"{name}_{suffix}"] = fxy[:, k, 0], fxy[:, k, 1], f3[:, k]
    return cols


# ----------------------------------------------------------------------------- clips and clocks
def clip_frames(r, view):
    """Source-video frame shown by every frame of the run's clip; None when the run has no clip of this view.
    align_videos.cut_clip seeks with ffmpeg -ss max(start_s, 0) (4 decimals): the first source frame at or after the
    seek time is output frame 0 when it lies less than half a frame after it, else output frame 1, and the
    constant-rate output repeats it as frame 0. Apart from that frame, clip frame f is source frame
    max(<view>_start_frame, 0) + f, except in one clip whose start_s is exactly half a frame (frames 1 on are one frame
    earlier)."""
    p = r[f"{view}_clip_path"]
    if not isinstance(p, str) or not (C.ROOT / p).exists():
        return None
    cap = cv2.VideoCapture(str(C.ROOT / p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    s = B.segments().loc[(r.pid, int(r.aligned_mat_row), view)]
    seek = round(max(float(s.start_s), 0.0), 4) * float(s.fps)          # in source frames
    first = int(np.ceil(seek - 1e-6))
    late = first - seek > 0.5 - 1e-6
    frames = first - int(late) + np.arange(n)
    frames[:1] = first
    return frames


def video_frames(r, view):
    """(released video frame indices, source frames) of the video frames that show an image (None, None without a
    clip): video frame k shows clip frame k + start_frame - max(start_frame, 0), as blur_clips.clips() writes it."""
    frames = clip_frames(r, view)
    if frames is None:
        return None, None
    n, s0 = int(r[f"{view}_n_frames"]), int(r[f"{view}_start_frame"])
    c = np.arange(n) + s0 - max(s0, 0)
    ok = (c >= 0) & (c < len(frames))
    return np.flatnonzero(ok), frames[c[ok]]


def run_time(r, view, frames):
    """Run time (s) of source-video frames of the run's video."""
    return B.frame_clock(r.pid, int(r.aligned_mat_row), view, frames) - C.run_t0(r.pid, r.aligned_mat_row)


# ----------------------------------------------------------------------------- cameras
def video_camera(r, view):
    """Camera id of the run's video of this view (None without a video)."""
    video = r[f"{view}_source_video"]
    return CAL.camera_id(r.pid, int(r.visit), view, video) if isinstance(video, str) else None


def depth(P, X):
    """Projective depth of points (N, 3) under P (sign = side of the focal plane)."""
    return X @ P[2, :3] + P[2, 3]


@lru_cache(None)
def camera_checks(pid):
    """{camera id: (P, share, 'ses-N view')} of the participant's calibrated top / side videos with VICON trials:
    P signed so that depth > 0 on the side of the majority of the video's VICON marker samples (in front), and the
    share of those samples in front."""
    cams, vic = E.cameras(), B.markers(pid)[0]
    samples = {}
    for _, r in C.manifest().query("pid == @pid").iterrows():
        for view in LABEL_VIEWS:
            cid = video_camera(r, view)
            if cid in cams and int(r.aligned_mat_row) in vic:
                X = vic[int(r.aligned_mat_row)]["X"].reshape(-1, 3)
                samples.setdefault((cid, f"{C.ses_id(r.visit)} {view}"), []).append(X[~np.isnan(X).any(1)])
    out = {}
    for (cid, name), xs in samples.items():
        d = depth(cams[cid], np.concatenate(xs))
        front = np.sign(np.median(d))
        out[cid] = (cams[cid] * front, float(np.mean(d * front > 0)), name)
    return out


def usable_cameras(pid):
    """{camera id: P} of the cameras with at least MIN_IN_FRONT of their marker samples in front (camera_checks). A
    calibration fitted to the blobs of another person can pass its reprojection test with the child behind the focal
    plane; such a camera gets no labels."""
    return {cid: P for cid, (P, share, _) in camera_checks(pid).items() if share >= MIN_IN_FRONT}


def project(P, X):
    """(..., 3) mm -> (..., 2) px with vicon_eval.project (as labels.py); NaN for missing points and points behind the
    camera (P signed as in camera_checks)."""
    flat = X.reshape(-1, 3)
    xy = E.project(P, flat)
    xy[~(depth(P, flat) > 0)] = np.nan
    return xy.reshape(X.shape[:-1] + (2,))


# ----------------------------------------------------------------------------- labels2d
def joint_reference(v, side, joint):
    """(N, 3) 6 Hz low-passed VICON proxy (evaluate.ref_track); NaN when the trial lacks one of its markers."""
    if any(side + n not in v["labels"] for n in B.JOINTS[joint]):
        return np.full((len(v["t"]), 3), np.nan)
    return E.ref_track(v, side, joint)


def marker_positions(v, t):
    """(T, 24, 3) gap-filled markers at absolute times t in common.MARKERS order; NaN where missing."""
    Xr = CAL.interp_markers(v, t)
    X = np.full((len(t), len(C.MARKERS), 3), np.nan)
    for c, name in enumerate(C.MARKERS):
        if name in v["labels"]:
            X[:, c] = Xr[:, v["labels"].index(name)]
    return X


def blob_records(r, view):
    """{source frame: (B, 2) white blob centres} of the clip (outputs/benchmark/blobs); empty without a blob file."""
    f = B.aux_file("blobs", r.pid, view, B.clip_stem(r, view))
    if not f.exists():
        return {}
    z = np.load(f)
    return dict(zip(z["frame"].tolist(), np.split(z["xy"], np.cumsum(z["n"])[:-1])))


def marker_visibility(blobs, frames, mxy):
    """(T, 24): 1 = a blob within VIS_PX of the projected marker, 0 = none; NaN without a blob record or a position."""
    vis = np.full(mxy.shape[:2], np.nan)
    for i, fr in enumerate(frames):
        xy = blobs.get(int(fr))
        if xy is None:
            continue
        ok = ~np.isnan(mxy[i]).any(1)
        vis[i, ok] = 0.0
        if len(xy) and ok.any():
            vis[i, ok] = np.linalg.norm(mxy[i][ok][:, None] - xy[None], axis=2).min(1) <= E.VIS_PX
    return vis


def joint_visibility(mvis, record, jxy):
    """(T, 10): 1 = every marker of the joint visible (a marker without a position is not, as in labels.py), 0 = not;
    NaN where the frame has no blob record or the joint no position."""
    idx = [[C.MARKERS.index(s + n) for n in B.JOINTS[j]] for s in "LR" for j in B.JOINT_NAMES]
    jv = np.stack([(mvis[:, i] == 1).all(1) for i in idx], 1).astype(float)
    jv[~record] = np.nan
    jv[np.isnan(jxy).any(2)] = np.nan
    return jv


def label_arrays(r, view, P, v, frames):
    """Projected joints (T, 10, 2), markers (T, 24, 2) and their visibility at the clip frames."""
    t = B.frame_clock(r.pid, int(r.aligned_mat_row), view, frames)
    joints = np.stack([E.at(v["t"], joint_reference(v, s, j), t) for s in "LR" for j in B.JOINT_NAMES], 1)
    jxy, mxy = project(P, joints), project(P, marker_positions(v, t))
    blobs = blob_records(r, view)
    record = np.array([int(f) in blobs for f in frames], bool)
    mvis = marker_visibility(blobs, frames, mxy)
    return jxy, mxy, joint_visibility(mvis, record, jxy), mvis


def label_table(r, view, P, v, frames, k):
    jxy, mxy, jvis, mvis = label_arrays(r, view, P, v, frames)
    cols = {"frame": k, "time": fmt(run_time(r, view, frames), 4)}
    cols.update(point_columns(C.JOINT_SET, jxy, jvis, "visible", 0))
    cols.update(point_columns(C.MARKERS, mxy, mvis, "visible", 0))
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------- keypoints2d
@lru_cache(None)
def benchmark_lr_table():
    """The benchmark's per-video left/right decision of the side-checked models over the valid clips
    (benchlib.lr_swap_table). Read from models/<model>/lr_swap_by_video.csv: lr_swap_table itself rewrites that file."""
    table = {}
    for m in B.SIDE_CHECKED:
        t = pd.read_csv(C.BENCH / "models" / m / "lr_swap_by_video.csv")
        table.update({(m, x.pid, x.view, x.video): bool(x.swapped) for x in t.itertuples()})
    return table


@lru_cache(None)
def lr_swapped(model, pid, view, video):
    """Whether a side-checked model's left/right labels are swapped in a video: the benchmark's decision, or for a
    video without valid clips the same rule (median benchlib.lr_margin < 0) over every clip of the video with outputs
    of the model and its reference. None without such a clip (benchlib.model_joints then uses MIRRORED_VIEWS)."""
    if (model, pid, view, video) in benchmark_lr_table():
        return benchmark_lr_table()[(model, pid, view, video)]
    ref, margins = B.SIDE_CHECKED[model], []
    for _, r in C.manifest().query("pid == @pid").iterrows():
        files = [B.model_file(m, pid, view, B.clip_stem(r, view)) for m in (model, ref)]
        if r[f"{view}_source_video"] == video and all(f.exists() for f in files):
            margins.append(B.lr_margin(np.load(files[0]), np.load(files[1]), model, ref))
    margins = np.array(margins, float)
    margins = margins[~np.isnan(margins)]
    return bool(np.median(margins) < 0) if len(margins) else None


def load_model(model, r, view):
    """The model's output for the clip with its left/right decision (as benchlib.set_lr_swap)."""
    z = dict(np.load(B.model_file(model, r.pid, view, B.clip_stem(r, view))))
    if model in B.SIDE_CHECKED:
        swapped = lr_swapped(model, r.pid, view, r[f"{view}_source_video"])
        if swapped is not None:
            z["lr_swapped"] = swapped
    return z


def model_side(z, model, side, view):
    """The model's own label of a body side, chosen as in benchlib.model_joints (mirror 'auto')."""
    swap = bool(z["lr_swapped"]) if "lr_swapped" in z else view in B.MIRRORED_VIEWS.get(model, ())
    return {"L": "R", "R": "L"}[side] if swap else side


def model_keypoints(z, model, side, view):
    """(T, 5, 2) positions and (T, 5) scores of a body side: the keypoints benchlib.model_joints takes (same mapping
    and side decision) without its score threshold; NaN where the model gives no position or no score."""
    own = model_side(z, model, side, view)
    xy = np.full((len(z["frame"]), len(B.JOINT_NAMES), 2), np.nan)
    score = np.full(xy.shape[:2], np.nan)
    for j, idx in enumerate(B.KP[model][own]):
        if isinstance(idx, str):
            xy[:, j] = z[f"hand_{own}"][:, int(idx[1:])]
            if f"hand_s_{own}" in z:
                score[:, j] = z[f"hand_s_{own}"][:, int(idx[1:])]
        elif idx != -1:
            xy[:, j], score[:, j] = z["k"][:, idx], z["s"][:, idx]
    if model in SCORELESS:
        score[:] = np.nan
    return xy, score


def keypoint_table(model, r, view, frames, k):
    z = load_model(model, r, view)
    pos = np.searchsorted(z["frame"], frames)
    done = (pos < len(z["frame"])) & (z["frame"][np.minimum(pos, len(z["frame"]) - 1)] == frames)
    sides = [model_keypoints(z, model, s, view) for s in "LR"]
    xy = np.concatenate([p for p, _ in sides], 1)[pos[done]]
    score = np.floor(np.concatenate([s for _, s in sides], 1)[pos[done]] * 1e4) / 1e4   # truncated: >= THR as in benchlib
    cols = {"frame": k[done], "time": fmt(run_time(r, view, frames[done]), 4)}
    cols.update(point_columns(C.JOINT_SET, xy, score, "score", 4))
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------- boxes
def box_table(r, view, frames, k):
    """Person box of the participant per released video frame (outputs/benchmark/boxes: the benchmark's crop box for
    every model); None without a box file."""
    f = B.aux_file("boxes", r.pid, view, B.clip_stem(r, view))
    if not f.exists():
        return None
    z = np.load(f)
    pos = np.searchsorted(z["frame"], frames)
    done = (pos < len(z["frame"])) & (z["frame"][np.minimum(pos, len(z["frame"]) - 1)] == frames)
    box = z["box"][pos[done]]
    cols = {"frame": k[done], "time": fmt(run_time(r, view, frames[done]), 4)}
    cols.update({c: fmt(box[:, i], 2) for i, c in enumerate(("x0", "y0", "x1", "y1"))})
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------- tracks3d
def tracks_table(r, f):
    """Long table (time, source, joint, x, y, z) of a run's two-view tracks (benchmark/e4_kinematics.py, 50 Hz on the
    VICON clock); a model has no rows for a joint it does not estimate."""
    z = np.load(f)
    side = str(r.hand)[0]
    time = fmt(z["t"] - C.run_t0(r.pid, r.aligned_mat_row), 4)
    blocks = []
    for src in ["vicon"] + [m for m in MODELS if m in z.files]:
        for a, joint in enumerate(E4.J4):
            if src != "vicon" and B.KP[src][side][E4.JIDX[joint]] == -1:
                continue
            X = fmt(z[src][a], 2)
            blocks.append(pd.DataFrame({"time": time, "source": src, "joint": joint,
                                        "x": X[:, 0], "y": X[:, 1], "z": X[:, 2]}))
    return pd.concat(blocks, ignore_index=True)


# ----------------------------------------------------------------------------- sidecars
FRAME_COL = {"Description": "Index of the frame in the run's released clip of this view (videos/), from 0; frame 0 "
                            "is the movement start. Where the source video began after the movement start, the clip "
                            "starts with black frames (manifest <view>_lead_black_frames) and the first row is the "
                            "first frame with an image; frames without an image have no row. A row describes the "
                            "image of its frame: where a clip's first image repeats the next one (a seek artefact of "
                            "the clip cut), the two rows are equal."}
TIME_COL = {"Description": "Capture time of the frame: run time, seconds since the movement start, through the "
                           "video-game clock alignment (manifest <view>_align_err).", "Units": "s"}
IMAGE_AXES = "image px of the full frame, origin at the top-left corner, x to the right, y down"
KEYPOINT = {"shoulder": "shoulder", "elbow": "elbow", "wrist": "wrist",
            "hand": "middle-finger knuckle (MCP, under the HND marker)", "fingertip": "index fingertip"}
HAND_UNSCORED = {"blazepose"}      # hand landmarks without a per-keypoint confidence (no hand_s_* in the outputs)
SIDE_NOTE = {"blazepose": " BlazePose labels sides as if the person faced the camera: left and right are assigned per "
                          "video by agreement with ViTPose+-B's arms (the benchmark's rule), or by the view (top "
                          "mirrored) for a video without ViTPose+-B output."}
SCORE_SCALE = {"vitpose_b": "heatmap maximum, about 0-1",
               "rtmpose_wb": "SimCC confidence, about 0-1, can exceed 1",
               "dwpose_l": "SimCC confidence, about 0-1, can exceed 1",
               "rtmw_x": "SimCC confidence, not normalised, about 0-10",
               "rtmpose_hand": "SimCC confidence of the hand model on hand crops, about 0-1",
               "sapiens2_0.4b": "heatmap maximum, about 0-1",
               "sapiens2_1b": "heatmap maximum, about 0-1",
               "blazepose": "landmark visibility, 0-1",
               "ft_b_full_pooled_traj": "heatmap maximum of the fine-tuned head, about 0-1",
               "ft_sapiens2_0.4b_head": "heatmap maximum of the fine-tuned head, about 0-1.5",
               "ft_sapiens2_1b_head": "heatmap maximum of the fine-tuned head, about 0-1.5"}


def proxy_markers(joint, prefix=""):
    """'LSHO' or 'mean of LWRA and LWRB' (prefix 'L'): the markers of a joint's VICON proxy."""
    names = [prefix + n for n in B.JOINTS[joint]]
    return names[0] if len(names) == 1 else "mean of " + " and ".join(names)


def labels_sidecar():
    d = {"frame": FRAME_COL, "time": TIME_COL}
    points = [(f"{s}_{j}", f"{s} {j} joint (VICON proxy: {proxy_markers(j, s[0].upper())}, low-pass filtered at 6 Hz)",
               "every marker of the joint has")
              for s in ("left", "right") for j in B.JOINT_NAMES]
    points += [(m, f"marker {m} ({C.MARKER_PLACEMENT[m]}; gap-filled VICON position)", "the marker has")
               for m in C.MARKERS]
    for name, what, rule in points:
        where = ("n/a where VICON has no position; projected with the camera of this view (cameras/), so it can lie "
                 "outside the image and is n/a behind the camera.")
        d[f"{name}_x"] = {"Description": f"Horizontal position of the {what}, {IMAGE_AXES}; {where}", "Units": "px"}
        d[f"{name}_y"] = {"Description": f"Vertical position of the {what}, {IMAGE_AXES}; {where}", "Units": "px"}
        d[f"{name}_visible"] = {
            "Description": f"Whether {rule} a white marker blob within 8 px of its projection. Blobs were detected on "
                           "every third source frame of the clips valid in this view; n/a on the other frames and "
                           "where the point has no position.",
            "Levels": {"0": "not visible (occluded, or no blob near the projection)", "1": "visible"}}
    d["frame"] = dict(FRAME_COL, Description=FRAME_COL["Description"] + " Labels of clips with vicon_valid or "
                      "<view>_valid false (manifest) are not verified and can be misaligned; the benchmark uses "
                      "vicon_valid & <view>_valid.")
    return d


def score_description(model, idx):
    name = C.MODEL_NAMES[model]
    if model in SCORELESS:
        return "n/a: the temporal head gives no confidence (the benchmark uses every position)."
    if isinstance(idx, str) and model in HAND_UNSCORED:
        return f"n/a: the hand landmarker of {name} gives no per-keypoint confidence (the benchmark uses every position)."
    return (f"Confidence of the keypoint as given by {name} ({SCORE_SCALE[model]}), not comparable across models; "
            f"truncated to 4 decimals. The paper's benchmark uses a position only where score >= {B.THR[model]:g}.")


def keypoints_sidecar(model):
    name, d = C.MODEL_NAMES[model], {"frame": FRAME_COL, "time": TIME_COL}
    for s in ("left", "right"):
        for j, idx in zip(B.JOINT_NAMES, B.KP[model][s[0].upper()]):
            cols = [f"{s}_{j}_{c}" for c in ("x", "y", "score")]
            if idx == -1:
                d.update({c: {"Description": f"n/a: {name} has no keypoint for the {s} {j}."} for c in cols})
                continue
            what = (f"the {s} {j} VICON proxy ({proxy_markers(j, s[0].upper())}) as estimated by {name}"
                    if model in C.WITH_METHOD else f"the {KEYPOINT[j]} keypoint of the person's {s} side from {name}")
            for c, axis in zip(cols, ("Horizontal", "Vertical")):
                d[c] = {"Description": f"{axis} position of {what}, as output (not cut at a score threshold), "
                                       f"{IMAGE_AXES}; n/a where the model gives none.{SIDE_NOTE.get(model, '')}",
                        "Units": "px"}
            d[cols[2]] = {"Description": score_description(model, idx)}
    d["frame"] = dict(FRAME_COL, Description=FRAME_COL["Description"] + " " + COVERAGE[model.split("_")[0]
                                                                                       if model[:3] in ("ft_", "seq") else "zs"])
    return d


COVERAGE = {
    "zs": "Keypoints exist for the clips valid in their view (manifest <view>_valid), except a few clips of "
          "VICON-invalid trials.",
    "ft": "Keypoints exist for the clips with vicon_valid and <view>_valid. They are the output of the single released "
          "checkpoint (checkpoints/), trained on the participants of folds 2-5 with early stopping on fold 1: only "
          "the test participants of splits.json are held out.",
    "seq": "Keypoints exist for the clips of the test participants (splits.json) with vicon_valid and <view>_valid."}


def boxes_sidecar():
    px = {"Units": "px"}
    return {"frame": FRAME_COL, "time": TIME_COL,
            "x0": dict(px, Description=f"Left edge of the participant's person box, {IMAGE_AXES}."),
            "y0": dict(px, Description=f"Top edge of the person box, {IMAGE_AXES}."),
            "x1": dict(px, Description=f"Right edge of the person box, {IMAGE_AXES}."),
            "y1": dict(px, Description=f"Bottom edge of the person box, {IMAGE_AXES}. The box is the one the "
                       "benchmark used to crop the participant for every model: RT-DETR person detections (score "
                       ">= 0.3), per frame the one with the highest overlap (IoU > 0.3) with a reference box of the "
                       "video (an experimenter is sometimes in view), gaps linearly interpolated, then a 5-frame "
                       "running median. Rows exist for the clips valid in their view, except a few clips of "
                       "VICON-invalid trials.")}


def tracks_sidecar():
    triangulated = ("two-view linear triangulation (top and side cameras, cameras/) of the model's 2D joints: "
                    "positions at or above the benchmark's score threshold, spikes removed (Hampel) and low-pass "
                    "filtered at 6 Hz in each view, interpolated to the VICON clock (n/a across gaps over 0.1 s); "
                    "rows only for the joints the model estimates")
    levels = {"vicon": "VICON joint proxies (joint), low-pass filtered at 6 Hz"}
    levels.update({m: f"{C.MODEL_NAMES[m]}: {triangulated}" for m in MODELS})
    xyz = {"Description": "Position in the VICON laboratory frame (the frame of the VICON motion files, z up); n/a "
                          "where missing.", "Units": "mm"}
    return {"time": {"Description": "Run time of the sample, seconds since the movement start; every second sample "
                                    "of the 100 Hz VICON clock (50 Hz). Only the active arm (shoulder, elbow, wrist, "
                                    "fingertip) of runs with two usable cameras is tracked. Rows of the ViconTraj-MS "
                                    "sources (ft_*, seq_*) of non-test participants are not held out (see "
                                    "keypoints2d).",
                     "Units": "s"},
            "source": {"Description": "Origin of the track.", "Levels": levels},
            "joint": {"Description": "Joint of the active arm (manifest hand).",
                      "Levels": {j: f"{j}; VICON proxy: {proxy_markers(j)} of the active arm" for j in E4.J4}},
            "x": xyz, "y": xyz, "z": xyz}


def write_sidecars(out):
    d = Path(out, C.DERIV)
    name = lambda suffix: f"task-{C.TASK}_{suffix}.json"
    C.write_json(labels_sidecar(), d / "labels2d" / name("labels"))
    for m in MODELS:
        C.write_json(keypoints_sidecar(m), d / "keypoints2d" / m / name("keypoints"))
    C.write_json(tracks_sidecar(), d / "tracks3d" / name("tracks3d"))
    C.write_json(boxes_sidecar(), d / "boxes" / name("boxes"))


# ----------------------------------------------------------------------------- per participant
def out_file(out, table, r, suffix, view=None, model=None):
    sub, ses, run = C.row_ids(r)
    base = Path(out, C.DERIV, table, *([model] if model else []))
    return C.subses_dir(base, sub, ses) / C.bids_name(sub, ses, suffix, ".tsv", acq=view, run=run)


def clear(out, sub):
    """Remove the participant's files of the four tables (a re-run replaces them)."""
    d = Path(out, C.DERIV)
    for p in [d / "labels2d" / sub, d / "tracks3d" / sub, d / "boxes" / sub] + [d / "keypoints2d" / m / sub for m in MODELS]:
        shutil.rmtree(p, ignore_errors=True)


def write(df, path, sizes, table):
    C.write_tsv(df, path)
    sizes[f"{table}_files"] += 1
    sizes[f"{table}_rows"] += len(df)
    sizes[f"{table}_bytes"] += path.stat().st_size


def build_participant(pid, out):
    """Write the participant's tables; returns the files, rows and bytes written per table."""
    clear(out, C.sub_id(pid))
    vic, cams = B.markers(pid)[0], usable_cameras(pid)
    sizes = Counter()
    for _, r in C.manifest().query("pid == @pid").iterrows():
        v = vic.get(int(r.aligned_mat_row))
        for view in LABEL_VIEWS:
            k, frames = video_frames(r, view)
            if frames is None or len(frames) == 0:
                continue
            P = cams.get(video_camera(r, view))
            if P is not None and v is not None:
                write(label_table(r, view, P, v, frames, k), out_file(out, "labels2d", r, "labels", view), sizes,
                      "labels2d")
            for model in MODELS:
                if B.model_file(model, pid, view, B.clip_stem(r, view)).exists():
                    write(keypoint_table(model, r, view, frames, k),
                          out_file(out, "keypoints2d", r, "keypoints", view, model), sizes, "keypoints2d")
            if r[f"{view}_valid"]:
                bt = box_table(r, view, frames, k)
                if bt is not None and len(bt):
                    write(bt, out_file(out, "boxes", r, "boxes", view), sizes, "boxes")
        f = E4.TRK / f"{pid}_r{int(r.aligned_mat_row)}.npz"
        if f.exists() and all(video_camera(r, view) in cams for view in LABEL_VIEWS):
            write(tracks_table(r, f), out_file(out, "tracks3d", r, "tracks3d"), sizes, "tracks3d")
    return sizes


def main():
    args = C.cli(__doc__.splitlines()[0]).parse_args()
    write_sidecars(args.out)
    for pid in C.selected(args):
        s = build_participant(pid, args.out)
        excluded = [f"{name} ({1 - share:.1%} of the marker samples behind it)"
                    for _, share, name in camera_checks(pid).values() if share < MIN_IN_FRONT]
        print(C.sub_id(pid) + ": " + ", ".join(
            f"{t} {s[t + '_files']} files {s[t + '_rows']} rows {s[t + '_bytes'] / 1e6:.1f} MB" for t in TABLES)
            + (f"; camera not used: {', '.join(excluded)}" if excluded else ""), flush=True)


if __name__ == "__main__":
    main()
