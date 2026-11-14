"""Shared definitions of the benchmark: directories, manifest, joints and marker proxies, clocks, model registry.

VICON samples and video frames of a trial ("path" in the code) are placed on the absolute game clock (Unix s):
    VICON sample n : trial_start_time - relative_start_sec + time[n]       (export_markers.m)
    video frame f  : t0_visit + (f / fps - offset_s) / drift_slope          (align_videos.py)
t0_visit is the earliest trial_start_time of the visit; offset_s and drift_slope come from video_segments_all.csv.
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))      # cached weights, so jobs can run offline
sys.path.append(str(ROOT / "src"))                    # src/ modules; appended, so benchmark/ wins on name clashes

ALIGNED = ROOT / "outputs" / "game_vicon_aligned" / "reaching"      # aligned trials (alignment/)
MANIFEST = ALIGNED / "benchmark_manifest.csv"                        # one row per trial (build_benchmark_manifest.py)
BENCH = ROOT / "outputs" / "benchmark"                               # every benchmark output
VIEWS = ("top", "side")

# Joint -> VICON markers averaged into its proxy (prefixed with the side, L / R).
JOINTS = {"shoulder": ["SHO"], "elbow": ["ELB"], "wrist": ["WRA", "WRB"], "hand": ["HND"], "fingertip": ["FIN"]}
JOINT_NAMES = list(JOINTS)

# Keypoint index of each joint per model and side; -1 = not provided, "h<k>" = point k of the model's hand output.
KP = {
    "vitpose_b": {"L": [5, 7, 9, -1, -1], "R": [6, 8, 10, -1, -1]},
    # COCO-WholeBody hands start at 91 (left) / 112 (right): +9 middle-finger MCP (HND marker), +8 index tip
    "rtmpose_wb": {"L": [5, 7, 9, 91 + 9, 91 + 8], "R": [6, 8, 10, 112 + 9, 112 + 8]},
    "blazepose": {"L": [11, 13, 15, "h9", "h8"], "R": [12, 14, 16, "h9", "h8"]},
    # RTMPose-m hand on hand crops: 0 wrist, 9 middle MCP, 8 index tip
    "rtmpose_hand": {"L": [-1, -1, "h0", "h9", "h8"], "R": [-1, -1, "h0", "h9", "h8"]},
    # Sapiens2 308 keypoints (Sociopticon): shoulder, elbow, wrist, middle_finger_third_joint, forefinger4
    "sapiens2_0.4b": {"L": [5, 7, 62, 53, 46], "R": [6, 8, 41, 32, 25]},
}
KP["vitpose_l"] = KP["vitpose_h"] = KP["vitpose_b"]
KP["dwpose_l"] = KP["rtmw_x"] = KP["rtmpose_wb"]
KP["sapiens2_1b"] = KP["sapiens2_0.4b"]
FT_KP = {"L": [0, 1, 2, 3, 4], "R": [5, 6, 7, 8, 9]}     # joint-set head of the fine-tuned models (finetune.py)
# Keypoint score threshold per model.
THR = {"vitpose_b": 0.3, "vitpose_l": 0.3, "vitpose_h": 0.3, "rtmpose_wb": 0.3, "dwpose_l": 0.3, "rtmw_x": 0.3,
       "rtmpose_hand": 0.3, "sapiens2_0.4b": 0.3, "sapiens2_1b": 0.3, "blazepose": 0.5}
# Display name per model (tables and figures).
MODEL_LABEL = {"vitpose_b": "ViTPose+-B", "vitpose_l": "ViTPose+-L", "vitpose_h": "ViTPose+-H",
               "rtmpose_wb": "RTMPose-WB", "dwpose_l": "DWPose-L", "rtmw_x": "RTMW-X",
               "rtmpose_hand": "RTMPose-m hand (RTMW-X crops)", "sapiens2_0.4b": "Sapiens2-0.4B",
               "sapiens2_1b": "Sapiens2-1B", "blazepose": "BlazePose+Hands"}


def _register_finetuned():
    """Add each run with outputs in models/ft_* (fine-tuned) or models/seq_* (temporal head) to KP, THR, MODEL_LABEL."""
    dirs = sorted(list((BENCH / "models").glob("ft_*")) + list((BENCH / "models").glob("seq_*"))) if (BENCH / "models").exists() else []
    for d in dirs:
        cfg = BENCH / "finetune" / d.name / "config.json"
        if not cfg.exists():
            continue
        c = json.loads(cfg.read_text())
        KP[d.name], THR[d.name] = (KP["sapiens2_1b"] if c.get("kp_layout") == "sapiens2" else FT_KP), 0.3
        extra = [f"n={c['n_train']}" if c.get("n_train") else "", f"shift {c['label_shift_ms']:+d} ms" if c.get("label_shift_ms") else "",
                 "all labels" if c.get("mask") == "has" else "",
                 f"trajectory ({c.get('traj_mode', 'fill')})" if c.get("traj_weight") else ""]
        views = "pooled" if "," in c["views"] else c["views"]
        size = {"base": "B", "large": "L", "huge": "H"}.get(c.get("backbone", "base"), "")
        # label from config.json (Sapiens2 and seq_* runs), else built from the ViTPose+ run settings
        MODEL_LABEL[d.name] = c.get("label") or (f"ViTPose+-{size} FT {c['mode']}, {views}" + "".join(f", {e}" for e in extra if e))


_register_finetuned()
# BlazePose labels sides as if the person faced the camera: its sides are checked per video against ViTPose+-B.
MIRRORED_VIEWS = {"blazepose": {"top"}}               # fixed alternative (mirror="view"): sides swapped in these views
SIDE_CHECKED = {"blazepose": "vitpose_b"}             # model -> reference model for its left/right
# Shoulder, elbow and wrist keypoints per side of the models that lr_margin compares.
ARM_IDX = {"blazepose": {"L": [11, 13, 15], "R": [12, 14, 16]}, "vitpose_b": {"L": [5, 7, 9], "R": [6, 8, 10]}}


def lr_margin(a, b, m, ref):
    """Side margin of a clip (px): cross- minus same-side arm distance to the reference model; < 0 = swapped."""
    ka = {sd: np.where((a["s"][:, ARM_IDX[m][sd]] >= THR[m])[..., None], a["k"][:, ARM_IDX[m][sd]], np.nan) for sd in "LR"}
    kb = {sd: np.where((b["s"][:, ARM_IDX[ref][sd]] >= THR[ref])[..., None], b["k"][:, ARM_IDX[ref][sd]], np.nan) for sd in "LR"}
    d = lambda x, y: np.nanmedian(np.nanmean(np.linalg.norm(x - y, axis=2), axis=1))
    same = d(ka["L"], kb["L"]) + d(ka["R"], kb["R"])
    cross = d(ka["L"], kb["R"]) + d(ka["R"], kb["L"])
    return cross - same


@lru_cache(None)
def lr_swap_table():
    """{(model, pid, view, video): median clip margin < 0}; also written to models/<model>/lr_swap_by_video.csv."""
    rec = []
    for m, ref in SIDE_CHECKED.items():
        for view in VIEWS:
            for _, r in valid_rows(view).iterrows():
                fa, fb = model_file(m, r.pid, view, clip_stem(r, view)), model_file(ref, r.pid, view, clip_stem(r, view))
                if fa.exists() and fb.exists():
                    rec.append(dict(model=m, pid=r.pid, view=view, video=r[f"{view}_source_video"],
                                    margin=lr_margin(np.load(fa), np.load(fb), m, ref)))
    d = pd.DataFrame(rec, columns=["model", "pid", "view", "video", "margin"])
    t = d.groupby(["model", "pid", "view", "video"]).margin.agg(
        n_clips="size", clips_swapped=lambda x: float(np.mean(x < 0)), median_margin_px="median").reset_index()
    t["swapped"] = t.median_margin_px < 0                 # median over the clips of the video
    for m, g in t.groupby("model"):
        f = BENCH / "models" / m / "lr_swap_by_video.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(f, index=False, float_format="%.3f")
    return {(r.model, r.pid, r.view, r.video): bool(r.swapped) for r in t.itertuples()}


def set_lr_swap(zs, pid, view, video):
    """Set zs[model]['lr_swapped'] for the side-checked models from the per-video decision."""
    for m in SIDE_CHECKED:
        key = (m, pid, view, video)
        if m in zs and key in lr_swap_table():
            zs[m]["lr_swapped"] = lr_swap_table()[key]


BEHAVIOR = ROOT / "data" / "Behavioral_Data_Hand_Limb.csv"      # affected arm: paretic (UCP) or non-dominant (TD)


@lru_cache(None)
def _affected_side():
    """{pid: 'L' or 'R'} of the affected arm."""
    b = pd.read_csv(BEHAVIOR)
    return dict(zip(b["Participant ID"], b["Paretic (UCP)/ Nondominant (TD) UL"].str.strip().str[0]))


def limb_of(pid, hand):
    """'affected' if hand ('Left' / 'Right' / 'L' / 'R') is the affected arm, else 'less_affected' (or 'unknown')."""
    side = _affected_side().get(pid)
    if side is None:
        return "unknown"
    return "affected" if str(hand)[0].upper() == side else "less_affected"


def clip_stem(row, view):
    """Stem of the trial's clip in this view, the file name of every per-clip output."""
    return Path(str(row[f"{view}_clip_path"])).stem


def model_file(model, pid, view, stem):
    """Keypoints of a model for one clip."""
    return BENCH / "models" / model / pid / view / f"{stem}.npz"


def aux_file(kind, pid, view, stem):
    """Person boxes (kind 'boxes') or marker blobs ('blobs') of a clip, written by zero_shot.py."""
    return BENCH / kind / pid / view / f"{stem}.npz"


QA_FILE = BENCH / "qa_vicon_trajectory.csv"                 # VICON fingertip vs game trajectory, per visit (e3_screen)


@lru_cache(None)
def manifest():
    """Manifest with group (TD / UCP) and trajectory_qa_failed (the visit fails the QA_FILE test; see valid_rows)."""
    m = pd.read_csv(MANIFEST)
    m["group"] = np.where(m.motor_impaired, "UCP", "TD")
    m["trajectory_qa_failed"] = False
    if QA_FILE.exists():
        q = pd.read_csv(QA_FILE)
        bad = set(map(tuple, q.loc[q.failed, ["pid", "visit"]].values))
        m["trajectory_qa_failed"] = [(p, v) in bad for p, v in zip(m.pid, m.visit)]
    return m


def valid_rows(view, pids=None):
    """Trials with valid VICON and a valid video of this view, in visits that pass the trajectory QA."""
    m = manifest()
    s = m[m.vicon_valid & m[f"{view}_valid"] & ~m.trajectory_qa_failed]
    return s if pids is None else s[s.pid.isin(pids)]


@lru_cache(None)
def segments():
    """Video segment of every trial and view (offset_s, drift_slope, fps), indexed by (pid, row, view)."""
    s = pd.read_csv(ALIGNED / "video" / "video_segments_all.csv")
    return s.set_index(["pid", "row", "view"])


def splits():
    """Participant splits: test, folds (five development folds) and cross_visit."""
    return json.loads((BENCH / "splits.json").read_text())


def _sec(ts):
    """Unix seconds of a timestamp string."""
    return pd.Timestamp(str(ts)).value / 1e9


@lru_cache(None)
def markers(pid):
    """{row: dict(X (N, 24, 3) mm, t (N,) absolute game s, labels, fps, visit)} and {visit: t0_visit}."""
    m = sio.loadmat(BENCH / "markers" / f"{pid}_reaching_markers.mat", squeeze_me=True, struct_as_record=False)
    out, t0 = {}, {}
    for p in np.atleast_1d(m["paths"]):
        start = _sec(p.trial_start_time)
        v = int(p.visit)
        t0[v] = min(t0.get(v, np.inf), start)
        X = np.asarray(p.X, float)
        if X.ndim != 3 or X.shape[2] == 0:
            continue
        out[int(p.row)] = dict(X=X.transpose(2, 1, 0), t=start - float(p.relative_start_sec) + np.atleast_1d(p.time),
                               labels=[str(s) for s in p.labels], fps=float(p.framerate), visit=v)
    return out, t0


def frame_clock(pid, row, view, frames):
    """Absolute game time (s) of frames of the source video of a trial in this view."""
    s = segments().loc[(pid, row, view)]
    _, t0 = markers(pid)
    return t0[int(s.visit)] + (np.asarray(frames) / float(s.fps) - float(s.offset_s)) / float(s.drift_slope)


def proxy(v, side, joint):
    """(N, 3) VICON proxy of a joint of one side ('L' / 'R'): the mean of its markers, mm."""
    idx = [v["labels"].index(side + n) for n in JOINTS[joint]]
    return v["X"][:, idx].mean(1)


def marker_index(v, side, joint):
    """Indices of the joint's markers in v["labels"]."""
    return [v["labels"].index(side + n) for n in JOINTS[joint]]


def model_joints(z, model, side, view, mirror="auto"):
    """(T, 5, 2) joint px (NaN below THR); mirror: 'auto' (set_lr_swap, else MIRRORED_VIEWS), 'view' or False."""
    swap = False
    if mirror == "view" or (mirror == "auto" and "lr_swapped" not in z):
        swap = view in MIRRORED_VIEWS.get(model, ())
    elif mirror == "auto":
        swap = bool(z["lr_swapped"])
    if swap:
        side = {"L": "R", "R": "L"}[side]
    k, s = z.get("k"), z.get("s")                                       # None for the hand-only model
    out = np.full((len(z["frame"]), len(JOINT_NAMES), 2), np.nan)
    for j, idx in enumerate(KP[model][side]):
        if idx == -1 or (k is None and not isinstance(idx, str)):
            continue
        if isinstance(idx, str):
            h = int(idx[1:])
            out[:, j] = z[f"hand_{side}"][:, h]
            if f"hand_s_{side}" in z:                                   # scored hand model
                out[z[f"hand_s_{side}"][:, h] < THR[model], j] = np.nan
        else:
            ok = s[:, idx] >= THR[model]
            out[ok, j] = k[ok, idx]
    return out
