"""Shared conventions of the ViKi-Traj release builders: IDs, BIDS file names, time origins, paths and constants."""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

import benchlib  # noqa: E402

RELEASE = ROOT / "release" / "ViKi-Traj"
PRIVATE = ROOT / "release" / "private"
DERIV = "derivatives/vikitraj"
TASK = "reaching"
DATASET_NAME = "ViKi-Traj"
VERSION = "1.0.0"

ALIGNED = ROOT / "outputs" / "game_vicon_aligned" / "reaching"
BENCH = ROOT / "outputs" / "benchmark"
GAME = ROOT / "data" / "game_data"
VICON_RAW = ROOT / "data" / "vicon_data"
BEHAVIOR = ROOT / "data" / "Behavioral_Data_Hand_Limb.csv"

# 12.9-inch iPad Pro: 1366 x 1024 logical points at 132 points per inch. Game positions are in points, origin at
# the screen centre.
MM_PER_GAME_PT = 25.4 / 132.0
SCREEN_PT = (1366, 1024)

VICON_HZ = 100
MARKERS = ['LFHD', 'RFHD', 'LBHD', 'C7', 'T10', 'CLAV', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB',
           'LHND', 'LFIN', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'RHND', 'RFIN', 'RBHD', 'RFRM', 'RUPA']
_PLACE = {"FHD": "front of the head (headband, temple)", "BHD": "back of the head (headband)",
          "C7": "spinous process of the 7th cervical vertebra", "T10": "spinous process of the 10th thoracic vertebra",
          "CLAV": "jugular notch of the sternum", "RBAK": "right scapula, below the scapular spine",
          "SHO": "tip of the acromion", "UPA": "lateral upper arm, between shoulder and elbow",
          "ELB": "lateral humeral epicondyle", "FRM": "dorsal forearm, between elbow and wrist",
          "WRA": "radial styloid process", "WRB": "ulnar styloid process",
          "HND": "dorsum of the hand, below the middle-finger knuckle", "FIN": "tip of the index finger"}
MARKER_PLACEMENT = {m: (("left " if m[0] == "L" else "right ") + _PLACE[m[1:]]) if m[1:] in _PLACE else _PLACE[m]
                    for m in MARKERS}
HEAD_MARKERS = ("LFHD", "RFHD", "LBHD", "RBHD")
JOINTS = list(benchlib.JOINT_NAMES)                                 # shoulder, elbow, wrist, hand, fingertip
JOINT_SET = [f"{s}_{j}" for s in ("left", "right") for j in JOINTS]  # order of the 10-joint benchmark set
JOINT_MARKERS = {j: m for j, m in benchlib.JOINTS.items()}           # joint -> marker suffixes (wrist: mean of WRA/WRB)
VIEWS = ("top", "side")                 # released video views (acq-top, acq-side)
VIDEOS = "sourcedata/videos"            # released clips: top unblurred, side head-blurred

ZERO_SHOT = ["vitpose_b", "rtmpose_wb", "rtmpose_hand", "dwpose_l", "rtmw_x", "sapiens2_0.4b", "sapiens2_1b", "blazepose"]
WITH_METHOD = ["ft_b_full_pooled_traj", "ft_sapiens2_0.4b_head", "ft_sapiens2_1b_head",
               "seq_rtmpose_wb", "seq_rtmpose_hand", "seq_dwpose_l", "seq_rtmw_x", "seq_blazepose"]
MODEL_NAMES = {"vitpose_b": "ViTPose+-B", "rtmpose_wb": "RTMPose-WB", "rtmpose_hand": "RTMPose-m hand",
               "dwpose_l": "DWPose-L", "rtmw_x": "RTMW-X", "sapiens2_0.4b": "Sapiens2-0.4B", "sapiens2_1b": "Sapiens2-1B",
               "blazepose": "BlazePose+Hands", "ft_b_full_pooled_traj": "ViTPose+-B + ViTMS (fine-tuned)",
               "ft_sapiens2_0.4b_head": "Sapiens2-0.4B + ViTMS (head fine-tuned)",
               "ft_sapiens2_1b_head": "Sapiens2-1B + ViTMS (head fine-tuned)"}
MODEL_NAMES.update({f"seq_{m}": f"{MODEL_NAMES[m]} + ViTMS temporal head"
                    for m in ("rtmpose_wb", "rtmpose_hand", "dwpose_l", "rtmw_x", "blazepose")})


# ----------------------------------------------------------------------------- identifiers
@lru_cache(None)
def manifest():
    """The benchmark manifest (benchlib.manifest): one row per trial (696), with group and trajectory_qa_failed."""
    return benchlib.manifest().copy()


@lru_cache(None)
def pids():
    return sorted(manifest().pid.unique())


@lru_cache(None)
def id_map():
    """{XXXXXXX###: sub-NN}, numbered in sorted order of the study ID; saved to PRIVATE/id_map.tsv (not released)."""
    m = {p: f"sub-{i + 1:02d}" for i, p in enumerate(pids())}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"study_id": list(m), "participant_id": list(m.values())}).to_csv(PRIVATE / "id_map.tsv", sep="\t", index=False)
    return m


def sub_id(pid):
    return id_map()[pid]


def ses_id(visit):
    return f"ses-{int(visit)}"


def run_index(trial_in_visit, path):
    """Run (1-12) of a trial: trial_in_visit = game round of the visit (1-3), path = trial within the round (1-4)."""
    return (int(trial_in_visit) - 1) * 4 + int(path)


def run_id(trial_in_visit, path):
    return f"run-{run_index(trial_in_visit, path):02d}"


def row_ids(r):
    """(sub, ses, run) of a manifest row."""
    return sub_id(r.pid), ses_id(r.visit), run_id(r.trial_in_visit, r.path)


def bids_name(sub, ses, suffix, ext, tracksys=None, acq=None, run=None, task=TASK, **extra):
    """BIDS file name, entities in the Motion-BIDS order sub, ses, task, tracksys, acq, run, then extra (e.g. desc)."""
    parts = [sub, ses, f"task-{task}"]
    if tracksys:
        parts.append(f"tracksys-{tracksys}")
    if acq:
        parts.append(f"acq-{acq}")
    if run:
        parts.append(run if str(run).startswith("run-") else f"run-{int(run):02d}")
    parts += [f"{k}-{v}" for k, v in extra.items()]
    return "_".join(parts) + f"_{suffix}{ext}"


def subses_dir(base, sub, ses, *more):
    d = Path(base, sub, ses, *more)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- time
def game_seconds(ts):
    """Absolute time (s) of a timestamp string of the game data. Used only for differences, never released."""
    return pd.Timestamp(str(ts)).value / 1e9


@lru_cache(None)
def session_t0(pid, visit):
    """Absolute game time (s) of the first trial start of the visit: the origin of session time."""
    return float(benchlib.markers(pid)[1][int(visit)])


@lru_cache(None)
def paths_plain(pid):
    """{aligned_mat_row: struct} of <pid>_reaching_paths_plain.mat: game timing, trajectory and VICON crop per trial."""
    import scipy.io as sio
    p = sio.loadmat(ALIGNED / f"{pid}_reaching_paths_plain.mat", squeeze_me=True, struct_as_record=False)["paths"]
    return {int(e.row): e for e in np.atleast_1d(p)}


def run_t0(pid, aligned_mat_row):
    """Absolute game time (s) of the movement start of a trial: the origin of run time (sample 0, clip frame 0)."""
    return game_seconds(paths_plain(pid)[int(aligned_mat_row)].move_start_time)


# ----------------------------------------------------------------------------- writing
def write_tsv(df, path, header=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, header=header, na_rep="n/a", float_format="%.4f", lineterminator="\n")


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def rel(path, out):
    return str(Path(path).relative_to(out))


def cli(description):
    """Argument parser shared by the builders: --participant XXXXXXX### | --all, and --out."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--participant", help="study ID (XXXXXXX###), not sub-NN")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=RELEASE)
    return ap


def selected(args):
    return pids() if args.all else [args.participant]
