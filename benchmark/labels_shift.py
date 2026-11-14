"""Time-shifted joint labels for fine-tuning on a misaligned reference (alignment-shift analysis, E6).

Adds kp_shift+<ms> (N, 10, 2) to every outputs/benchmark/labels/<pid>/<view>/<clip>.npz: the 6 Hz VICON joint proxies
at the frame time + shift, projected with the video's camera and mapped to crop px with the frame's crop affine (NaN
outside the VICON window). Crops and visibility are unchanged.

    python benchmark/labels_shift.py [--shifts 67,200]       (shifts in ms)
"""
import argparse

import numpy as np

from benchlib import JOINT_NAMES, frame_clock, markers, manifest
from calibrate import camera_id
from evaluate import at, cameras, ref_track
from labels import LAB
from vicon_eval import project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", default="67,200")
    args = ap.parse_args()
    shifts = [int(s) for s in args.shifts.split(",")]
    cams = cameras()
    m = manifest()
    n = 0
    for f in sorted(LAB.glob("*/*/*.npz")):
        if ".tmp" in f.name:
            continue
        pid, view = f.parent.parent.name, f.parent.name
        z = dict(np.load(f))
        todo = [s for s in shifts if f"kp_shift{s:+d}" not in z]
        if not todo:
            continue
        r = m[(m.pid == pid) & (m[f"{view}_clip_path"].astype(str).str.endswith(f.stem + ".mp4"))].iloc[0]
        row = int(r.aligned_mat_row)
        P = cams[camera_id(pid, int(r.visit), view, r[f"{view}_source_video"])]
        v = markers(pid)[0][row]
        tracks = [ref_track(v, s, j) for s in "LR" for j in JOINT_NAMES]
        t = frame_clock(pid, row, view, z["frame"])
        for s in todo:
            ref = np.stack([at(v["t"], tr, t + s / 1000.0) for tr in tracks], 1)               # (N, 10, 3)
            kp = np.full(ref.shape[:2] + (2,), np.nan, np.float32)
            for i in range(len(t)):
                ok = ~np.isnan(ref[i]).any(1)
                if ok.any():
                    kp[i, ok] = project(P, ref[i][ok]) @ z["M"][i][:, :2].T + z["M"][i][:, 2]
            z[f"kp_shift{s:+d}"] = kp
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **z)
        tmp.replace(f)
        n += 1
    print(f"shifted labels added to {n} clips")


if __name__ == "__main__":
    main()
