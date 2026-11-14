"""Trajectory labels of ViconTraj-MS: the game's end-effector trajectory as the fingertip label of the active hand.

The screen plane X = [s, 1] B (game px -> VICON mm) is fitted on the active-hand FIN marker per VICON recording, else
per visit; the label exists wherever the game trajectory does. Adds kp_traj (N, 2), crop px of the trajectory point at
each frame time, and side_traj (active hand: 0 = left, 1 = right) to each labels/<pid>/<view>/<clip>.npz and writes
the median fit residuals (mm) to labels_traj_fit[_<labels>]/<all | PID_PID>.csv, both in outputs/benchmark/.

    python benchmark/labels_traj.py [--participants PID,PID] [--labels labels_hr]
"""
import argparse

import numpy as np
import pandas as pd

from benchlib import BENCH, frame_clock, manifest, markers
from calibrate import camera_id
from e3_screen import game_paths
from evaluate import at, cameras, ref_track
from labels import LAB
from vicon_eval import project


def fit_plane(S, X):
    """B (3 x 3) with X = [S, 1] B, game px (n, 2) -> VICON mm (n, 3), least squares in two passes trimming the worst
    10 %; returns B and the median residual (mm)."""
    A = np.c_[S, np.ones(len(S))]
    keep = np.ones(len(S), bool)
    for _ in range(2):
        B = np.linalg.lstsq(A[keep], X[keep], rcond=None)[0]
        r = np.linalg.norm(A @ B - X, axis=1)
        keep = r <= np.percentile(r, 90)
    return B, float(np.median(r))


def planes(pid, m):
    """Screen planes {(visit, trial_in_visit): B} per VICON recording and {(visit, None): B} per visit, fit records."""
    gp = game_paths(pid)
    vic = markers(pid)[0]
    pairs = {}
    rows = m[(m.pid == pid) & m.vicon_valid]
    for _, r in rows.iterrows():
        v = vic.get(int(r.aligned_mat_row))
        if v is None or int(r.aligned_mat_row) not in gp:
            continue
        t, s = gp[int(r.aligned_mat_row)]
        X = at(v["t"], ref_track(v, str(r.hand)[0], "fingertip"), t)
        ok = ~np.isnan(X).any(1)
        pairs.setdefault((int(r.visit), int(r.trial_in_visit)), []).append((s[ok], X[ok]))
    out, rec = {}, []
    by_visit = {}
    for key, lst in pairs.items():
        S, X = np.concatenate([a for a, _ in lst]), np.concatenate([b for _, b in lst])
        if len(S) >= 200:
            out[key], res = fit_plane(S, X)
            rec.append(dict(pid=pid, visit=key[0], trial=key[1], n=len(S), resid_med_mm=res))
        by_visit.setdefault(key[0], []).append((S, X))
    for visit, lst in by_visit.items():
        S, X = np.concatenate([a for a, _ in lst]), np.concatenate([b for _, b in lst])
        if len(S) >= 200:
            out[(visit, None)], res = fit_plane(S, X)
            rec.append(dict(pid=pid, visit=visit, trial=-1, n=len(S), resid_med_mm=res))
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None)
    ap.add_argument("--labels", default=None, help="label directory name under outputs/benchmark (default labels)")
    args = ap.parse_args()
    global LAB
    if args.labels:
        LAB = BENCH / args.labels
    cams = cameras()
    m = manifest()
    pids = args.participants.split(",") if args.participants else sorted(p.name for p in LAB.iterdir())
    fits, n_clips = [], 0
    for pid in pids:
        B, rec = planes(pid, m)
        fits += rec
        gp = game_paths(pid)
        for f in sorted((LAB / pid).glob("*/*.npz")):
            if ".tmp" in f.name:
                continue
            view = f.parent.name
            z = dict(np.load(f))
            r = m[(m.pid == pid) & (m[f"{view}_clip_path"].astype(str).str.endswith(f.stem + ".mp4"))].iloc[0]
            key = (int(r.visit), int(r.trial_in_visit))
            Bk = B.get(key, B.get((int(r.visit), None)))
            kp = np.full((len(z["frame"]), 2), np.nan, np.float32)
            if Bk is not None and int(r.aligned_mat_row) in gp:
                t, s = gp[int(r.aligned_mat_row)]
                tf = frame_clock(pid, int(r.aligned_mat_row), view, z["frame"])
                sf = at(t, s, tf)
                ok = ~np.isnan(sf).any(1)
                if ok.any():
                    X = np.c_[sf[ok], np.ones(ok.sum())] @ Bk
                    P = cams[camera_id(pid, int(r.visit), view, r[f"{view}_source_video"])]
                    img = project(P, X)
                    M = z["M"][ok]
                    kp[ok] = np.einsum("nij,nj->ni", M[:, :, :2], img) + M[:, :, 2]
            z["kp_traj"] = kp
            z["side_traj"] = np.int8(0 if str(r.hand).upper().startswith("L") else 1)
            tmp = f.with_name(f.stem + ".tmp.npz")
            np.savez(tmp, **z)
            tmp.replace(f)
            n_clips += 1
        print(f"[{pid}] {len(rec)} screen planes, labels added to the clips", flush=True)
    d = pd.DataFrame(fits)
    out = BENCH / ("labels_traj_fit" if not args.labels else f"labels_traj_fit_{args.labels}")
    out.mkdir(exist_ok=True)
    d.to_csv(out / (("all" if not args.participants else "_".join(pids)) + ".csv"), index=False, float_format="%.3f")
    print(f"trajectory labels in {n_clips} clips; screen-plane residual median {d.resid_med_mm.median():.2f} mm "
          f"(max {d.resid_med_mm.max():.2f}) over {len(d)} fits")


if __name__ == "__main__":
    main()
