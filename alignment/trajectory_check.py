"""Test the game-VICON alignment of every trial against the end-effector trajectory logged by the game.

The VICON fingertip of the active hand, mapped to the screen by an affine transform, must reproduce the game trajectory
(RMov_path) at the game's timestamps. The alignment itself uses only the speed of the game object, not this trajectory.
Reads the aligned trials and the exported markers (alignment/vicon/export_markers.m); writes
outputs/game_vicon_aligned/reaching/trajectory_check.csv, read by build_benchmark_manifest.py (trajectory_align_pass).

    python alignment/trajectory_check.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from benchlib import ALIGNED, markers, proxy           # noqa: E402
from kinematics import lowpass_runs                     # noqa: E402  (src/, via benchlib)
from vicon_eval import apply_affine3, fit_affine3       # noqa: E402

MAX_LAG_S = 0.1         # s, residual game-VICON lag
MAX_SELF_MM = 10.0      # RMSE of the trial's own affine
MAX_LOPO_MM = 20.0      # RMSE of the affine fitted on the other trials of the recording
LAGS = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.01), 2)       # s


def game_paths(pid):
    """Game trajectory of every trial: {row: t (absolute s), xy (px), visit, hand, path}."""
    P = np.atleast_1d(sio.loadmat(ALIGNED / f"{pid}_reaching_paths_plain.mat", squeeze_me=True,
                                  struct_as_record=False)["paths"])
    out = {}
    for p in P:
        t = np.atleast_1d(np.asarray(p.RMov_time_step_sec, float))
        xy = np.atleast_2d(np.asarray(p.RMov_path, float))
        n = min(len(t), len(xy))
        out[int(p.row)] = dict(t=pd.Timestamp(str(p.trial_start_time)).value / 1e9 + t[:n], xy=xy[:n],
                               visit=int(p.visit), hand=str(p.trial_hand), path=int(p.trial_no))
    return out


def fin_track(v, side):
    """6 Hz low-passed VICON fingertip (N, 3), mm."""
    x = proxy(v, side, "fingertip")
    return np.stack([lowpass_runs(x[:, d], v["fps"], 6.0) for d in range(3)], 1)


def at(v, x, t):
    """VICON track x interpolated at absolute times t (NaN outside the VICON data)."""
    return np.stack([np.interp(t, v["t"], x[:, d], left=np.nan, right=np.nan) for d in range(3)], 1)


def scale_of(M):
    return np.linalg.svd(M[:3].T, compute_uv=False).mean()          # game px per mm


def rmse(M, X, y):
    """RMSE (mm) of the affine M from VICON X (mm) to the game trajectory y (px)."""
    ok = ~np.isnan(X).any(1)
    return np.sqrt(np.mean(np.sum((apply_affine3(M, X[ok]) - y[ok]) ** 2, 1))) / scale_of(M)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    mf = pd.read_csv(ALIGNED / "aligned_paths.csv")         # outputTable order (summarize_aligned.m)
    mf["aligned_mat_row"] = mf.groupby("pid").cumcount() + 1
    mf["trial_in_visit"] = mf.vicon_round
    rec = []
    for pid in sorted(mf.pid.unique()):
        vic, _ = markers(pid)
        gp = game_paths(pid)
        trial = mf[mf.pid == pid].set_index("aligned_mat_row").trial_in_visit.to_dict()
        X0, fins = {}, {}
        for row, g in gp.items():
            if row in vic:
                fins[row] = fin_track(vic[row], g["hand"][0])
                X0[row] = at(vic[row], fins[row], g["t"])
        for row in X0:
            g, v, fin = gp[row], vic[row], fins[row]
            ok = ~np.isnan(X0[row]).any(1)
            # rmse_self_mm: the trial's own affine at lag 0 (>= 30 samples; the other trials' affine needs >= 100)
            self0 = rmse(fit_affine3(X0[row][ok], g["xy"][ok]), X0[row], g["xy"]) if ok.sum() >= 30 else np.nan
            # rmse_lopo_mm: screen mapping fitted on the other trials of the same recording (same clock offset and
            # screen position; the iPad was sometimes moved between recordings), else of the visit and hand
            others = [o for o in X0 if o != row and gp[o]["visit"] == g["visit"] and trial[o] == trial[row]]
            # lag_method: 'trial affine' = other trials of the recording (trial_in_visit), 'visit affine' = other
            # trials of the visit and hand, 'path affine' = refitted on this trial at every lag
            method = "trial affine"
            if not others:
                others = [o for o in X0 if o != row and gp[o]["visit"] == g["visit"] and gp[o]["hand"] == g["hand"]]
                method = "visit affine"
            M, lopo = None, np.nan
            if others:
                Xo = np.concatenate([X0[o] for o in others])
                yo = np.concatenate([gp[o]["xy"] for o in others])
                oko = ~np.isnan(Xo).any(1)
                if oko.sum() >= 100:
                    M = fit_affine3(Xo[oko], yo[oko])
                    lopo = rmse(M, X0[row], g["xy"]) if ok.sum() >= 30 else np.nan
            if M is None:
                method = "path affine"
            # lag_s: best lag with that mapping held fixed, so that refitting cannot absorb a time shift
            # (refitted at every lag only when the trial has no such mapping)
            best = (np.inf, np.nan)
            for lag in LAGS:
                X = at(v, fin, g["t"] + lag)
                okl = ~np.isnan(X).any(1)
                if okl.sum() < max(0.6 * len(okl), 30):         # VICON at >= 60 % (and >= 30) of the game samples
                    continue
                Ml = M if M is not None else fit_affine3(X[okl], g["xy"][okl])
                best = min(best, (rmse(Ml, X, g["xy"]), lag))
            rec.append(dict(pid=pid, aligned_mat_row=row, visit=g["visit"], path=g["path"], hand=g["hand"],
                            lag_s=best[1], lag_method=method, rmse_best_lag_mm=best[0], rmse_self_mm=self0,
                            rmse_lopo_mm=lopo))
        print(f"[{pid}] {sum(x['pid'] == pid for x in rec)} paths", flush=True)
    d = pd.DataFrame(rec)
    d["trajectory_align_pass"] = ((d.lag_s.abs() <= MAX_LAG_S) & (d.rmse_self_mm <= MAX_SELF_MM)
                                  & ((d.rmse_lopo_mm <= MAX_LOPO_MM) | d.rmse_lopo_mm.isna()))
    why = []
    for r in d.itertuples():
        w = []
        if not np.isfinite(r.lag_s):
            w.append("too little VICON overlap")
        elif abs(r.lag_s) > MAX_LAG_S:
            w.append(f"lag {r.lag_s:+.2f} s")
        if r.rmse_self_mm > MAX_SELF_MM:
            w.append(f"fingertip vs game trajectory {r.rmse_self_mm:.0f} mm")
        if r.rmse_lopo_mm > MAX_LOPO_MM:
            w.append(f"inconsistent with its visit ({r.rmse_lopo_mm:.0f} mm)")
        why.append("; ".join(w))
    d["trajectory_align_reason"] = why
    d.to_csv(ALIGNED / "trajectory_check.csv", index=False, float_format="%.4f")
    print(f"{len(d)} paths with VICON data; {int(d.trajectory_align_pass.sum())} pass")
    print(d.lag_s.abs().describe(percentiles=[.5, .9, .95, .99]).round(3).to_string())
    print(d[["rmse_self_mm", "rmse_lopo_mm"]].describe(percentiles=[.5, .9, .95, .99]).round(2).to_string())


if __name__ == "__main__":
    main()
