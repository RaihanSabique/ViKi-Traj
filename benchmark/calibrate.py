"""Camera calibration: one 3x4 projection (VICON mm -> image px) per top / side video, from its valid trials.

An initial DLT from the zero-shot elbows and wrists (median over the models) is refitted on VICON markers matched to
white image blobs (mutual nearest blob, shrinking radius), so the camera does not depend on any pose model. Fitted on
the even trials of the video (aligned_mat_row), tested on the odd ones (alternate blob frames for a single trial).
Writes outputs/benchmark/cameras/<camera>.npz (P) and calibration.csv; only calib_ok cameras are used downstream.

    python benchmark/calibrate.py [--participant PID]       (with --participant: calibration_<PID>.csv)
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from benchlib import (BENCH, ROOT, VIEWS, aux_file, clip_stem, frame_clock, manifest, markers, model_file,
                      model_joints, proxy, set_lr_swap, valid_rows, KP)
from vicon_eval import dlt, fit_robust, px_per_mm, refine_P, project

MAX_TEST_PX = 8.0             # px: calib_ok needs a held-out median <= MAX_TEST_PX over >= 30 pairs
CAM = BENCH / "cameras"


def interp_markers(v, t):
    """(T, M, 3) marker positions at absolute times t; NaN outside the VICON window."""
    X, tv = v["X"], v["t"]
    i = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1)
    w = ((t - tv[i - 1]) / (tv[i] - tv[i - 1]))[:, None, None]
    out = X[i - 1] * (1 - w) + X[i] * w
    out[(t < tv[0]) | (t > tv[-1])] = np.nan
    return out


def camera_id(pid, visit, view, video):
    """<pid>_V<visit>_<view>_<video file stem, non-alphanumerics as _>"""
    return f"{pid}_V{visit}_{view}_{re.sub(r'[^A-Za-z0-9]+', '_', Path(video).stem)}"


def blobs(pid, view, stem):
    """Frames of a clip with blob detections and the (B, 2) blob centres (px) of each."""
    z = np.load(aux_file("blobs", pid, view, stem))
    return z["frame"], np.split(z["xy"], np.cumsum(z["n"])[:-1])


def calibrate_video(pid, view, video, g):
    """(P, stats) of one video from its valid trials g (manifest rows)."""
    vic, _ = markers(pid)
    cap = cv2.VideoCapture(str(ROOT / video))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    sc = W / 1920.0                                  # px thresholds below are for 1920 px wide video
    # initial camera from the models' elbows and wrists
    X, x = [], []
    for _, r in g.iterrows():
        stem, v = clip_stem(r, view), vic[r.aligned_mat_row]
        zs = {m: dict(np.load(model_file(m, pid, view, stem))) for m in KP if model_file(m, pid, view, stem).exists()}
        if not zs:
            raise FileNotFoundError(2, "no model output", str(model_file("vitpose_b", pid, view, stem)))
        set_lr_swap(zs, pid, view, video)
        fr = next(iter(zs.values()))["frame"]
        t = frame_clock(pid, r.aligned_mat_row, view, fr)
        for side in "LR":
            fused = np.nanmedian(np.stack([model_joints(z, m, side, view) for m, z in zs.items()]), 0)
            for j, jn in ((1, "elbow"), (2, "wrist")):
                ref = np.stack([np.interp(t, v["t"], proxy(v, side, jn)[:, d], left=np.nan, right=np.nan) for d in range(3)], 1)
                ok = ~np.isnan(ref).any(1) & ~np.isnan(fused[:, j]).any(1)
                X.append(ref[ok][::3]); x.append(fused[ok, j][::3])
    X, x = np.concatenate(X), np.concatenate(x)
    P = fit_robust(X, x, thr=40 * sc)
    # VICON markers <-> white blobs of every blob frame
    samples = []                                  # (train?, markers (M, 3), blobs (B, 2))
    single = len(g) == 1
    for k, (_, r) in enumerate(g.iterrows()):
        stem, v = clip_stem(r, view), vic[r.aligned_mat_row]
        bf, bxy = blobs(pid, view, stem)
        Xm = interp_markers(v, frame_clock(pid, r.aligned_mat_row, view, bf))
        for i in range(len(bf)):
            train = (i % 2 == 0) if single else (r.aligned_mat_row % 2 == 0)
            samples.append((train, Xm[i], bxy[i]))

    def match(P, radius, which):
        Xs, xs, ds = [np.zeros((0, 3))], [np.zeros((0, 2))], [np.zeros(0)]
        for train, Xm, bl in samples:
            if (which is not None and train != which) or len(bl) == 0:
                continue
            ok = ~np.isnan(Xm).any(1)
            if not ok.any():
                continue
            pr = project(P, Xm[ok])
            D = np.linalg.norm(pr[:, None] - bl[None], axis=2)
            i = D.argmin(1)
            d = D[np.arange(len(pr)), i]
            good = (d < radius) & (D.argmin(0)[i] == np.arange(len(pr)))
            Xs.append(Xm[ok][good]); xs.append(bl[i[good]]); ds.append(d[good])
        return np.concatenate(Xs), np.concatenate(xs), np.concatenate(ds)

    rmin = max(10 * sc, 8.0)                      # final match radius (px); 8 px floor for 720p videos
    for radius in (40 * sc, 25 * sc, 15 * sc, rmin):
        Xs, xs, _ = match(P, radius, True)
        if len(Xs) < 12:
            break
        P = refine_P(dlt(Xs, xs), Xs, xs)
    _, _, e_test = match(P, rmin, False)
    _, _, e_train = match(P, rmin, True)
    Xs, xs, _ = match(P, rmin, None)
    P_all = refine_P(dlt(Xs, xs), Xs, xs) if len(Xs) >= 12 else P
    med = lambda e: float(np.median(e)) if len(e) else np.nan
    st = dict(pid=pid, visit=int(g.visit.iloc[0]), view=view, video=video, width=W, height=H, n_paths=len(g),
              n_init_pairs=len(X), n_blob_pairs=len(Xs), n_test_pairs=len(e_test),
              reproj_train_median_px=med(e_train), reproj_test_median_px=med(e_test),
              reproj_test_p90_px=float(np.percentile(e_test, 90)) if len(e_test) else np.nan,
              px_per_mm_median=float(np.median(px_per_mm(P_all, Xs))) if len(Xs) else np.nan)
    st["calib_ok"] = bool(st["reproj_test_median_px"] <= MAX_TEST_PX and len(e_test) >= 30)
    return P_all, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default=None)
    args = ap.parse_args()
    CAM.mkdir(parents=True, exist_ok=True)
    pids = [args.participant] if args.participant else sorted(manifest().pid.unique())
    stats = []
    for pid in pids:
        for view in VIEWS:
            for video, g in valid_rows(view, [pid]).groupby(f"{view}_source_video"):
                cid = camera_id(pid, int(g.visit.iloc[0]), view, video)
                try:
                    P, st = calibrate_video(pid, view, video, g)
                except FileNotFoundError as e:
                    print(f"[{cid}] missing input: {e.filename}")
                    continue
                st["camera"] = cid
                np.savez(CAM / f"{cid}.npz", P=P)
                stats.append(st)
                print(f"[{cid}] {st['n_paths']} paths, {st['n_blob_pairs']} marker-blob pairs, held-out median "
                      f"{st['reproj_test_median_px']:.1f} px (p90 {st['reproj_test_p90_px']:.1f}), "
                      f"{st['px_per_mm_median']:.2f} px/mm -> {'OK' if st['calib_ok'] else 'FAIL'}", flush=True)
    out = CAM / ("calibration.csv" if not args.participant else f"calibration_{args.participant}.csv")
    pd.DataFrame(stats).to_csv(out, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
