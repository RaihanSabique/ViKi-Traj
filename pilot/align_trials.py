"""Pilot analysis of one participant (supplement), step 1: align each visit video to the game trials by optical flow.

Reads data/<pid>/: <pid>_reaching_plain.mat (export_trials.m: the aligned trial table provided with the data, made
outside alignment/vicon; game trajectory with clock times and VICON markers per trial), one video per visit and view
(V1_*_Top.mp4, V1_*_Side.mp4, V2_*_Top.mp4, V2_*_Side.mp4) and rois.json (per video, a polygon in px: the screen in
the top view, the strip where the finger meets the screen in the side view).
Writes outputs/<pid>/alignment/: trial_segments.csv (per trial and view: video start and end in s and frames),
alignment_summary.csv (per video: offset, score, margin, drift slope, residual SD), alignment_<visit>.png (QA),
flow_<video>.npz (cached flow) and, with --cut, clips/<visit>_<view>_T<row>_g<group>_t<trial>_<hand>.mp4.

    python pilot/align_trials.py --participant PID [--cut]
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GRID_HZ = 30.0                      # common resampling rate for the correlation
VISITS = {"Day-1": "V1", "Day-2": "V2"}     # study_day of the trial table -> visit


# ----------------------------------------------------------------------------- game side
def load_trials(participant):
    """One row per trial (row: 1-based index in the trial table): visit, group (VICON recording), trial, hand, start
    and end timestamps, and the game-object trajectory rmov_xy (game px) at rmov_t (s since the trial start)."""
    m = sio.loadmat(DATA / participant / f"{participant}_reaching_plain.mat",
                    squeeze_me=True, struct_as_record=False)
    rows = []
    for i, r in enumerate(np.atleast_1d(m["trials"])):
        rows.append(dict(row=i + 1, visit=VISITS[str(r.study_day)], group=int(r.group_no),
                         trial=int(r.trial_no), hand=str(r.trial_hand),
                         start=pd.Timestamp(str(r.trial_start_time)),
                         end=pd.Timestamp(str(r.trial_end_time)),
                         rmov_t=np.asarray(r.RMov_time_step_ms, float) / 1000.0,  # s since start
                         rmov_xy=np.asarray(r.RMov_path, float)))
    return pd.DataFrame(rows)


def game_velocity(trials, t0, n):
    """Game-object velocity (game px/s) on a GRID_HZ grid of the visit clock (s since the visit's first trial start);
    0 between trials."""
    vx, vy = np.zeros(n), np.zeros(n)
    for _, r in trials.iterrows():
        ts = (r.start - t0).total_seconds() + r.rmov_t
        v = np.gradient(r.rmov_xy, r.rmov_t, axis=0)
        i0, i1 = int(np.ceil(ts[0] * GRID_HZ)), min(int(ts[-1] * GRID_HZ), n)
        g = np.arange(i0, i1) / GRID_HZ
        vx[i0:i1] = np.interp(g, ts, uniform_filter1d(v[:, 0], 9))
        vy[i0:i1] = np.interp(g, ts, uniform_filter1d(v[:, 1], 9))
    return vx, vy


# ----------------------------------------------------------------------------- video side
def screen_axes(roi):
    """Unit vectors of the ROI's horizontal (TL->TR) and vertical (TL->BL) edges."""
    q = np.asarray(roi, float)
    u = (q[1] - q[0]) + (q[2] - q[3])
    v = (q[3] - q[0]) + (q[2] - q[1])
    return u / np.linalg.norm(u), v / np.linalg.norm(v)


def motion_flow(path, roi, cache, scale=0.5):
    """Per-frame dense optical flow (Farneback) inside the ROI: mean flow of the moving
    pixels, projected on the ROI's horizontal / vertical axes (full-res px per frame)."""
    if cache.exists():
        z = np.load(cache)
        return z["fu"], z["fv"], z["t"], float(z["fps"])
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = np.asarray(roi, float)
    x0, y0 = q.min(0).astype(int)
    x1, y1 = q.max(0).astype(int)
    w, h = int((x1 - x0) * scale), int((y1 - y0) * scale)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [((q - [x0, y0]) * scale).astype(np.int32)], 1)
    m = mask.astype(bool)
    ax_u, ax_v = screen_axes(roi)
    fu, fv, t, prev = [], [], [], None
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=path.name):
        ok, im = cap.read()
        if not ok:
            break
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        g = cv2.cvtColor(cv2.resize(im[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        if prev is None:
            fu.append(0.0)
            fv.append(0.0)
        else:
            f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)[m] / scale
            mv = f[np.linalg.norm(f, axis=1) > 1.0]     # moving pixels: finger, hand, game object
            mean = mv.mean(0) if len(mv) > 10 else np.zeros(2)
            fu.append(float(mean @ ax_u))
            fv.append(float(mean @ ax_v))
        prev = g
    fu, fv, t = np.asarray(fu), np.asarray(fv), np.asarray(t)
    t = t - t[0]
    np.savez(cache, fu=fu, fv=fv, t=t, fps=fps)
    return fu, fv, t, fps


def to_grid(x, t, n=None):
    n = n or int(t[-1] * GRID_HZ) + 1
    return np.interp(np.arange(n) / GRID_HZ, t, x)


# ----------------------------------------------------------------------------- matching
def corr_at(sig, tmpl, k):
    """Pearson r between tmpl and sig shifted by k grid samples (overlap only)."""
    if k >= 0:
        b = sig[k:k + len(tmpl)]
        a = tmpl[:len(b)]
    else:
        a = tmpl[-k:]
        b = sig[:len(a)]
        a = a[:len(b)]
    if len(a) < 10:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score(view, fu, fv, gx, gy, k):
    """Screen right = game +x; image down = game -y. The side view sees only vertical motion."""
    if view == "Top":
        return corr_at(fu, gx, k) - corr_at(fv, gy, k)
    return -corr_at(fv, gy, k)


def global_offset(view, fu, fv, gx, gy, lags):
    """Offset (s; video time = visit clock + offset) with the best score over lags, the score, and its margin over the
    best score more than 1 s away (trials recur about every 25 s)."""
    s = np.array([score(view, fu, fv, gx, gy, int(round(l * GRID_HZ))) for l in lags])
    k = int(np.nanargmax(s))
    far = np.abs(lags - lags[k]) > 1.0
    return lags[k], s[k], s[k] - np.nanmax(s[far])


def refine(view, fu, fv, gx, gy, lag, i0, i1, search_s=1.0, pad_s=0.5):
    """Best shift (s) within +/- search_s for one trial window [i0, i1) of the game grid."""
    p = int(pad_s * GRID_HZ)
    a0, a1 = max(i0 - p, 0), i1 + p
    best = (-np.inf, 0.0)
    for d in np.arange(-search_s, search_s + 1e-9, 1 / GRID_HZ):
        k = int(round((lag + d) * GRID_HZ)) + a0
        if k < 0 or k + (a1 - a0) > len(fu):
            continue
        s = score(view, fu[k:k + a1 - a0], fv[k:k + a1 - a0], gx[a0:a1], gy[a0:a1], 0)
        if s > best[0]:
            best = (s, d)
    return best[1], best[0]


# ----------------------------------------------------------------------------- QA + clips
def qa_plot(path, visit, res, sigs, gy):
    """QA figure of one visit: upward finger flow per view against the vertical game-object velocity, offset applied,
    with the aligned trials shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    views = list(sigs)
    fig, axes = plt.subplots(len(views), 1, figsize=(14, 2.8 * len(views)), facecolor="#fcfcfb",
                             sharex=True)
    for ax, view in zip(np.atleast_1d(axes), views):
        fu, fv, lag = sigs[view]
        ax.plot(np.arange(len(fv)) / GRID_HZ, -fv / (np.std(fv) + 1e-9), color="#2a78d6", lw=0.7,
                label="video: upward finger flow (z)")
        ax.plot(np.arange(len(gy)) / GRID_HZ + lag, gy / (np.std(gy[gy != 0]) + 1e-9),
                color="#eb6834", lw=0.7, alpha=0.8, label="game: game-object vertical velocity (z), offset applied")
        for _, r in res[res.view == view].iterrows():
            ax.axvspan(r.video_start_s, r.video_end_s, color="#1baf7a", alpha=0.12, lw=0)
            ax.text((r.video_start_s + r.video_end_s) / 2, 3.6, f"T{r.row}", ha="center",
                    fontsize=7, color="#52514e")
        ax.set_ylim(-4, 4)
        ax.set_ylabel(f"{view} view", fontsize=9, color="#52514e")
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("Video time (s)")
    fig.suptitle(f"{visit}: finger flow in the video vs game-object velocity "
                 f"(green = aligned trials)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cut_clip(src, dst, t0, t1):
    """Re-encode src from t0 to t1 (s) to dst (H.264, CRF 20, no audio)."""
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}",
                    "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-an", str(dst)], check=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="XXXXXXX115")
    ap.add_argument("--cut", action="store_true", help="write per-trial clips")
    ap.add_argument("--pad", type=float, default=1.0, help="seconds kept before/after each clip")
    args = ap.parse_args()

    ddir = DATA / args.participant
    odir = OUT / args.participant / "alignment"
    (odir / "clips").mkdir(parents=True, exist_ok=True)
    rois = json.loads((ddir / "rois.json").read_text())
    trials = load_trials(args.participant)
    rows, summary = [], []
    for visit, tv in trials.groupby("visit"):
        tv = tv.sort_values("start")
        t0 = tv.start.min()                           # visit clock origin: first trial start
        span = (tv.end.max() - t0).total_seconds()
        gx, gy = game_velocity(tv, t0, int(span * GRID_HZ) + 1)
        sigs = {}
        for view in ("Top", "Side"):
            vids = sorted(ddir.glob(f"{visit}_*_{view}.mp4"))
            if not vids:
                continue
            vp = vids[0]
            fu, fv, t, fps = motion_flow(vp, rois[vp.stem], odir / f"flow_{vp.stem}.npz")
            fu, fv = uniform_filter1d(to_grid(fu, t), 3), uniform_filter1d(to_grid(fv, t), 3)
            vdur = len(fu) / GRID_HZ
            # The video must contain the visit, give or take 30 s.
            lags = np.arange(-30, vdur - span + 30, 1 / GRID_HZ)
            lag, peak, margin = global_offset(view, fu, fv, gx, gy, lags)
            sigs[view] = (fu, fv, lag)
            summary.append(dict(visit=visit, view=view, video=vp.name, offset_s=lag,
                                score=peak, margin=margin))
            print(f"[{visit} {view}] offset {lag:7.2f} s  score {peak:.3f}  margin over next-best {margin:.3f}")
            for _, r in tv.iterrows():
                gs = (r.start - t0).total_seconds() + r.rmov_t[0]
                ge = (r.end - t0).total_seconds()
                d, rr = refine(view, fu, fv, gx, gy, lag, int(gs * GRID_HZ), int(ge * GRID_HZ))
                vs, ve = gs + lag + d, ge + lag + d
                rows.append(dict(participant=args.participant, visit=visit, view=view,
                                 video=vp.name, row=r.row, group=r.group, trial=r.trial,
                                 hand=r.hand, game_start=r.start, game_end=r.end,
                                 visit_clock_start_s=gs, visit_clock_end_s=ge,
                                 offset_s=lag, refine_shift_s=d, trial_score=rr,
                                 video_start_s=vs, video_end_s=ve,
                                 start_frame=int(round(vs * fps)), end_frame=int(round(ve * fps)),
                                 fps=fps))
        res = pd.DataFrame([r for r in rows if r["visit"] == visit])
        qa_plot(odir / f"alignment_{visit}.png", visit, res, sigs, gy)
    res = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    # One line per video (offset and clock drift) through the refined trial starts. A single-trial refinement can lock
    # onto a neighbouring sub-movement (a shift in one view only), so trials more than 0.2 s off the line are excluded
    # and the line refitted; every trial takes its times from the line, the refined start stays as QA.
    res["refined_start_s"] = res.video_start_s
    for (visit, view), g in res.groupby(["visit", "view"]):
        x, y = g.visit_clock_start_s.values, g.video_start_s.values
        keep = np.ones(len(g), bool)
        for _ in range(3):
            a, b = np.polyfit(x[keep], y[keep], 1)
            keep = np.abs(y - (a * x + b)) <= 0.2
        resid = y - (a * x + b)
        i = g.index
        fps = g.fps.iloc[0]
        res.loc[i, "video_start_s"] = a * g.visit_clock_start_s + b
        res.loc[i, "video_end_s"] = a * g.visit_clock_end_s + b
        res.loc[i, "start_frame"] = np.round(res.loc[i, "video_start_s"] * fps).astype(int)
        res.loc[i, "end_frame"] = np.round(res.loc[i, "video_end_s"] * fps).astype(int)
        res.loc[i, "refine_residual_s"] = resid
        res.loc[i, "refine_outlier"] = ~keep
        j = (summ.visit == visit) & (summ.view == view)
        summ.loc[j, "drift_slope"] = a
        summ.loc[j, "intercept_s"] = b
        summ.loc[j, "resid_sd_inliers_s"] = resid[keep].std()
        summ.loc[j, "n_outliers"] = int((~keep).sum())
        print(f"[{visit} {view}] video = {a:.5f} x visit clock + {b:.3f} s; inlier residual SD "
              f"{resid[keep].std() * 1000:.0f} ms over {keep.sum()} trials; outliers: "
              f"{list(g.row[~keep])} ({', '.join(f'{v:+.2f}' for v in resid[~keep])} s)")
    # Top and side see the same movements: their per-trial start difference should be constant.
    p = res.pivot_table(index=["visit", "row"], columns="view", values="video_start_s")
    if {"Top", "Side"} <= set(p.columns):
        for visit, g in (p.Top - p.Side).groupby(level="visit"):
            print(f"[{visit}] top - side start: mean {g.mean():.3f} s, SD {g.std() * 1000:.0f} ms")
    res.to_csv(odir / "trial_segments.csv", index=False, float_format="%.4f")
    summ.to_csv(odir / "alignment_summary.csv", index=False, float_format="%.4f")
    print(f"wrote {odir / 'trial_segments.csv'} ({len(res)} rows)")
    if args.cut:
        for _, r in tqdm(res.iterrows(), total=len(res), desc="clips"):
            name = f"{r.visit}_{r['view']}_T{r.row:02d}_g{r.group}_t{r.trial}_{r.hand}.mp4"
            cut_clip(ddir / r.video, odir / "clips" / name,
                     max(r.video_start_s - args.pad, 0), r.video_end_s + args.pad)


if __name__ == "__main__":
    main()
