"""Align the top and side videos of a participant to the game clock and cut one clip per trial and view.

Reads the trials (paths in the game data) from outputs/game_vicon_aligned/reaching/<pid>_reaching_paths_plain.mat and
the visit videos from data/raw_video_data/<pid>/. Writes video_segments.csv, video_alignment_summary.csv, QA plots and
clips/<view>/<pid>_V<visit>_T<recording>_P<path>_<hand>.mp4 to outputs/game_vicon_aligned/reaching/video/<pid>/.
Each clip covers the movement window of its trial. Method and columns: alignment/video/README.md.

    python alignment/video/align_videos.py --participant XXXXXXX115 [--no-cut]
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_video_data"
ALIGNED = ROOT / "outputs" / "game_vicon_aligned" / "reaching"
GRID_HZ = 30.0              # common rate (Hz) of the cell flow and the game-object velocity
FLOW_WIDTH = 320            # optical flow is computed on frames resized to this width
GRID = (16, 9)              # cells across, down
TOP_CELLS = 6               # cells combined into the alignment score
VIEWS = {"top": "Logitech", "side": "Tripod"}       # view -> raw video folder name


# ----------------------------------------------------------------------------- game side
def load_paths(pid):
    """One row per trial (row = outputTable row), in time order."""
    m = sio.loadmat(ALIGNED / f"{pid}_reaching_paths_plain.mat", squeeze_me=True,
                    struct_as_record=False)
    rows = []
    for r in np.atleast_1d(m["paths"]):
        rows.append(dict(
            row=int(r.row), visit=int(r.visit), group=int(r.group_no), path=int(r.trial_no),
            hand=str(r.trial_hand), start=pd.Timestamp(str(r.trial_start_time)),
            end=pd.Timestamp(str(r.trial_end_time)),
            move_start=pd.Timestamp(str(r.move_start_time)), move_end=pd.Timestamp(str(r.move_end_time)),
            rmov_t=np.atleast_1d(np.asarray(r.RMov_time_step_sec, float)),     # s from trial_start_time
            rmov_xy=np.atleast_2d(np.asarray(r.RMov_path, float)),              # game-object position (px)
            review_status=str(r.review_status)))
    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    # recording within the visit (1..3; one game group of 4 trials per VICON recording), in time order
    df["trial_in_visit"] = df.groupby("visit").group.rank(method="dense").astype(int)
    return df


def game_velocity(paths, t0, n):
    """Game-object velocity (px/s) on the GRID_HZ grid of the visit clock, and a mask of the samples inside trials."""
    vx, vy, mask = np.zeros(n), np.zeros(n), np.zeros(n)
    for _, r in paths.iterrows():
        t, xy = r.rmov_t, r.rmov_xy
        k = min(len(t), len(xy))
        t, xy = t[:k], xy[:k]
        t, iu = np.unique(t, return_index=True)
        xy = xy[iu]
        if len(t) < 5:
            continue
        ts = (r.start - t0).total_seconds() + t
        v = np.gradient(xy, t, axis=0)
        i0, i1 = int(np.ceil(ts[0] * GRID_HZ)), min(int(ts[-1] * GRID_HZ), n)
        if i1 <= i0:
            continue
        g = np.arange(i0, i1) / GRID_HZ
        vx[i0:i1] = np.interp(g, ts, uniform_filter1d(v[:, 0], 9))
        vy[i0:i1] = np.interp(g, ts, uniform_filter1d(v[:, 1], 9))
        mask[i0:i1] = 1
    return vx, vy, mask


# ----------------------------------------------------------------------------- video side
def grid_flow(path, cache):
    """Mean Farneback flow (u, v, px/frame at FLOW_WIDTH) of each grid cell, per frame."""
    if cache.exists():
        z = np.load(cache)
        return z["f"], z["t"], float(z["fps"])
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = FLOW_WIDTH
    h = int(round(H * w / W))
    gx, gy = GRID
    ex, ey = np.linspace(0, w, gx + 1).astype(int), np.linspace(0, h, gy + 1).astype(int)
    feats, ts, prev = [], [], None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        g = cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is None:
            feats.append(np.zeros((gy, gx, 2), np.float32))
        else:
            f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            c = np.add.reduceat(np.add.reduceat(f, ey[:-1], axis=0), ex[:-1], axis=1)
            area = np.outer(np.diff(ey), np.diff(ex))[..., None]
            feats.append((c / area).astype(np.float32))
        prev = g
    cap.release()
    f = np.asarray(feats).reshape(len(feats), -1, 2)          # frames x cells x (u, v)
    t = np.asarray(ts)
    t = t - t[0]
    if len(t) > 1 and not np.all(np.diff(t) > 0):              # broken timestamps: use fps
        t = np.arange(len(t)) / fps
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, f=f, t=t, fps=fps)
    return f, t, fps


def to_grid(f, t):
    """Cell flow resampled to GRID_HZ and smoothed over 3 samples."""
    n = int(t[-1] * GRID_HZ) + 1
    tg = np.arange(n) / GRID_HZ
    out = np.empty((n,) + f.shape[1:], np.float32)
    for c in range(f.shape[1]):
        for k in range(f.shape[2]):
            out[:, c, k] = np.interp(tg, t, f[:, c, k])
    return uniform_filter1d(out, 3, axis=0)


# ----------------------------------------------------------------------------- matching
def xcorr(a, b, L):
    """c[k] = sum_j a[k + j] b[j] for every circular shift k (columns of a vs vector b)."""
    return np.fft.irfft(np.fft.rfft(a, L, axis=0) * np.conj(np.fft.rfft(b, L))[:, None], L, axis=0)


def cell_scores(F, gx, gy, mask, min_overlap):
    """Rotation-invariant Pearson score of every cell at every lag, over the game samples inside trials (mask).

    F: video grid flow (Nv x cells x 2). Game sample j sits at video sample j + k.
    Returns lags (s) and scores (lags x cells)."""
    Nv, C, _ = F.shape
    Ng = len(gx)
    L = 1 << int(np.ceil(np.log2(Nv + Ng)))
    A = F.reshape(Nv, C * 2).astype(np.float64)
    W = np.ones(Nv)
    n = xcorr(W[:, None], mask, L)[:, 0]
    sa = xcorr(A, mask, L)
    saa = xcorr(A ** 2, mask, L)
    r = []
    for g in (gx, gy):
        sg = xcorr(W[:, None], g * mask, L)[:, 0]
        sgg = xcorr(W[:, None], g ** 2 * mask, L)[:, 0]
        sag = xcorr(A, g * mask, L)
        num = n[:, None] * sag - sa * sg[:, None]
        den = np.sqrt(np.maximum(n[:, None] * saa - sa ** 2, 0) * np.maximum(n * sgg - sg ** 2, 0)[:, None])
        r.append(num / np.maximum(den, 1e-12))
    shifts = np.r_[np.arange(0, Nv), np.arange(-(L - Nv), 0)]
    keep = (n >= min_overlap) & (shifts > -Ng) & (shifts < Nv)      # n: in-trial game samples inside the video
    order = np.argsort(shifts[keep])
    rx, ry = r[0][keep][order], r[1][keep][order]                  # lags x (cells*2)
    s = np.sqrt((rx[:, 0::2] ** 2 + ry[:, 0::2] ** 2 + rx[:, 1::2] ** 2 + ry[:, 1::2] ** 2) / 2)
    return shifts[keep][order] / GRID_HZ, s


def global_offset(lags, s):
    """Lag (s) where the TOP_CELLS best cells score highest, that score, its margin over lags > 1 s away, the cells."""
    agg = -np.sort(-s, axis=1)[:, :TOP_CELLS].mean(1)
    k = int(np.nanargmax(agg))
    far = np.abs(lags - lags[k]) > 1.0
    cells = np.argsort(-s[k])[:TOP_CELLS]
    return lags[k], agg[k], agg[k] - np.nanmax(agg[far]), cells, agg


def window_score(F, cells, gx, gy, k0, a0, a1):
    """Mean rotation-invariant score of the chosen cells for game window [a0, a1) at lag k0."""
    v = F[k0 + a0:k0 + a1, cells, :]
    g = np.stack([gx[a0:a1], gy[a0:a1]], 1)
    if len(v) < a1 - a0 or len(v) < 10:
        return -np.inf
    v = v - v.mean(0)
    g = g - g.mean(0)
    nv = np.linalg.norm(v, axis=0) + 1e-9                 # cells x 2
    ng = np.linalg.norm(g, axis=0) + 1e-9                 # 2
    r = np.einsum("tcu,tg->cug", v, g) / (nv[:, :, None] * ng[None, None, :])
    return float(np.sqrt((r ** 2).sum((1, 2)) / 2).mean())


def refine(F, cells, gx, gy, lag, i0, i1, search_s=1.0, pad_s=0.5):
    """Correction to lag (s, within +/-search_s) that best fits game samples [i0, i1) padded by pad_s, and its score."""
    p = int(pad_s * GRID_HZ)
    a0, a1 = max(i0 - p, 0), min(i1 + p, len(gx))
    best = (-np.inf, 0.0)
    for d in np.arange(-search_s, search_s + 1e-9, 1 / GRID_HZ):
        k0 = int(round((lag + d) * GRID_HZ))
        if k0 + a0 < 0:
            continue
        s = window_score(F, cells, gx, gy, k0, a0, a1)
        if s > best[0]:
            best = (s, d)
    return best[1], best[0]


# ----------------------------------------------------------------------------- QA + clips
def qa_plot(path, title, F, cells, lag, gx, gy, segs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tv = np.arange(len(F)) / GRID_HZ
    tg = np.arange(len(gx)) / GRID_HZ + lag
    fig, axes = plt.subplots(2, 1, figsize=(15, 5.5), sharex=True, facecolor="white")
    c = cells[0]
    # the best cell, its flow component that best matches each game axis
    for ax, g, name in zip(axes, (gx, gy), ("x", "y")):
        k = int(round(lag * GRID_HZ))
        best, sig = 0, None
        for comp in (0, 1):
            a = F[:, c, comp]
            j0, j1 = max(0, -k), min(len(g), len(a) - k)
            if j1 - j0 < 10:
                continue
            rr = np.corrcoef(a[k + j0:k + j1], g[j0:j1])[0, 1]
            if abs(rr) > abs(best):
                best, sig = rr, a * np.sign(rr)
        if sig is not None:
            ax.plot(tv, sig / (sig.std() + 1e-9), lw=0.6, color="#2a78d6", label="video: best cell flow (z)")
        gz = g / (g[g != 0].std() + 1e-9) if np.any(g) else g
        ax.plot(tg, gz, lw=0.6, color="#eb6834", alpha=0.8, label=f"game: game-object v{name} (z), offset applied")
        for _, s in segs.iterrows():
            ax.axvspan(s.start_s, s.end_s, color="#1baf7a", alpha=0.12, lw=0)
        ax.set_ylim(-5, 5)
        ax.set_ylabel(f"v{name}")
        ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("video time (s)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def cut_clip(src, dst, start_s, n_frames):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{max(start_s, 0):.4f}", "-i", str(src), "-frames:v", str(n_frames),
                    "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-an", str(dst)], check=True)


def clip_name(pid, r):
    return f"{pid}_V{r.visit}_T{r.trial_in_visit}_P{r.path}_{r.hand}.mp4"


# ----------------------------------------------------------------------------- main
def visit_files(pdir, folder_key, visit):
    """Videos of a visit in the camera folders of one view (a visit can be split over several files)."""
    dirs = [d for d in pdir.iterdir() if d.is_dir() and folder_key.lower() in d.name.lower()]
    files = []
    for d in dirs:
        files += sorted(f for f in d.glob("*.mp4") if re.match(rf"V{visit}[_ ]", f.name))
    return files


def align_view(pid, view, paths, odir):
    rows, summary = [], []
    for visit, pv in paths.groupby("visit"):
        files = visit_files(RAW / pid, VIEWS[view], visit)
        if not files:
            print(f"[{pid} V{visit} {view}] no video")
            continue
        t0 = pv.start.min()                    # visit clock: s since the first trial start of the visit
        span = (max(pv.end.max(), pv.move_end.max()) - t0).total_seconds()
        gx, gy, mask = game_velocity(pv, t0, int(span * GRID_HZ) + 1)
        per_file = []
        for vp in files:
            f, t, fps = grid_flow(vp, odir / f"flow_{view}_{re.sub(r'[^A-Za-z0-9]+', '_', vp.stem)}.npz")
            F = to_grid(f, t)
            # in-trial game samples the video must overlap: 15 % of them for a visit split over several files,
            # else the smaller of 50 % of them and 50 % of the video; at least 20 s
            if len(files) > 1:
                min_overlap = max(0.15 * mask.sum(), 20 * GRID_HZ)
            else:
                min_overlap = max(min(0.5 * mask.sum(), 0.5 * len(F)), 20 * GRID_HZ)
            lags, s = cell_scores(F, gx, gy, mask, min_overlap)
            if len(lags) == 0:
                print(f"[{pid} V{visit} {view}] {vp.name}: too short to match; skipped")
                continue
            lag, peak, margin, cells, _ = global_offset(lags, s)
            print(f"[{pid} V{visit} {view}] {vp.name}: offset {lag:8.2f} s  score {peak:.3f}  margin {margin:.3f}")
            res = []
            for _, r in pv.iterrows():
                gs = (r.move_start - t0).total_seconds()         # clip = movement window
                ge = (r.move_end - t0).total_seconds()
                vs, ve = gs + lag, ge + lag
                if ve < 0 or vs > len(F) / GRID_HZ:
                    continue                                   # trial not in this file
                d, sc = refine(F, cells, gx, gy, lag, int(gs * GRID_HZ), int(ge * GRID_HZ))
                res.append(dict(row=r.row, clock_start=gs, clock_end=ge, refined=vs + d, refine_shift_s=d,
                                path_score=sc))
            res = pd.DataFrame(res)
            if res.empty:
                print(f"[{pid} V{visit} {view}] {vp.name}: no path inside this video; skipped")
                continue
            # line through the refined starts, refitted without those > 0.2 s off: video time = a * visit clock + b
            if len(res) >= 3:
                x, y = res.clock_start.values, res.refined.values
                keep = np.ones(len(res), bool)
                for _ in range(3):
                    a, b = np.polyfit(x[keep], y[keep], 1)
                    keep = np.abs(y - (a * x + b)) <= 0.2
                    if keep.sum() < 3:
                        a, b, keep = 1.0, lag, np.abs(y - (x + lag)) <= 0.2
                        break
            else:
                a, b, keep = 1.0, lag, np.ones(len(res), bool)
            res["start_s"] = a * res.clock_start + b
            res["end_s"] = a * res.clock_end + b
            res["residual_s"] = res.refined - res.start_s
            res["refine_outlier"] = ~keep
            dur = len(F) / GRID_HZ
            res["coverage"] = ((np.minimum(res.end_s, dur) - np.maximum(res.start_s, 0)).clip(0)
                               / (res.end_s - res.start_s))
            res["file"], res["fps"], res["offset_s"], res["drift_slope"] = vp, fps, b, a
            res["video_score"], res["video_margin"] = peak, margin
            per_file.append(res)
            summary.append(dict(pid=pid, visit=visit, view=view, video=vp.name, fps=fps,
                                duration_s=dur, offset_s=b, drift_slope=a, score=peak, margin=margin,
                                n_paths=len(res), n_outliers=int((~keep).sum()),
                                resid_sd_inliers_s=float(res.residual_s[keep].std()) if keep.sum() > 1 else np.nan))
            qa_plot(odir / f"alignment_V{visit}_{view}_{re.sub(r'[^A-Za-z0-9]+', '_', vp.stem)}.png",
                    f"{pid} V{visit} {view} ({vp.name}): offset {b:.2f} s, drift {a:.5f}, score {peak:.2f}, "
                    f"margin {margin:.2f} (green = paths)", F, cells, lag, gx, gy, res)
        if not per_file:
            continue
        # each trial from the file that covers it best
        allres = pd.concat(per_file).sort_values(["row", "coverage", "path_score"], ascending=[True, False, False])
        for _, r in allres.drop_duplicates("row").iterrows():
            rows.append(dict(view=view, video_file=str(r.file.relative_to(RAW)), fps=r.fps, row=r.row,
                             start_s=r.start_s, end_s=r.end_s,
                             start_frame=int(round(r.start_s * r.fps)),
                             n_frames=int(round((r.end_s - r.start_s) * r.fps)) + 1,
                             coverage=min(r.coverage, 1.0), offset_s=r.offset_s, drift_slope=r.drift_slope,
                             video_score=r.video_score, video_margin=r.video_margin,
                             path_score=r.path_score, refine_shift_s=r.refine_shift_s,
                             residual_s=r.residual_s, refine_outlier=bool(r.refine_outlier)))
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--no-cut", action="store_true", help="align only, do not write clips")
    args = ap.parse_args()
    pid = args.participant
    odir = ALIGNED / "video" / pid
    odir.mkdir(parents=True, exist_ok=True)
    paths = load_paths(pid)

    rows, summary = [], []
    for view in VIEWS:
        r, s = align_view(pid, view, paths, odir)
        rows += r
        summary += s
    seg = pd.DataFrame(rows).merge(
        paths[["row", "visit", "trial_in_visit", "group", "path", "hand", "start", "end", "move_start", "move_end",
               "review_status"]],
        on="row", how="left")
    seg.insert(0, "pid", pid)
    seg["clip"] = [f"clips/{v}/{clip_name(pid, r)}" for v, (_, r) in zip(seg.view, seg.iterrows())]
    seg = seg.sort_values(["view", "row"])
    # Need Review: ambiguous video match, trial off its video's line, or partial video
    bad = (seg.video_margin < 0.1) | (seg.refine_outlier == True) | (seg.coverage < 0.9)  # noqa: E712
    seg["video_status"] = np.where(bad, "Need Review", "OK")
    seg.to_csv(odir / "video_segments.csv", index=False, float_format="%.4f")
    pd.DataFrame(summary).to_csv(odir / "video_alignment_summary.csv", index=False, float_format="%.4f")
    missing = sorted(set(paths.row) - set(seg.row))
    print(f"[{pid}] {len(seg)} segments; paths without top/side video: {missing}")
    if not args.no_cut:
        for _, r in seg.iterrows():
            if r.coverage <= 0:
                continue
            cut_clip(RAW / r["video_file"], odir / r["clip"], r["start_s"], int(r["n_frames"]))
        print(f"[{pid}] clips written to {odir / 'clips'}")


if __name__ == "__main__":
    main()
