"""Screen and alignment panels of Figure 2 and the figure style, imported by make_fig_dataset.py.

screen_paths / draw_screen: one trial (a path in the identifiers) on the screen: game trajectory, VICON fingertip and
a model's two-view fingertip. alignment_traces / draw_alignment: the game, VICON and video clocks of one visit,
aligned from motion.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "alignment" / "video"))
from align_videos import GRID_HZ, game_velocity, load_paths               # noqa: E402
from benchlib import ALIGNED, MODEL_LABEL, manifest, markers              # noqa: E402
from e3_screen import apply_affine3, fit_affine3, game_paths, sources_for_path   # noqa: E402
from evaluate import ref_track                                            # noqa: E402
from figstyle import use_fonts                                            # noqa: E402

INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FT = "ft_b_full_pooled"


def style():
    plt.rcParams.update({"axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                         "axes.axisbelow": True, "legend.frameon": False})
    use_fonts()                                                           # Times at one size (figstyle.py)


# ----------------------------------------------------------------------------- screen
def screen_paths(r, cams, model=FT):
    """Game trajectory, VICON fingertip and the model's two-view fingertip of trial r on the screen (game px), the
    latter two mapped by the affine of E3 fitted on the other trials of the same visit and hand; and game px per mm."""
    m = manifest()
    grp = m[(m.pid == r.pid) & (m.visit == r.visit) & (m.hand == r.hand) & m.vicon_valid & m.top_valid & m.side_valid]
    gp = game_paths(r.pid)
    per = {}
    for _, rr in grp.iterrows():
        t, xy = gp[int(rr.aligned_mat_row)]
        src, _ = sources_for_path(rr, cams, t)
        per[int(rr.aligned_mat_row)] = (t, xy, src)
    key_ft = next(k for k in per[int(r.aligned_mat_row)][2] if k.startswith(MODEL_LABEL[model]) and k.endswith("2 views"))
    row = int(r.aligned_mat_row)
    t, xy, src = per[row]
    others = [k for k in per if k != row]
    out = {}
    for name, lab in (("Vicon FIN", "Vicon"), (key_ft, "ft")):
        Xf = np.concatenate([per[o][2][name][2] for o in others if name in per[o][2]])
        yf = np.concatenate([per[o][1] for o in others if name in per[o][2]])
        ok = ~np.isnan(Xf).any(1)
        A = fit_affine3(Xf[ok], yf[ok])
        est = src[name][2]
        good = ~np.isnan(est).any(1)
        pred = np.full_like(xy, np.nan)
        pred[good] = apply_affine3(A, est[good])
        out[lab] = pred
        if lab == "Vicon":
            scale = float(np.linalg.svd(A[:3].T, compute_uv=False).mean())      # game px per mm
    return t, xy, out, scale


def draw_screen(ax, t, xy, paths, scale, r, video_name="video (fine-tuned)"):
    g = xy / scale                                   # game y points up (origin at the screen centre), as the child sees it
    ax.plot(g[:, 0], g[:, 1], color=INK, lw=2.0, label="game trajectory", solid_capstyle="round")
    for lab, col, name in (("Vicon", SLOT[2], "VICON markers"), ("ft", SLOT[4], video_name)):
        p = paths[lab] / scale
        e = np.sqrt(np.nanmean(np.sum((p - g) ** 2, 1)))
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.0, label=f"{name}: {e:.1f} mm")
    ax.set_aspect("equal")
    ax.set_xlabel("screen x (mm)")
    ax.set_ylabel("screen y (mm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=1.3, labelspacing=0.2)
    ax.grid(False)
    ax.tick_params(length=2)


# ----------------------------------------------------------------------------- alignment
def alignment_traces(pid, visit, view, win_paths):
    """Game-object speed, VICON fingertip speed and the optical flow of the best grid cell of the visit's video on
    one clock (s from the first trial start, GRID_HZ), and the plotting window around the trials win_paths (rows)."""
    paths = load_paths(pid)
    pv = paths[paths.visit == visit].reset_index(drop=True)
    t0 = pv.start.min()
    n = int((pv.end.max() - t0).total_seconds() * GRID_HZ) + 1
    gx, gy, mask = game_velocity(pv, t0, n)
    obj = np.hypot(gx, gy)
    tg = np.arange(n) / GRID_HZ
    # VICON fingertip speed on the same grid
    mk, _ = markers(pid)
    fin = np.full(n, np.nan)
    for _, p in pv.iterrows():
        v = mk.get(int(p.row))
        if v is None:
            continue
        x = ref_track(v, p.hand[0], "fingertip")
        tt = v["t"] - t0.value / 1e9
        sp = np.linalg.norm(np.gradient(x, tt, axis=0), axis=1)
        ok = ~np.isnan(sp)
        i0, i1 = int(np.ceil(tt[0] * GRID_HZ)), min(int(tt[-1] * GRID_HZ), n)
        if i1 > i0 and ok.sum() > 10:
            fin[i0:i1] = np.interp(np.arange(i0, i1) / GRID_HZ, tt[ok], sp[ok])
    # optical flow of the best cell, on the visit clock after the video's offset and drift
    summ = pd.read_csv(ALIGNED / "video" / pid / "video_alignment_summary.csv")
    s = summ[(summ.visit == visit) & (summ.view == view)].iloc[0]
    z = np.load(ALIGNED / "video" / pid / f"flow_{view}_{Path(s.video).stem}.npz")
    tv = (z["t"] - s.offset_s) / s.drift_slope
    mag = np.linalg.norm(z["f"], axis=2)                                     # frames x cells
    F = np.full((n, mag.shape[1]), np.nan)
    inside = (tg >= tv[0]) & (tg <= tv[-1])
    for c in range(mag.shape[1]):
        F[inside, c] = np.interp(tg[inside], tv, mag[:, c])
    F = uniform_filter1d(np.nan_to_num(F), 3, axis=0)
    sel = mask > 0
    corr = [np.corrcoef(F[sel, c], obj[sel])[0, 1] for c in range(F.shape[1])]
    best = int(np.nanargmax(corr))
    # window around the requested trials
    w = pv[pv.row.isin(win_paths)]
    a, b = (w.move_start.min() - t0).total_seconds() - 2, (w.move_end.max() - t0).total_seconds() + 2
    a = max(a, tv[0] + 0.5)                                                  # where all three streams exist
    segs = [((p.move_start - t0).total_seconds(), (p.move_end - t0).total_seconds()) for _, p in pv.iterrows()]
    return tg, obj, fin, F[:, best], sel, (a, b), segs, corr[best]


def draw_alignment(ax, tg, obj, fin, flow, sel, win, segs, r_best, label_fs=5.5, short=False):
    """The three speed traces on the visit clock, each with its label above it; short: shorter labels, wider spacing."""
    a, b = win
    m = (tg >= a) & (tg <= b)
    for s0, s1 in segs:
        if s1 > a and s0 < b:
            ax.axvspan(max(s0, a), min(s1, b), color=SLOT[2], alpha=0.10, lw=0)

    def z(x):
        v = x[sel & ~np.isnan(x)]
        return (x - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    off, dy = ((11.0, 5.5, 0.0), 3.6) if short else ((9.2, 4.6, 0.0), 2.75)   # trace offsets; label height above its trace
    ax.plot(tg[m], np.clip(z(obj), -2, 3.0)[m] + off[0], color=INK, lw=0.7)
    ax.plot(tg[m], np.clip(z(fin), -2, 3.0)[m] + off[1], color=SLOT[2], lw=0.7)
    ax.plot(tg[m], np.clip(z(flow), -2, 3.0)[m] + off[2], color=SLOT[0], lw=0.7)
    if short:
        labels = ("Game: object speed", "VICON: fingertip marker speed", f"Top video: optical flow (r = {r_best:.2f})")
    else:
        labels = ("Game trajectory: object speed", "VICON: LFIN / RFIN marker speed",
                  f"Top-view video: optical flow, best grid cell (r = {r_best:.2f})")
    for y, col, lab in zip(off, (INK, SLOT[2], SLOT[0]), labels):
        ax.text(a + 0.4, y + dy, lab, color=col, fontsize=label_fs, va="center", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=1.2))
    ax.set_yticks([])
    ax.set_xlim(a, b)
    ax.set_ylim(-2.3, 15.6 if short else 13.0)
    ax.set_xlabel("game clock (s into the visit)" if short else "game clock (s into the visit); shaded: movement trials")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
