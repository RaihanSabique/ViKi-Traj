"""Bottom row of Figure 1 and Figure 2 (alignment) of the paper, from the benchmark outputs.

fig/streams_moment.pdf: one test trial (--pid study ID, --row aligned_mat_row) at one moment in VICON 3D, the top and
the side video, with the zero-shot Sapiens2-1B keypoints and their errors. fig/alignment.pdf: (a) the three clocks
aligned from motion over one VICON recording of the same participant whose four trials are valid in every stream
(--clock-visit, --clock-round), (b) the residual timing error of every valid trial, (c) the trial on the screen.
fig/video_alignment.pdf (supplement): one visit video aligned to the game by optical flow. Each figure is drawn at
its printed width (analysis/figstyle.py).

    HF_HUB_OFFLINE=1 python analysis/make_fig_dataset.py [--pid XXXXXXX110 --row 14] [--only-alignment]
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_panels import SLOT, alignment_traces, draw_alignment, draw_screen, screen_paths, style   # noqa: E402
from benchlib import MODEL_LABEL, manifest                               # noqa: E402
from calibrate import camera_id                                          # noqa: E402
from e7_path_frames import INK, draw_streams                             # noqa: E402
from evaluate import cameras                                             # noqa: E402
from figstyle import COLUMNWIDTH, FS, FS_LETTER, TEXTWIDTH, unify_text    # noqa: E402
from trial_overlays import FaceBlur                                      # noqa: E402

DOC = ROOT / "outputs" / "paper"
FIG = DOC / "fig"
MODEL = "sapiens2_1b"                                                   # zero-shot model shown in Figures 1 and 2


def camera_view(P):
    """(elev, azim, roll) of a 3D axes that looks along the principal axis of camera P (Hartley & Zisserman 6.2.2)."""
    M = P[:, :3]
    d = np.sign(np.linalg.det(M)) * M[2]
    cam = -d / np.linalg.norm(d)
    return float(np.degrees(np.arcsin(cam[2]))), float(np.degrees(np.arctan2(cam[1], cam[0]))), 0.0


def residual_panel(ax, m):
    bins = np.arange(0, 105, 5)
    hs = []
    for i, (col, flag, lab) in enumerate((("vicon_align_err_s", "vicon_valid", "VICON–game"), ("top_align_err_s", "top_valid", "top video–game"),
                                          ("side_align_err_s", "side_valid", "side video–game"))):
        e = np.clip(m.loc[m[flag], col].dropna() * 1000, 0, 100)
        ax.hist(e, bins=bins, histtype="step", lw=1.3, color=SLOT[i])
        hs.append(Line2D([], [], color=SLOT[i], lw=1.3, label=f"{lab} ({np.median(e):.0f} ms)"))      # lines, not boxes, in the legend
    ax.set_xlabel("residual timing error of a trial (ms)")
    ax.set_ylabel("trials")
    ax.legend(handles=hs, loc="upper right", handlelength=1.4, title="median", borderaxespad=0.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 320)                                  # room above the bars for the legend
    ax.set_yticks(range(0, 201, 50))
    ax.spines["left"].set_bounds(0, 200)


def fig_moment(r, cams, blur, moment=0.5):
    """fig/streams_moment.pdf: the trial at one moment in VICON 3D, the top and the side video, side by side; VICON in
    ink and Sapiens2-1B in one colour in all three panels, with the model's 2D errors in the video panels."""
    fig = plt.figure(figsize=(0.85 * TEXTWIDTH, 2.3))                   # the bottom row of Figure 1, at its printed width
    gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.0, right=0.995, top=0.81, bottom=0.12, width_ratios=(1.12, 1.0, 1.0))
    view3d = camera_view(cams[camera_id(r.pid, int(r.visit), "top", r["top_source_video"])])
    rec, hs, axes = draw_streams(fig, gs, r, MODEL, cams, blur, fs=0.72, label="VICON", view3d=view3d, errors3d=False, zoom3d=0.72,
                                 nticks3d=3, labelpad3d=-1, moments=(moment,), colors=["#2a78d6"], cell=lambda kind, c: gs[0, kind],
                                 row_labels=False, where=("", ""), vicon_col=INK,
                                 traj_label="End-effector Trajectory")
    x0 = [gs[0, k].get_position(fig).x0 + (0.03 if k == 0 else 0.0) for k in range(3)]
    # captions under the panels, on one baseline
    for k, title in enumerate(("VICON (3-D)", "Top video (2-D)", "Side video (2-D)")):
        p = gs[0, k].get_position(fig)
        fig.text((p.x0 + p.x1) / 2, 0.09, title, color=INK, va="bottom", ha="center")
    for k in (1, 2):
        axes[(k, 0)].set_anchor("N")                                       # video frames start right under their legend
    p3 = axes[(0, 0)].get_position()                                          # lift the 3D box clear of its caption
    axes[(0, 0)].set_position([p3.x0, p3.y0 + 0.06, p3.width, p3.height])
    # legends under the titles, left-aligned: the 3D sources over the 3D panel, the 2D ones over the side video
    for handles, k in ((hs[:3], 0), (hs[3:5], 2)):
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0[k] - 0.004, 0.995), ncol=1, frameon=False,
                   borderaxespad=0, borderpad=0, handlelength=2.2, labelspacing=0.25)
    unify_text(fig)                                                       # every text at FS, as in the other figures
    for a in (axes[(0, 0)].xaxis, axes[(0, 0)].yaxis, axes[(0, 0)].zaxis):
        a.set_tick_params(labelsize=FS)                                   # 3D tick labels are made at draw time
    # this figure writes 3D / 2D, not 3-D / 2-D
    for txt in fig.findobj(matplotlib.text.Text):
        if "3-D" in txt.get_text() or "2-D" in txt.get_text():
            txt.set_text(txt.get_text().replace("3-D", "3D").replace("2-D", "2D"))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"streams_moment.{ext}", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG / "streams_moment.pdf", pd_round(rec))


def fig_video_alignment(pid, visit, view="top", name="video_alignment"):
    """Supplementary figure: one visit video aligned to the game, from the cached optical flow and the stored alignment
    (video_alignment_summary.csv, video_segments.csv); the plot of align_videos.qa_plot at one column width."""
    import align_videos as AV
    paths = AV.load_paths(pid)
    pv = paths[paths.visit == visit]
    vp = AV.visit_files(AV.RAW / pid, AV.VIEWS[view], visit)[0]
    t0 = pv.start.min()
    span = (max(pv.end.max(), pv.move_end.max()) - t0).total_seconds()
    gx, gy, mask = AV.game_velocity(pv, t0, int(span * AV.GRID_HZ) + 1)
    odir = AV.ALIGNED / "video" / pid
    f, t, _ = AV.grid_flow(vp, odir / f"flow_{view}_{re.sub(r'[^A-Za-z0-9]+', '_', vp.stem)}.npz")    # cached flow
    F = AV.to_grid(f, t)
    lags, sc = AV.cell_scores(F, gx, gy, mask, max(min(0.5 * mask.sum(), 0.5 * len(F)), 20 * AV.GRID_HZ))
    lag, _, _, cells, _ = AV.global_offset(lags, sc)
    summ = pd.read_csv(odir / "video_alignment_summary.csv")
    sm = summ[(summ.visit == visit) & (summ.view == view) & (summ.video == vp.name)].iloc[0]
    seg = pd.read_csv(odir / "video_segments.csv")
    seg = seg[(seg.visit == visit) & (seg.view == view)]
    tv = np.arange(len(F)) / AV.GRID_HZ
    tg = np.arange(len(gx)) / AV.GRID_HZ + lag
    fig, axes = plt.subplots(2, 1, figsize=(COLUMNWIDTH, 2.3), sharex=True)   # printed at one column
    c, k = cells[0], int(round(lag * AV.GRID_HZ))
    for ax, g, comp_name in zip(axes, (gx, gy), ("x", "y")):
        best, sig = 0, None
        for comp in (0, 1):                                              # the flow component that best matches this game axis
            a = F[:, c, comp]
            j0, j1 = max(0, -k), min(len(g), len(a) - k)
            if j1 - j0 < 10:
                continue
            rr = np.corrcoef(a[k + j0:k + j1], g[j0:j1])[0, 1]
            if abs(rr) > abs(best):
                best, sig = rr, a * np.sign(rr)
        for _, r in seg.iterrows():
            ax.axvspan(r.start_s, r.end_s, color=SLOT[2], alpha=0.12, lw=0)
        ax.plot(tv, sig / (sig.std() + 1e-9), lw=0.5, color=SLOT[0], label="video: best-cell optical flow")
        gz = g / (g[g != 0].std() + 1e-9) if np.any(g) else g
        ax.plot(tg, gz, lw=0.5, color=SLOT[1], alpha=0.85, label="game: object velocity")
        ax.set_ylim(-5, 5)
        ax.set_yticks([-4, 0, 4])
        ax.set_ylabel(f"$v_{comp_name}$ (z)")
        ax.grid(False)
    axes[-1].set_xlabel("video time (s)")
    axes[0].set_title(f"offset {sm.offset_s:.2f} s, drift {sm.drift_slope:.4f}, score {sm.score:.2f}, margin {sm.margin:.2f}".replace("-", "\u2212"),
                      loc="left", pad=2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0,
               handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def pd_round(rec):
    return [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items()} for d in rec]


def fig_alignment(r, m, cams, clock_visit, clock_round):
    """(a) the three clocks over VICON recording clock_round of visit clock_visit, (b) the residual timing error of every
    valid trial, (c) the trial r on the screen."""
    # drawn at its printed width, 0.85 \textwidth
    fig = plt.figure(figsize=(0.85 * TEXTWIDTH, 2.35))
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.99, top=0.88, bottom=0.2, wspace=0.34, width_ratios=(1.6, 1.1, 0.9))
    ax_clock = fig.add_subplot(gs[0, 0])
    rows = m[(m.pid == r.pid) & (m.visit == clock_visit) & (m.trial_in_visit == clock_round)]
    draw_alignment(ax_clock, *alignment_traces(r.pid, clock_visit, "top", rows.aligned_mat_row.astype(int).tolist()),
                   label_fs=FS, short=True)
    ax_res = fig.add_subplot(gs[0, 1])
    residual_panel(ax_res, m)
    ax_scr = fig.add_subplot(gs[0, 2])
    t, xy, paths, scale = screen_paths(r, cams, MODEL)
    draw_screen(ax_scr, t, xy, paths, scale, r, video_name=f"video ({MODEL_LABEL[MODEL]})")
    ax_scr.set_anchor("S")                           # plot at the bottom of its cell, legend in the space above
    fig.canvas.draw()
    pos = ax_scr.get_position()
    ax_scr.legend(loc="lower left", bbox_to_anchor=(pos.x0 - 0.045, pos.y1 + 0.05), bbox_transform=fig.transFigure,
                  ncol=1, handlelength=1.2, labelspacing=0.2, borderaxespad=0)
    # panel letters on one line above the panels: (a) at its axes, (b) and (c) at the left edge of the panel
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    left = lambda ax: to_fig.transform(ax.get_tightbbox(fig.canvas.get_renderer()))[0, 0]   # noqa: E731
    for ax, lab, x in ((ax_clock, "a", ax_clock.get_position().x0), (ax_res, "b", left(ax_res)), (ax_scr, "c", left(ax_scr))):
        fig.text(x, 0.905, lab, fontweight="bold", fontsize=FS_LETTER, ha="left", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"alignment.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "alignment.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pid", default="XXXXXXX110", help="study ID of the trial")
    ap.add_argument("--row", type=int, default=14, help="aligned_mat_row of the trial")
    ap.add_argument("--moment", type=float, default=0.5, help="fraction of the trial for fig/streams_moment.pdf")
    ap.add_argument("--only-moment", action="store_true", help="write only fig/streams_moment.pdf")
    ap.add_argument("--only-alignment", action="store_true", help="write only fig/alignment.pdf")
    ap.add_argument("--only-video-alignment", action="store_true", help="write only fig/video_alignment.pdf")
    ap.add_argument("--video-pid", default="XXXXXXX103", help="study ID of the visit video in fig/video_alignment.pdf")
    ap.add_argument("--clock-visit", type=int, default=2, help="visit of the VICON recording in panel (a) of fig/alignment.pdf")
    ap.add_argument("--clock-round", type=int, default=3, help="VICON recording (round of trials) of that visit in panel (a)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    if args.only_video_alignment:
        fig_video_alignment(args.video_pid, 1, "top")
        return
    m = manifest()
    r = m[(m.pid == args.pid) & (m.aligned_mat_row == args.row)].iloc[0]
    cams, blur = cameras(), FaceBlur()
    if not args.only_alignment:
        fig_moment(r, cams, blur, args.moment)
    if not args.only_moment:
        fig_alignment(r, m, cams, args.clock_visit, args.clock_round)


if __name__ == "__main__":
    main()
