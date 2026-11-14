"""Bottom row of Figure 1 and Figure 2 (alignment) of the paper, from the benchmark outputs.

fig/streams_moment.pdf: one test trial (--pid study ID, --row aligned_mat_row) at one moment in VICON 3D, the top and
the side video, with the zero-shot Sapiens2-1B keypoints and their errors. fig/alignment.pdf: (a) the trial on the
screen, (b) the three clocks of its visit aligned from motion, (c) the residual timing error of every valid trial.

    HF_HUB_OFFLINE=1 python analysis/make_fig_dataset.py [--pid XXXXXXX110 --row 14]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_panels import (INK2, SLOT, alignment_traces, draw_alignment, draw_screen,  # noqa: E402
                              screen_paths, style)
from benchlib import MODEL_LABEL, manifest, markers                      # noqa: E402
from calibrate import camera_id                                          # noqa: E402
from e7_path_frames import INK, draw_streams                             # noqa: E402
from evaluate import cameras                                             # noqa: E402
from trial_overlays import FaceBlur                                      # noqa: E402

DOC = ROOT / "outputs" / "paper"
FIG = DOC / "fig"
MODEL = "sapiens2_1b"                                                   # zero-shot model shown in Figures 1 and 2


def public_ids():
    """Study ID -> released participant ID (sub-NN), from the map written by the release builders (dataset/)."""
    t = pd.read_csv(ROOT / "release" / "private" / "id_map.tsv", sep="\t")
    return dict(zip(t.study_id, t.participant_id))


def camera_view(P):
    """(elev, azim, roll) of a 3D axes that looks along the principal axis of camera P (Hartley & Zisserman 6.2.2)."""
    M = P[:, :3]
    d = np.sign(np.linalg.det(M)) * M[2]
    cam = -d / np.linalg.norm(d)
    return float(np.degrees(np.arcsin(cam[2]))), float(np.degrees(np.arctan2(cam[1], cam[0]))), 0.0


def residual_panel(ax, m):
    bins = np.arange(0, 105, 5)
    for i, (col, flag, lab) in enumerate((("vicon_align_err_s", "vicon_valid", "Vicon–game"), ("top_align_err_s", "top_valid", "top video–game"),
                                          ("side_align_err_s", "side_valid", "side video–game"))):
        e = np.clip(m.loc[m[flag], col].dropna() * 1000, 0, 100)
        ax.hist(e, bins=bins, histtype="step", lw=1.3, color=SLOT[i], label=f"{lab} (median {np.median(e):.0f} ms)")
    ax.set_xlabel("residual timing error of a trial (ms)")
    ax.set_ylabel("trials")
    ax.legend(loc="upper right", handlelength=1.4)
    ax.set_xlim(0, 100)


def fig_moment(r, cams, blur, moment=0.5):
    """fig/streams_moment.pdf: the trial at one moment in VICON 3D, the top and the side video, side by side; VICON in
    ink and Sapiens2-1B in one colour in all three panels, with the model's 2D errors in the video panels."""
    fig = plt.figure(figsize=(7.1, 2.85))
    gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.0, right=0.995, top=0.78, bottom=0.14, width_ratios=(1.12, 1.0, 1.0))
    view3d = camera_view(cams[camera_id(r.pid, int(r.visit), "top", r["top_source_video"])])
    rec, hs, axes = draw_streams(fig, gs, r, MODEL, cams, blur, fs=0.72, label="VICON", view3d=view3d, errors3d=False, zoom3d=0.8,
                                 nticks3d=4, labelpad3d=-4, moments=(moment,), colors=["#2a78d6"], cell=lambda kind, c: gs[0, kind],
                                 row_labels=False, where=("", ""), vicon_col=INK,
                                 traj_label="End-effector Trajectory")
    t = rec[0]["t_s"]
    x0 = [gs[0, k].get_position(fig).x0 + (0.03 if k == 0 else 0.0) for k in range(3)]
    # captions under the panels, on one baseline
    for k, title in enumerate(("VICON (3-D)", "Top video (2-D)", "Side video (2-D)")):
        p = gs[0, k].get_position(fig)
        fig.text((p.x0 + p.x1) / 2, 0.09, title, fontsize=7.5, color=INK, va="bottom", ha="center")
    for k in (1, 2):
        axes[(k, 0)].set_anchor("N")                                       # video frames start right under their legend
    p3 = axes[(0, 0)].get_position()                                          # lift the 3D box clear of its caption
    axes[(0, 0)].set_position([p3.x0, p3.y0 + 0.035, p3.width, p3.height])
    # legends under the titles, left-aligned: the 3D sources over the 3D panel, the 2D ones over the side video
    for handles, k in ((hs[:3], 0), (hs[3:5], 2)):
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0[k] - 0.004, 0.935), ncol=1, fontsize=7.5, frameon=False,
                   borderaxespad=0, borderpad=0, handlelength=2.2, labelspacing=0.25)
    sub = public_ids()[r.pid]
    v = markers(r.pid)[0][int(r.aligned_mat_row)]
    fig.text(0.005, 0.995, f"{sub} ({'TD' if not r.motor_impaired else 'UCP'} test child), {r.hand.lower()} hand, visit {r.visit}, "
             f"trial {r.path}, t = {t:.1f} s of {v['t'][-1] - v['t'][0]:.1f} s ({int(moment * 100)} % of the trial): "
             f"VICON vs {MODEL_LABEL[MODEL]} (zero-shot)", fontsize=6.5, color=INK2, va="top")
    # this figure writes 3D / 2D, not 3-D / 2-D
    for txt in fig.findobj(matplotlib.text.Text):
        if "3-D" in txt.get_text() or "2-D" in txt.get_text():
            txt.set_text(txt.get_text().replace("3-D", "3D").replace("2-D", "2D"))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"streams_moment.{ext}", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG / "streams_moment.pdf", pd_round(rec))


def pd_round(rec):
    return [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items()} for d in rec]


def fig_alignment(r, m, cams):
    fig = plt.figure(figsize=(7.1, 2.35))
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.99, top=0.88, bottom=0.2, wspace=0.3, width_ratios=(0.9, 1.75, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    t, xy, paths, scale = screen_paths(r, cams, MODEL)
    draw_screen(ax, t, xy, paths, scale, r, video_name=f"video ({MODEL_LABEL[MODEL]})")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.08, -0.3), ncol=1, handlelength=1.2, labelspacing=0.2)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    trial_rows = m[(m.pid == r.pid) & (m.visit == r.visit) & (m.trial_in_visit == r.trial_in_visit)].aligned_mat_row.astype(int).tolist()
    draw_alignment(ax, *alignment_traces(r.pid, int(r.visit), "top", trial_rows))
    ax.set_title("b", loc="left", fontweight="bold", fontsize=8)
    ax = fig.add_subplot(gs[0, 2])
    residual_panel(ax, m)
    ax.set_title("c", loc="left", fontweight="bold", fontsize=8)
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
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    m = manifest()
    r = m[(m.pid == args.pid) & (m.aligned_mat_row == args.row)].iloc[0]
    cams, blur = cameras(), FaceBlur()
    fig_moment(r, cams, blur, args.moment)
    if args.only_moment:
        return
    fig_alignment(r, m, cams)


if __name__ == "__main__":
    main()
