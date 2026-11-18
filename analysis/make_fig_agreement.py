"""Figure 4 of the paper: Bland-Altman agreement of the end-effector peak speed with VICON on the test trials.

For every test trial valid in all three streams (trajectory check passed), the fingertip peak speed of the kinematic
features (benchmark/e4_kinematics.py, results/E4_kin_paths.csv) from video is compared with VICON: Sapiens2-1B
zero-shot and with ViTMS, with the bias (mean difference) and the 95 % limits of agreement
(bias +- 1.96 SD) over all trials. Differences above 1300 mm/s are drawn as triangles at the top edge. The agreement
of ViTPose+-B with ViTMS, quoted in the text, is computed too (zero-shot ViTPose+-B has no fingertip keypoint).

Reads outputs/benchmark. Writes outputs/paper/fig/bland_altman.{pdf,png}, one column wide, and
outputs/paper/tables/bland_altman.csv.

    python analysis/make_fig_agreement.py
"""
import argparse
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
from alignment_panels import INK2, INK3, style                           # noqa: E402
from benchlib import BENCH, manifest, splits                             # noqa: E402
from figstyle import COLUMNWIDTH                                         # noqa: E402
import make_results as R                                                 # noqa: E402

DOC = ROOT / "outputs" / "paper"
TAB, FIG = DOC / "tables", DOC / "fig"
SHOWN, ALSO = "sapiens2_1b", "vitpose_b"                                 # zero-shot model of the figure; the other one quoted
YMAX = 1300                                                              # mm/s, top of the axis


def test_trials(m):
    return m[m.vicon_valid & m.top_valid & m.side_valid & ~m.trajectory_qa_failed & m.pid.isin(splits()["test"])]


def paired(trials, source, feature="fingertip_peak_speed"):
    """VICON and video values of a feature, one row per trial (pid, row) in which both exist."""
    k = pd.read_csv(BENCH / "results" / "E4_kin_paths.csv")
    keep = set(zip(trials.pid, trials.aligned_mat_row.astype(int)))
    k = k[[(p, int(r)) in keep for p, r in zip(k.pid, k.row)]]
    ref = k[k.source == "vicon"].set_index(["pid", "row"])[feature]
    x = k[k.source == source].set_index(["pid", "row"])[feature]
    return pd.concat([ref.rename("vicon"), x.rename("video")], axis=1).dropna()


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    _, q2 = R.zero_shot()
    trials = test_trials(manifest())
    ft = {v: R.best_with(q2, v) for v in (SHOWN, ALSO)}                  # the ViTMS variant of each model
    pairs = {s: paired(trials, s) for s in (SHOWN, ft[SHOWN], ft[ALSO])}
    rows = []
    for s, j in pairs.items():
        d = j.video - j.vicon
        bias, sd = d.mean(), d.std()
        rows.append(dict(source=s, trials=len(j), vicon_mean=j.vicon.mean(), bias=bias, loa_lo=bias - 1.96 * sd,
                         loa_hi=bias + 1.96 * sd, above_axis=int((d > YMAX).sum())))
    st = pd.DataFrame(rows).set_index("source")
    st.round(1).to_csv(TAB / "bland_altman.csv")
    print(st.round(0).to_string())

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 2.0))                  # printed at one column
    fig.subplots_adjust(left=0.14, right=0.905, bottom=0.2, top=0.97)
    ax.grid(False)
    for s, col in ((SHOWN, R.ZS_COL), (ft[SHOWN], R.FT_COL)):
        j, b = pairs[s], st.loc[s]
        mean, dif = (j.video + j.vicon) / 2, j.video - j.vicon
        inside = dif <= YMAX
        ax.scatter(mean[inside], dif[inside], s=6, color=col, alpha=0.75, linewidths=0, zorder=3)
        ax.scatter(mean[~inside], np.full((~inside).sum(), YMAX - 45), s=12, marker="^", color=col, linewidths=0, zorder=3)
        ax.axhline(b.bias, color=col, lw=1.2)
        for y in (b.loa_lo, b.loa_hi):
            ax.axhline(y, color=col, lw=0.8, ls=(0, (3, 2)))
        ax.text(1.0, b.bias, f" {b.bias:+.0f}", transform=ax.get_yaxis_transform(), color=col, va="center", ha="left")
    ax.axhline(0, color=INK3, lw=0.6)
    ax.set_ylim(-850, YMAX)
    ax.set_xlabel("mean of video and VICON (mm/s)")
    ax.set_ylabel("video − VICON (mm/s)")
    ax.legend(handles=[Line2D([], [], color=R.ZS_COL, lw=1.2, marker="o", ms=3, mew=0, label="Zero-shot"),
                       Line2D([], [], color=INK2, lw=1.2, label="bias"),
                       Line2D([], [], color=R.FT_COL, lw=1.2, marker="o", ms=3, mew=0, label="ViTMS"),
                       Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="95% LoA")],
              loc="lower center", ncol=4, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"bland_altman.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / "bland_altman.pdf", TAB / "bland_altman.csv")


if __name__ == "__main__":
    main()
