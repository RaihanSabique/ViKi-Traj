"""Dataset tables, text macros and the dataset-QA figure of the paper, from the benchmark manifest.

Reads outputs/game_vicon_aligned/reaching (benchmark_manifest.csv, vicon_trial_frames.csv,
video/video_alignment_summary_all.csv) and release/private/id_map.tsv. Writes outputs/paper/tables/
{validity_by_group, alignment_quality, stats_macros}.tex and outputs/paper/fig/dataset_qa.pdf.

    python analysis/make_tables.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "outputs/game_vicon_aligned/reaching"
DOC = ROOT / "outputs" / "paper"
TAB, FIG = DOC / "tables", DOC / "fig"

INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"   # text, labels, axes, grid, page
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]   # series colours: blue, orange, aqua, yellow, violet


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def fmt_int(n):
    return f"{int(n):,}".replace(",", "{,}")


def public_ids():
    """Study ID -> released participant ID (sub-NN), from the map written by the release builders (dataset/)."""
    t = pd.read_csv(ROOT / "release" / "private" / "id_map.tsv", sep="\t")
    return dict(zip(t.study_id, t.participant_id))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(AB / "benchmark_manifest.csv")
    vs = pd.read_csv(AB / "video/video_alignment_summary_all.csv")
    vt = pd.read_csv(AB / "vicon_trial_frames.csv")
    m["group"] = np.where(m.motor_impaired, "UCP", "TD")
    m["all3"] = m.vicon_valid & m.top_valid & m.side_valid
    m["vicon_top"] = m.vicon_valid & m.top_valid
    m["vicon_side"] = m.vicon_valid & m.side_valid

    S = {}
    pid = m.groupby("pid").group.first()
    S["nPid"], S["nPidTD"], S["nPidUCP"] = len(pid), int((pid == "TD").sum()), int((pid == "UCP").sum())
    S["nPaths"] = len(m)
    S["nPathsPerPid"] = int(m.groupby("pid").size().iloc[0])
    S["nVisits"] = int(m.visit.nunique())
    S["nViconTrials"] = len(vt)
    S["viconTrialMin"] = round(vt.n_frames.sum() / 100 / 60)
    S["moveMedian"], S["moveQone"], S["moveQthree"] = [round(m.move_duration_s.quantile(q), 1) for q in (.5, .25, .75)]
    S["moveMin"], S["moveMax"] = round(m.move_duration_s.min(), 1), round(m.move_duration_s.max(), 1)
    S["moveTotalMin"] = round(m.move_duration_s.sum() / 60)
    S["moveMedianTD"] = round(m[m.group == "TD"].move_duration_s.median(), 1)
    S["moveMedianUCP"] = round(m[m.group == "UCP"].move_duration_s.median(), 1)
    S["viconFramesK"] = round(m.vicon_frames.sum() / 1000)
    S["topFramesK"] = round(m.top_n_frames.sum() / 1000)
    S["sideFramesK"] = round(m.side_n_frames.sum() / 1000)
    S["topValidFramesK"] = round(m[m.top_valid].top_n_frames.sum() / 1000)
    S["sideValidFramesK"] = round(m[m.side_valid].side_n_frames.sum() / 1000)
    S["nVideos"], S["nVideosTop"], S["nVideosSide"] = len(vs), int((vs.view == "top").sum()), int((vs.view == "side").sum())
    S["videoMinutes"] = round(vs.duration_s.sum() / 60)
    for col, key in [("vicon_valid", "Vicon"), ("top_valid", "Top"), ("side_valid", "Side"),
                     ("benchmark_valid", "Bench"), ("all3", "Three"), ("vicon_top", "ViconTop"), ("vicon_side", "ViconSide")]:
        S[f"n{key}"] = int(m[col].sum())
        S[f"pct{key}"] = round(100 * m[col].mean())
        S[f"nPid{key}"] = int(m[m[col]].pid.nunique())
    S["minThree"] = round(m[m.all3].move_duration_s.sum() / 60)
    S["minBench"] = round(m[m.benchmark_valid].move_duration_s.sum() / 60)
    for g in ("TD", "UCP"):
        d = m[m.group == g]
        for col, key in [("vicon_valid", "Vicon"), ("top_valid", "Top"), ("side_valid", "Side"), ("all3", "Three")]:
            S[f"pct{key}{g}"] = round(100 * d[col].mean())
    # alignment residuals (ms) of the trials that pass each stream's validity test
    for col, flag, key in [("vicon_align_err_s", "vicon_valid", "Vicon"), ("top_align_err_s", "top_valid", "Top"),
                           ("side_align_err_s", "side_valid", "Side")]:
        e = m.loc[m[flag], col].dropna() * 1000
        S[f"err{key}Med"], S[f"err{key}Pnn"] = round(e.median()), round(e.quantile(.9))
        S[f"err{key}Within"] = round(100 * (e <= 50).mean())
    S["videoScoreMed"] = round(vs.score.median(), 2)
    S["videoMarginMed"] = round(vs.margin.median(), 2)
    S["videoResidMs"] = round(vs.resid_sd_inliers_s.median() * 1000)
    S["videoDriftMaxPct"] = round(100 * (vs.drift_slope - 1).abs().max(), 2)
    S["nManualOnly"] = 4          # participants aligned to VICON from the manual marks only (run_alignment.m)
    S["nVideoAmbiguous"] = int((vs.margin < 0.1).sum())   # ambiguous video match: margin < 0.1, as in align_videos.py

    # ------------------------------------------------------------------ macros
    with open(TAB / "stats_macros.tex", "w") as f:
        f.write("% generated by analysis/make_tables.py -- do not edit\n")
        for k, v in S.items():
            f.write(macro(k, fmt_int(v) if isinstance(v, (int, np.integer)) and abs(v) >= 1000 else v))

    # ------------------------------------------------------------------ Table: validity by group
    with open(TAB / "validity_by_group.tex", "w") as f:
        f.write("% generated by analysis/make_tables.py\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n")
        f.write("Group & Trials & VICON & Top & Side & All three \\\\\n\\midrule\n")
        for grp in ("TD", "UCP", "All"):
            d = m if grp == "All" else m[m.group == grp]
            cells = [f"{int(d[c].sum())} ({100 * d[c].mean():.0f}\\%)" for c in
                     ("vicon_valid", "top_valid", "side_valid", "all3")]
            f.write(f"{grp} & {len(d)} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------------------ Table: alignment quality
    with open(TAB / "alignment_quality.tex", "w") as f:
        f.write("% generated by analysis/make_tables.py\n")
        f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
        f.write("Stream pair & Signal matched & Median & P90 & $\\le$50 ms \\\\\n\\midrule\n")
        for key, pair, sig in [("Vicon", "VICON--game", "fingertip speed vs.\\ game-object speed; manual starts"),
                               ("Top", "Top video--game", "optical flow vs.\\ game-object velocity"),
                               ("Side", "Side video--game", "optical flow vs.\\ game-object velocity")]:
            f.write(f"{pair} & {sig} & {S[f'err{key}Med']} ms & {S[f'err{key}Pnn']} ms & {S[f'err{key}Within']}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------------------ Figure: dataset QA
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7})
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), dpi=200, gridspec_kw=dict(width_ratios=(0.95, 1.4, 0.95)))
    fig.patch.set_facecolor(SURF)
    for a_ in axs.ravel():
        a_.set_facecolor(SURF)
    # (a) trial duration by group
    a1 = axs[0]
    bins = np.arange(0, 42, 2)
    for i, grp in enumerate(("TD", "UCP")):
        d = np.clip(m[m.group == grp].move_duration_s, 0, 40)
        a1.hist(d, bins=bins, histtype="step", lw=1.6, color=CAT[i], label=f"{grp} (n = {int((pid == grp).sum())})")
    a1.set_xlabel("trial duration (s, clipped at 40)")
    a1.set_ylabel("trials")
    a1.set_title("a  Trial duration by group", loc="left", fontsize=8.5, fontweight="bold")
    a1.legend(loc="upper right")
    # (b) usable trials per participant and stream
    a2 = axs[1]
    per = m.groupby("pid").agg(group=("group", "first"), vicon=("vicon_valid", "sum"), top=("top_valid", "sum"),
                               side=("side_valid", "sum"), three=("all3", "sum"))
    per = per.sort_values(["group", "three"], ascending=[True, False])
    x = np.arange(len(per))
    for i, (col, lab, mk) in enumerate([("vicon", "Vicon", "o"), ("top", "top", "s"), ("side", "side", "^"), ("three", "all three", "D")]):
        a2.scatter(x, per[col], s=14, marker=mk, color=CAT[i], label=lab, zorder=3, linewidths=0)
    a2.set_xticks(x)
    sub = public_ids()
    a2.set_xticklabels([sub[p] for p in per.index], rotation=90, fontsize=6)
    split = int((per.group == "TD").sum()) - 0.5
    a2.axvline(split, color=INK3, lw=0.8, ls=":")
    a2.text(split / 2, 25.5, "TD", ha="center", color=INK2, fontsize=7)
    a2.text(split + (len(per) - split) / 2, 25.5, "UCP", ha="center", color=INK2, fontsize=7)
    a2.set_ylim(0, 27)
    a2.set_ylabel("usable trials (of 24)")
    a2.set_xlabel("participant")
    a2.set_title("b  Usable trials per participant", loc="left", fontsize=8.5, fontweight="bold")
    a2.legend(loc="lower left", ncol=4, columnspacing=0.8, handletextpad=0.3)
    # (c) video match quality
    a3 = axs[2]
    for i, view in enumerate(("top", "side")):
        d = vs[vs.view == view]
        a3.scatter(d.margin, d.score, s=16, color=CAT[i], label=f"{view} videos (n = {len(d)})", linewidths=0, zorder=3)
    a3.axvline(0.1, color=INK3, lw=0.8, ls="--")
    a3.text(0.105, 0.97, "ambiguous\nmatch", color=INK2, fontsize=6.5, va="top")
    a3.set_xlabel("margin over 2nd-best lag")
    a3.set_ylabel("flow–game-object correlation")
    a3.set_title("c  Video–game match", loc="left", fontsize=8.5, fontweight="bold")
    a3.legend(loc="lower right", bbox_to_anchor=(1.02, 0.0), fontsize=6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "dataset_qa.pdf", facecolor=SURF)
    fig.savefig(FIG / "dataset_qa.png", facecolor=SURF)
    plt.close(fig)
    print("wrote", TAB, FIG / "dataset_qa.pdf")
    print(json.dumps(S, indent=0, default=float)[:3000])


if __name__ == "__main__":
    main()
