"""Result tables, figures and text macros of the paper, from the benchmark outputs (E1-E7).

Reads outputs/benchmark (results/*.csv, splits.json, cameras/, finetune/, labels/, tracks3d/, reports/E7/) and
release/private/id_map.tsv. Writes outputs/paper/tables/{res_*, splits}.tex, where res_macros.tex holds the numbers
quoted in the text, and outputs/paper/fig/{results_zs_ft, analyses, kin_icc, qualitative}.pdf. Scores: per-participant
median, mean over participants, bootstrap 95 % CI; test split (8 participants), active arm, unless stated.

    python analysis/make_results.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from benchlib import BENCH, JOINT_NAMES, MODEL_LABEL, manifest, splits    # noqa: E402

RES = BENCH / "results"
DOC = ROOT / "outputs" / "paper"
TAB, FIG = DOC / "tables", DOC / "fig"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# zero-shot models, and one colour per model
ZS = ["vitpose_b", "rtmpose_wb", "rtmpose_hand", "dwpose_l", "rtmw_x", "sapiens2_0.4b", "sapiens2_1b", "blazepose"]
MCOL = {"vitpose_b": "#2a78d6", "rtmpose_wb": "#eb6834", "rtmpose_hand": "#8c5a2b", "dwpose_l": "#eda100", "rtmw_x": "#6250d6",
        "sapiens2_0.4b": "#1baf7a", "sapiens2_1b": "#008300", "blazepose": "#e87ba4"}
LAB = {"vitpose_b": "ViTPose+-B", "vitpose_l": "ViTPose+-L", "vitpose_h": "ViTPose+-H", "rtmpose_wb": "RTMPose-WB", "dwpose_l": "DWPose-L",
       "rtmw_x": "RTMW-X", "rtmpose_hand": "RTMPose-m hand", "sapiens2_0.4b": "Sapiens2-0.4B", "sapiens2_1b": "Sapiens2-1B",
       "blazepose": "BlazePose+Hands", "ft_b_head_top": "head, top", "ft_b_head_side": "head, side", "ft_b_head_pooled": "head, both views",
       "ft_b_full_top": "full, top", "ft_b_full_side": "full, side", "ft_b_full_pooled": "full, both views",
       "ft_b_full_pooled_hasmask": "full, both views, all labels", "seq_sapiens2_1b": "Sapiens2-1B + temporal head",
       "seq_rtmw_x": "RTMW-X + temporal head", "ft_b_full_pooled_n4": "full, both, 4 children", "ft_b_full_pooled_n8": "full, both, 8 children",
       "ft_b_full_pooled_n12": "full, both, 12 children", "ft_b_full_pooled_shift67": "full, both, labels +67 ms",
       "ft_b_full_pooled_shift200": "full, both, labels +200 ms",
       "ft_b_full_pooled_traj": "full, both views, markers + trajectory (fill)", "ft_b_full_pooled_trajr": "full, both views, trajectory replaces the fingertip",
       "seq_ft_b_full_pooled_traj": "\\method{} + temporal head"}
LAB.update({"ft_l_full_pooled_traj": "ViTPose+-L, full, both views, markers + trajectory", "ft_h_full_pooled_traj": "ViTPose+-H, full, both views, markers + trajectory",
            "seq_ft_l_full_pooled_traj": "ViTPose+-L \\method{} + temporal head", "seq_ft_h_full_pooled_traj": "ViTPose+-H \\method{} + temporal head"})
for _v in ZS:
    LAB[f"seq_{_v}"] = LAB[_v] + " + temporal head"
LAB.update({"ft_sapiens2_0.4b_head": "Sapiens2-0.4B, head, markers + trajectory", "ft_sapiens2_1b_head": "Sapiens2-1B, head, markers + trajectory",
            "seq_ft_sapiens2_0.4b_head": "Sapiens2-0.4B \\method{} + temporal head", "seq_ft_sapiens2_1b_head": "Sapiens2-1B \\method{} + temporal head"})
FIGLAB = dict(LAB, **{"ft_b_head_pooled": "ViTPose+-B FT head", "ft_b_full_pooled": "ViTPose+-B FT full",
                      "seq_sapiens2_1b": "Sapiens2-1B + temporal", "seq_rtmw_x": "RTMW-X + temporal", "ft_b_full_pooled_traj": "ViTPose+-B FT + trajectory",
                      "ft_b_full_pooled_trajr": "ViTPose+-B FT, trajectory only", "seq_ft_b_full_pooled_traj": "FT + trajectory + temporal",
                      "ft_sapiens2_0.4b_head": "Sapiens2-0.4B FT head", "ft_sapiens2_1b_head": "Sapiens2-1B FT head",
                      "seq_ft_sapiens2_0.4b_head": "Sapiens2-0.4B FT + temporal", "seq_ft_sapiens2_1b_head": "Sapiens2-1B FT + temporal"})
for _v in ZS:
    FIGLAB[f"seq_{_v}"] = FIGLAB[_v] + " + temporal"
WITH = {"vitpose_b": ["ft_b_head_top", "ft_b_head_side", "ft_b_head_pooled", "ft_b_full_top", "ft_b_full_side", "ft_b_full_pooled", "ft_b_full_pooled_hasmask",
                      "ft_b_full_pooled_traj", "ft_b_full_pooled_trajr", "seq_ft_b_full_pooled_traj", "seq_vitpose_b"],
        "vitpose_l": ["ft_l_full_pooled_traj", "seq_ft_l_full_pooled_traj", "seq_vitpose_l"],
        "vitpose_h": ["ft_h_full_pooled_traj", "seq_ft_h_full_pooled_traj", "seq_vitpose_h"]}
for _v in ZS:
    WITH.setdefault(_v, [f"seq_{_v}"])
WITH["sapiens2_0.4b"] += ["ft_sapiens2_0.4b_head", "seq_ft_sapiens2_0.4b_head"]
WITH["sapiens2_1b"] += ["ft_sapiens2_1b_head", "seq_ft_sapiens2_1b_head"]


def with_desc(v):
    if v.startswith("seq_ft_"):
        return ("head fine-tuned" if "sapiens" in v else "fine-tuned") + ", trajectory labels, temporal head"
    if v.startswith("ft_"):
        return ("head fine-tuned" if "sapiens" in v else "fine-tuned") + (", trajectory labels" if "traj" in v or "sapiens" in v else "")
    return "temporal head"


def best_with(q2, v):
    """The ViconTraj-MS variant of zero-shot model v with the lowest mean 3D fingertip and wrist median in q2."""
    g = q2
    cand = [w for w in WITH[v] if w in set(g.variant)]
    if not cand:
        return None
    def key(w):
        t = g[(g.variant == w) & (g.joint == "fingertip")].median_mm
        r = g[(g.variant == w) & (g.joint == "wrist")].median_mm
        t = float(t.iloc[0]) if len(t) and np.isfinite(t.iloc[0]) else np.inf
        r = float(r.iloc[0]) if len(r) and np.isfinite(r.iloc[0]) else np.inf
        return ((t + r) / 2, t)
    return min(cand, key=key)

JLAB = {"shoulder": "Shoulder", "elbow": "Elbow", "wrist": "Wrist", "hand": "Hand", "fingertip": "Fingertip"}
MARK = {"vitpose_b": "o", "rtmpose_wb": "s", "dwpose_l": "P", "rtmw_x": "^", "sapiens2_1b": "D", "blazepose": "X", "ft_b_full_pooled": "s"}
MACROS = {}


def family_color(v):
    v = str(v)
    for pre, k in (("ft_", 4), ("seq_", 5), ("vitpose", 0), ("rtm", 1), ("dwpose", 1), ("sapiens", 2)):
        if v.startswith(pre):
            return SLOT[k]
    return SLOT[3]


def macro(name, value):
    assert name.isalpha(), name
    MACROS[name] = value


def f1(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def read(name):
    return pd.read_csv(RES / name)


def public_ids():
    """Study ID -> released participant ID (sub-NN), from the map written by the release builders (dataset/)."""
    t = pd.read_csv(ROOT / "release" / "private" / "id_map.tsv", sep="\t")
    return dict(zip(t.study_id, t.participant_id))


def write_table(name, header, rows, colspec, note=None):
    lines = [f"% generated by analysis/make_results.py -- do not edit", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
    lines += rows + ["\\bottomrule", "\\end{tabular}"]
    (TAB / name).write_text("\n".join(lines) + "\n")


def cell(v, best=False, cov=None, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{nd}f}"
    if best:
        s = f"\\textbf{{{s}}}"
    if cov is not None and cov < 0.9:                # coverage below 90 %: printed as a superscript, never ranked best
        s += f"$^{{{cov * 100:.0f}\\%}}$"
    return s


# ----------------------------------------------------------------------------- E1 / E2
def zero_shot():
    e1, e2 = read("E1_2d.csv"), read("E2_3d.csv")
    q1 = e1[(e1.split == "test") & (e1.group == "all") & (e1.role == "active") & (e1.labels == "visible")]
    q2 = e2[(e2.split == "test") & (e2.group == "all") & (e2.role == "active")]
    macro("nTestPid", int(q2.n_participants.max()))
    macro("nTestClipsTop", int(q1[q1.view == "top"].n_paths.max()))
    macro("nTestClipsSide", int(q1[q1.view == "side"].n_paths.max()))
    macro("nTestThree", int(q2.n_paths.max()))
    blocks = {"top": q1[q1.view == "top"], "side": q1[q1.view == "side"], "3D": q2}
    best = {(v, j): d[(d.joint == j) & d.variant.isin(ZS) & (d.coverage >= 0.9)].median_mm.min() for v, d in blocks.items() for j in JOINT_NAMES}
    rows = []
    for m in ZS:
        for i, (view, d) in enumerate(blocks.items()):
            g = d[d.variant == m].set_index("joint")
            cells = [cell(g.median_mm.get(j, np.nan), best=np.isclose(g.median_mm.get(j, np.nan), best[(view, j)])) for j in JOINT_NAMES]
            rows.append(f"{LAB[m] if i == 0 else ''} & {view} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace[1.5pt]")
    write_table("res_zeroshot.tex", "Model & View & Shoulder & Elbow & Wrist & Hand & Fingertip \\\\", rows[:-1], "@{}llccccc@{}")
    # macros: best zero-shot per view and joint, and a few named ones
    for view, d in blocks.items():
        for j in JOINT_NAMES:
            z = d[(d.joint == j) & d.variant.isin(ZS)].sort_values("median_mm")
            if len(z):
                key = {"top": "Top", "side": "Side", "3D": "ThreeD"}[view] + JLAB[j]
                macro("zs" + key, f1(z.iloc[0].median_mm))
                macro("zs" + key + "Model", LAB[z.iloc[0].variant].replace("$^\\dagger$", ""))
    g = q2.set_index(["variant", "joint"])
    macro("zsHandCov", f"{100 * g.loc[('rtmpose_hand', 'fingertip'), 'coverage']:.0f}")
    macro("zsHandTipThreeD", f1(g.loc[("rtmpose_hand", "fingertip"), "median_mm"]))
    macro("zsSapTipThreeD", f1(g.loc[("sapiens2_1b", "fingertip"), "median_mm"]))
    macro("zsSapTipThreeDPnn", f1(g.loc[("sapiens2_1b", "fingertip"), "p90_mm"]))
    macro("zsRtmwTipThreeD", f1(g.loc[("rtmw_x", "fingertip"), "median_mm"]))
    macro("zsSapWristThreeD", f1(g.loc[("sapiens2_1b", "wrist"), "median_mm"]))
    macro("zsBlazeWristThreeD", f1(g.loc[("blazepose", "wrist"), "median_mm"]))
    macro("zsBlazeElbowCov", f"{100 * g.loc[('blazepose', 'elbow'), 'coverage']:.0f}")
    macro("zsBlazeTipCov", f"{100 * g.loc[('blazepose', 'fingertip'), 'coverage']:.0f}")
    wr = q2[(q2.joint == "wrist") & q2.variant.isin(["vitpose_b", "rtmpose_wb", "dwpose_l", "rtmw_x", "sapiens2_0.4b", "sapiens2_1b"])]
    macro("zsWristThreeDLo", f1(wr.median_mm.min()))
    macro("zsWristThreeDHi", f1(wr.median_mm.max()))
    t = q1[(q1.view == "top") & (q1.joint == "fingertip") & q1.variant.isin(["rtmpose_wb", "dwpose_l", "rtmw_x"])]
    macro("zsRtmFamTipTopLo", f1(t.median_mm.min()))
    macro("zsRtmFamTipTopHi", f1(t.median_mm.max()))
    # TD vs UCP (all participants), zero-shot 3D wrist / fingertip
    a2 = e2[(e2.split == "all") & (e2.role == "active")]
    for v, k in (("sapiens2_1b", "Sap"), ("rtmw_x", "Rtmw")):
        for j in ("wrist", "fingertip"):
            for grp in ("TD", "UCP"):
                r = a2[(a2.variant == v) & (a2.joint == j) & (a2.group == grp)].iloc[0]
                macro(f"zs{k}{JLAB[j]}{grp}", f1(r.median_mm))
    macro("nAllPidTD", int(a2[a2.group == "TD"].n_participants.max()))
    macro("nAllPidUCP", int(a2[a2.group == "UCP"].n_participants.max()))
    macro("nAllThree", int(a2[a2.group == "all"].n_paths.max()))
    # resting arm, top view, zero-shot: ViTPose+-B wrist and Sapiens2-1B fingertip
    r = e1[(e1.split == "test") & (e1.group == "all") & (e1.role == "resting") & (e1.labels == "visible") & (e1.view == "top")]
    macro("zsRestWristTop", f1(r[(r.variant == "vitpose_b") & (r.joint == "wrist")].median_mm.iloc[0]))
    macro("zsRestTipTop", f1(r[(r.variant == "sapiens2_1b") & (r.joint == "fingertip")].median_mm.iloc[0]))
    # all labels vs visible labels: Sapiens2-1B fingertip (top, side), fine-tuned ViTPose+-B (side), label counts
    al = e1[(e1.split == "test") & (e1.group == "all") & (e1.role == "active")].set_index(["variant", "view", "joint", "labels"])
    macro("zsSapTipTopAll", f1(al.loc[("sapiens2_1b", "top", "fingertip", "all"), "median_mm"]))
    macro("zsSapTipSideAll", f1(al.loc[("sapiens2_1b", "side", "fingertip", "all"), "median_mm"]))
    macro("zsSapTipSideVis", f1(al.loc[("sapiens2_1b", "side", "fingertip", "visible"), "median_mm"]))
    macro("ftTipSideAll", f1(al.loc[("ft_b_full_pooled", "side", "fingertip", "all"), "median_mm"]))
    macro("ftWristSideAll", f1(al.loc[("ft_b_full_pooled", "side", "wrist", "all"), "median_mm"]))
    for view, vk in (("top", "Top"), ("side", "Side")):
        for lab, lk in (("visible", "Vis"), ("all", "All")):
            n = int(al.loc[("sapiens2_1b", view, "fingertip", lab), "n_labels"])
            macro(f"nTipLabels{vk}{lk}", f"{n:,}".replace(",", "{,}"))
    return q1, q2


# ----------------------------------------------------------------------------- E3
def bold_best(rows_vals, better):
    """(row, col) cells to bold: per column the lowest ('min'), highest ('max') or closest to 1 ('one'); NaN never."""
    out = set()
    for c, how in enumerate(better):
        vals = [r[c] for r in rows_vals]
        score = [(-np.inf if np.isnan(v) else (-v if how == "min" else v if how == "max" else -abs(v - 1))) for v in vals]
        if max(score) > -np.inf:
            out.add((int(np.argmax(score)), c))
    return out


def traj(q2):
    s = read("E3_screen.csv")
    q = s[(s.split == "test") & (s.group == "all") & (s.subset == "own")].set_index("source")
    cols = ["rmse_mm", "median_mm", "p90_mm", "path_ratio", "speed_r", "coverage"]
    better = ["min", "min", "min", "one", "max", "max"]

    def key_of(v, with_method):
        if with_method:
            b = best_with(q2, v)
            return (MODEL_LABEL[b] + ", 2 views") if b else None
        return "ViTPose+-B wrist, 2 views" if v == "vitpose_b" else f"{MODEL_LABEL[v]}, 2 views"

    def fmt_row(lab, r, bold_cols):
        vals = [r.rmse_mm, r.median_mm, r.p90_mm, r.path_ratio, r.speed_r, r.coverage]
        txt = [f"{vals[0]:.1f} [{r.rmse_ci_lo:.1f}, {r.rmse_ci_hi:.1f}]", f"{vals[1]:.1f}", f"{vals[2]:.1f}", f"{vals[3]:.2f}", f"{vals[4]:.2f}", f"{vals[5]:.2f}"]
        txt = [f"\\textbf{{{t}}}" if c in bold_cols else t for c, t in enumerate(txt)]
        return f"{lab} & " + " & ".join(txt) + " \\\\"
    v = q.loc["Vicon FIN"]
    rows = [fmt_row("VICON fingertip (markers)", v, set()), "\\addlinespace[1.5pt]"]
    for with_method, head in ((False, "Without \\method{} (zero-shot, two views)"), (True, "With \\method{} (best variant, two views)")):
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\")
        keys = [(vv, key_of(vv, with_method)) for vv in ZS]
        keys = [(vv, k) for vv, k in keys if k and k in q.index]
        # low coverage: never the best
        vals = [[q.loc[k][c] if q.loc[k].coverage >= 0.9 else np.nan for c in cols] for _, k in keys]
        bold = bold_best(vals, better)
        for n, (vv, k) in enumerate(keys):
            lab = LAB[vv] + (" (wrist)$^{\\ddagger}$" if vv == "vitpose_b" and not with_method else "")
            rows.append(fmt_row(lab, q.loc[k], {c for (rr, c) in bold if rr == n}))
        rows.append("\\addlinespace[1.5pt]")
    write_table("res_traj.tex", "Fingertip source & RMSE [95\\% CI] & Median & P90 & Length ratio & Speed $r$ & Cov. \\\\", rows[:-1], "@{}lcccccc@{}")
    for key, k in (("Vicon FIN", "Vicon"), ("Sapiens2-1B, 2 views", "SapTwo"), ("Sapiens2-1B, top", "SapTop"), ("RTMW-X, 2 views", "RtmwTwo"),
                   ("BlazePose+Hands, top", "BlazeTop"), ("ViTPose+-B wrist, 2 views", "VitWristTwo"), ("ViTPose+-B FT full, pooled, top", "FtTop"),
                   ("ViTPose+-B FT full, pooled, 2 views", "FtTwo"), ("ViTPose+-B FT head, pooled, top", "FtHeadTop"),
                   ("RTMPose-m hand (RTMW-X crops), top", "HandTop"), ("RTMPose-WB, top", "RtmposeTop"), ("RTMPose-WB, 2 views", "RtmposeTwo"),
                   ("BlazePose+Hands, 2 views", "BlazeTwo"), ("DWPose-L, 2 views", "DwposeTwo")):
        r = q.loc[key]
        macro(f"traj{k}Rmse", f1(r.rmse_mm))
        macro(f"traj{k}CiLo", f1(r.rmse_ci_lo))
        macro(f"traj{k}CiHi", f1(r.rmse_ci_hi))
        macro(f"traj{k}SpeedR", f1(r.speed_r, 2))
        macro(f"traj{k}Ratio", f1(r.path_ratio, 2))
        macro(f"traj{k}Pnn", f1(r.p90_mm))
    macro("nTestScreenPaths", int(q.loc["Vicon FIN"].n_paths))
    a = s[(s.split == "all") & (s.subset == "own")].set_index(["source", "group"])
    for grp in ("TD", "UCP"):
        macro(f"trajVicon{grp}Rmse", f1(a.loc[("Vicon FIN", grp), "rmse_mm"]))
        macro(f"trajSapTwo{grp}Rmse", f1(a.loc[("Sapiens2-1B, 2 views", grp), "rmse_mm"]))
    return q

# ----------------------------------------------------------------------------- E5
def finetune(q1, q2):
    e3 = read("E3_screen.csv")
    e3 = e3[(e3.split == "test") & (e3.group == "all") & (e3.subset == "own")].set_index("source")
    ang = read("E4_kin_angles.csv")
    ang = ang[(ang.split == "test") & (ang.group == "all") & (ang.angle == "elbow_flexion")].set_index("source")
    lift = pd.concat([read("E5_lift3d_sapiens2_1b.csv").assign(base="sapiens2_1b"), read("E5_lift3d_rtmw_x.csv").assign(base="rtmw_x")])
    lift = lift[lift.group == "all"].set_index(["base", "method", "joint"])

    def row(lab, v, screen_key):
        g2 = q2[q2.variant == v].set_index("joint")
        gt = q1[(q1.variant == v) & (q1.view == "top")].set_index("joint")
        gs = q1[(q1.variant == v) & (q1.view == "side")].set_index("joint")
        sc = e3.rmse_mm.get(screen_key, np.nan) if screen_key else np.nan
        el = ang.rmse_deg.get(v, np.nan)
        return (f"{lab} & {cell(g2.median_mm.get('wrist', np.nan), cov=g2.coverage.get('wrist', np.nan))} & "
                f"{cell(g2.median_mm.get('fingertip', np.nan), cov=g2.coverage.get('fingertip', np.nan))} & "
                f"{cell(gt.median_mm.get('fingertip', np.nan))} & {cell(gs.median_mm.get('fingertip', np.nan))} & {cell(sc)} & {cell(el)} \\\\")
    have = set(q2.variant)
    for v in ZS:
        b = best_with(q2, v)
        if b:
            k = {"vitpose_b": "VitB", "rtmpose_wb": "RtmposeWb", "dwpose_l": "DwposeL", "rtmw_x": "RtmwX",
                 "rtmpose_hand": "Hand", "sapiens2_0.4b": "SapSmall", "sapiens2_1b": "Sap", "blazepose": "Blaze"}[v]
            gb = q2[q2.variant == b].set_index("joint")
            macro(f"with{k}Variant", LAB[b].replace("\\method{}", "the protocol"))
            macro(f"with{k}Desc", with_desc(b))
            macro(f"with{k}TipThreeD", f1(gb.median_mm.get("fingertip", np.nan)))
            macro(f"with{k}WristThreeD", f1(gb.median_mm.get("wrist", np.nan)))
            macro(f"with{k}ElbowRmse", f1(ang.rmse_deg.get(b, np.nan)))
            macro(f"with{k}ScreenRmse", f1(e3.rmse_mm.get(MODEL_LABEL[b] + ", 2 views", np.nan)))
    rows = ["\\multicolumn{7}{@{}l}{\\emph{Without \\method: zero-shot}} \\\\"]
    for v in ("vitpose_b", "rtmw_x", "sapiens2_1b"):
        sk = None if v.startswith("vitpose") else f"{MODEL_LABEL[v]}, 2 views"
        rows.append(row(LAB[v], v, sk))
    rows.append("\\addlinespace[1.5pt]\\multicolumn{7}{@{}l}{\\emph{With \\method: ViTPose+-B fine-tuned on the marker labels}} \\\\")
    for v in ("ft_b_head_top", "ft_b_head_side", "ft_b_head_pooled", "ft_b_full_top", "ft_b_full_side", "ft_b_full_pooled", "ft_b_full_pooled_hasmask"):
        if v in have:
            rows.append(row("\\quad " + LAB[v], v, MODEL_LABEL[v] + ", 2 views"))
    if "ft_b_full_pooled_traj" in have or "ft_b_full_pooled_trajr" in have:
        rows.append("\\addlinespace[1.5pt]\\multicolumn{7}{@{}l}{\\emph{With \\method: marker and trajectory labels "
                    "(\\cref*{main:eq:loss} of the main paper)}} \\\\")
        for v in ("ft_b_full_pooled_traj", "ft_b_full_pooled_trajr"):
            if v in have:
                rows.append(row("\\quad " + LAB[v], v, MODEL_LABEL[v] + ", 2 views"))
    rows.append("\\addlinespace[1.5pt]\\multicolumn{7}{@{}l}{\\emph{With \\method: trajectory-supervised temporal head}} \\\\")
    for v in ("seq_sapiens2_1b", "seq_rtmw_x", "seq_ft_b_full_pooled_traj"):
        if v in have:
            rows.append(row("\\quad " + LAB[v], v, MODEL_LABEL[v] + ", 2 views"))
    rows.append("\\addlinespace[1.5pt]\\multicolumn{7}{@{}l}{\\emph{With \\method: two-view lifting of the triangulated Sapiens2-1B joints}} \\\\")
    for meth, lab in (("mlp", "per-frame network"), ("tcn", "temporal network")):
        w, t = lift.loc[("sapiens2_1b", meth, "wrist")], lift.loc[("sapiens2_1b", meth, "fingertip")]
        rows.append(f"\\quad {lab} & {w.median_mm:.1f} & {t.median_mm:.1f} & -- & -- & -- & -- \\\\")
    rows.append("\\addlinespace[1.5pt]\\multicolumn{7}{@{}l}{\\emph{With \\method: Sapiens2 head fine-tuned on the marker and trajectory labels}} \\\\")
    for v in ("ft_sapiens2_0.4b_head", "ft_sapiens2_1b_head", "seq_ft_sapiens2_0.4b_head", "seq_ft_sapiens2_1b_head"):
        if v in have:
            rows.append(row("\\quad " + LAB[v], v, MODEL_LABEL[v] + ", 2 views"))
    write_table("res_finetune_ablation.tex", "Model & Wrist 3D & Fingertip 3D & Fingertip top & Fingertip side & Screen RMSE & Elbow RMSE ($^\\circ$) \\\\",
                rows, "@{}lcccccc@{}")
    g = q2.set_index(["variant", "joint"])
    for v, k in (("ft_b_full_pooled", "Ft"), ("ft_b_head_pooled", "FtHead"), ("ft_b_full_top", "FtTopOnly"), ("ft_b_full_side", "FtSideOnly"),
                 ("ft_b_head_top", "FtHeadTopOnly"), ("ft_b_full_pooled_hasmask", "FtAll"), ("seq_sapiens2_1b", "SeqSap"), ("seq_rtmw_x", "SeqRtmw"),
                 ("ft_b_full_pooled_n4", "FtFour"), ("ft_b_full_pooled_shift67", "FtShiftA"), ("ft_b_full_pooled_shift200", "FtShiftB"),
                 ("ft_b_full_pooled_traj", "FtTraj"), ("ft_b_full_pooled_trajr", "FtTrajR"), ("seq_ft_b_full_pooled_traj", "SeqFtTraj"),
                 ("ft_sapiens2_0.4b_head", "SapSmallFt"), ("ft_sapiens2_1b_head", "SapFt"), ("seq_ft_sapiens2_0.4b_head", "SeqSapSmallFt"),
                 ("seq_ft_sapiens2_1b_head", "SeqSapFt")):
        if v not in have:
            continue
        for j, jk in (("wrist", "Wrist"), ("fingertip", "Tip"), ("elbow", "Elbow"), ("shoulder", "Shoulder")):
            macro(f"{k[0].lower() + k[1:]}{jk}ThreeD", f1(g.loc[(v, j), "median_mm"]))
        for view, vk in (("top", "Top"), ("side", "Side")):
            gg = q1[(q1.variant == v) & (q1.view == view)].set_index("joint")
            macro(f"{k[0].lower() + k[1:]}Wrist{vk}", f1(gg.median_mm.get("wrist", np.nan)))
            macro(f"{k[0].lower() + k[1:]}Tip{vk}", f1(gg.median_mm.get("fingertip", np.nan)))
    macro("ftTipThreeDPnn", f1(g.loc[("ft_b_full_pooled", "fingertip"), "p90_mm"]))
    macro("ftWristThreeDCiLo", f1(g.loc[("ft_b_full_pooled", "wrist"), "median_ci_lo"]))
    macro("ftWristThreeDCiHi", f1(g.loc[("ft_b_full_pooled", "wrist"), "median_ci_hi"]))
    for base, k in (("sapiens2_1b", "Sap"), ("rtmw_x", "Rtmw")):
        for meth, mk in (("mlp", "Mlp"), ("tcn", "Tcn"), ("triangulation", "Tri")):
            macro(f"lift{k}{mk}Wrist", f1(lift.loc[(base, meth, "wrist"), "median_mm"]))
            macro(f"lift{k}{mk}Tip", f1(lift.loc[(base, meth, "fingertip"), "median_mm"]))
    for v, k in (("ft_b_full_pooled", "Ft"), ("ft_b_full_pooled_traj", "FtTraj"), ("ft_b_full_pooled_trajr", "FtTrajR"), ("seq_ft_b_full_pooled_traj", "SeqFtTraj"),
                 ("ft_sapiens2_0.4b_head", "SapSmallFt"), ("ft_sapiens2_1b_head", "SapFt"), ("seq_sapiens2_1b", "SeqSap"), ("seq_rtmw_x", "SeqRtmw"),
                 ("seq_ft_sapiens2_1b_head", "SeqSapFt")):
        if v in have:
            key = MODEL_LABEL[v] + ", 2 views"
            macro(f"{k[0].lower() + k[1:]}ScreenRmse", f1(e3.rmse_mm.get(key, np.nan)))
            macro(f"{k[0].lower() + k[1:]}ScreenCiLo", f1(e3.rmse_ci_lo.get(key, np.nan)))
            macro(f"{k[0].lower() + k[1:]}ScreenCiHi", f1(e3.rmse_ci_hi.get(key, np.nan)))
            macro(f"{k[0].lower() + k[1:]}ScreenSpeedR", f1(e3.speed_r.get(key, np.nan), 2))
            macro(f"{k[0].lower() + k[1:]}ElbowRmse", f1(ang.rmse_deg.get(v, np.nan)))
    fits = pd.concat([pd.read_csv(f) for f in (BENCH / "labels_traj_fit").glob("*.csv") if f.stat().st_size > 10], ignore_index=True)
    fits = fits[fits.trial >= 0]
    macro("trajPlaneResid", f1(fits.resid_med_mm.median()))
    macro("trajPlaneResidMax", f1(fits.resid_med_mm.max()))
    macro("nTrajPlanes", len(fits))
    al = read("E1_2d.csv")
    al = al[(al.split == "test") & (al.group == "all") & (al.role == "active")].set_index(["variant", "view", "joint", "labels"])
    for v, k in (("ft_b_full_pooled_traj", "FtTraj"), ("ft_b_full_pooled_trajr", "FtTrajR")):
        if v in have:
            for view, vk in (("top", "Top"), ("side", "Side")):
                macro(f"{k[0].lower() + k[1:]}Tip{vk}All", f1(al.loc[(v, view, "fingertip", "all"), "median_mm"]))
    cfg = json.load(open(BENCH / "finetune" / "ft_b_full_pooled" / "config.json"))
    macro("nTrainFrames", f"{cfg['n_train_frames']:,}".replace(",", "{,}"))
    macro("nValFrames", f"{cfg['n_val_frames']:,}".replace(",", "{,}"))
    macro("nTrainPid", len(cfg["train_participants"]))
    macro("nValPid", len(cfg["val_participants"]))
    # Sapiens2 head fine-tunes: frames and validation error
    for run, k in (("ft_sapiens2_0.4b_head", "sapSmallFt"), ("ft_sapiens2_1b_head", "sapFt")):
        if (BENCH / "finetune" / run / "log.csv").exists():
            c = json.load(open(BENCH / "finetune" / run / "config.json"))
            macro(f"{k}Frames", f"{c['n_train_frames']:,}".replace(",", "{,}"))
            macro(f"{k}Epochs", c["epochs"])
            macro(f"{k}ValPx", f1(pd.read_csv(BENCH / "finetune" / run / "log.csv").val_median_px.min()))
    # body models released as ONNX / TFLite graphs (temporal head only)
    graph = [v for v in ("rtmpose_wb", "dwpose_l", "rtmw_x", "blazepose") if "seq_" + v in have]
    pct = lambda a, b: 100 * (b - a) / a                                                            # noqa: E731
    gw = [pct(g.loc[(v, "wrist"), "median_mm"], g.loc[("seq_" + v, "wrist"), "median_mm"]) for v in graph]
    gt = [pct(g.loc[(v, "fingertip"), "median_mm"], g.loc[("seq_" + v, "fingertip"), "median_mm"]) for v in graph]
    ge = [pct(ang.rmse_deg[v], ang.rmse_deg["seq_" + v]) for v in graph]
    macro("seqGraphWristMaxPct", f"{-min(gw):.0f}")
    macro("seqGraphTipMinPct", f"{-max(gt):.0f}")
    macro("seqGraphTipMaxPct", f"{-min(gt):.0f}")
    macro("seqGraphElbowMinPct", f"{-max(ge):.0f}")
    macro("seqGraphElbowMaxPct", f"{-min(ge):.0f}")
    # paired comparison: best zero-shot vs the full pooled fine-tune, per participant
    from scipy.stats import wilcoxon
    p2 = read("E2_3d_participants.csv")
    p2 = p2[(p2.role == "active") & p2.pid.isin(splits()["test"])]
    kp = read("E4_kin_paths.csv")
    kp = kp[kp.pid.isin(splits()["test"])]
    for j, k in (("wrist", "Wrist"), ("fingertip", "Tip")):
        zs = q2[(q2.joint == j) & q2.variant.isin(ZS) & (q2.coverage > 0.9)].sort_values("median_mm").iloc[0].variant
        pr = p2[p2.joint == j].pivot_table(index="pid", columns="variant", values="median_mm")[[zs, "ft_b_full_pooled"]].dropna()
        macro(f"paired{k}Lower", int((pr["ft_b_full_pooled"] < pr[zs]).sum()))
        macro(f"paired{k}N", len(pr))
        macro(f"paired{k}P", f1(wilcoxon(pr[zs], pr["ft_b_full_pooled"]).pvalue, 3))
        macro(f"paired{k}Model", LAB[zs])
        # fine-tuned Sapiens2-1B vs itself zero-shot and vs ViTPose+-B
        trio = ["sapiens2_1b", "ft_sapiens2_1b_head", "ft_b_full_pooled_traj"]
        if set(trio) <= have:
            pr = p2[p2.joint == j].pivot_table(index="pid", columns="variant", values="median_mm")[trio].dropna()
            macro(f"pairedSap{k}Lower", int((pr[trio[1]] < pr[trio[0]]).sum()))
            macro(f"pairedSap{k}N", len(pr))
            macro(f"pairedSap{k}P", f1(wilcoxon(pr[trio[0]], pr[trio[1]]).pvalue, 3))
            macro(f"pairedSapVit{k}Lower", int((pr[trio[1]] < pr[trio[2]]).sum()))
            macro(f"pairedSapVit{k}P", f1(wilcoxon(pr[trio[2]], pr[trio[1]]).pvalue, 3))
            macro(f"pairedVitSap{k}Lower", int((pr[trio[2]] < pr[trio[1]]).sum()))
    for a, k in (("elbow_flexion", "Elbow"), ("shoulder_elevation", "Shoulder")):
        angs = read("E4_kin_angles.csv")
        angs = angs[(angs.split == "test") & (angs.group == "all") & (angs.angle == a)]
        zs = angs[angs.source.isin(ZS)].sort_values("rmse_deg").iloc[0].source
        pr = kp.groupby(["pid", "source"])[f"{a}_rmse_deg"].median().unstack()[[zs, "ft_b_full_pooled"]].dropna()
        macro(f"paired{k}Lower", int((pr["ft_b_full_pooled"] < pr[zs]).sum()))
        macro(f"paired{k}P", f1(wilcoxon(pr[zs], pr["ft_b_full_pooled"]).pvalue, 3))
        macro(f"paired{k}Model", LAB[zs])


# ----------------------------------------------------------------------------- E4
def kinematics(q2):
    a, g, e = read("E4_kin_angles.csv"), read("E4_kin_agreement.csv"), read("E4_kin_effects.csv")
    qa = a[(a.split == "test") & (a.group == "all")].set_index(["source", "angle"])
    qg = g[g.split == "test"].set_index(["source", "feature"])
    feats = [("wrist_straightness", "straight."), ("wrist_time_to_peak", "time to peak"), ("wrist_sparc", "SPARC"),
             ("fingertip_peak_speed", "peak speed"), ("fingertip_straightness", "straight."), ("fingertip_time_to_peak", "time to peak"),
             ("fingertip_sparc", "SPARC")]
    rows = []
    for with_method, head in ((False, "Without \\method{} (zero-shot)"), (True, "With \\method{} (best variant)")):
        rows.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\")
        srcs = [(v, best_with(q2, v) if with_method else v) for v in ZS]
        srcs = [(v, s) for v, s in srcs if s]
        vals = []
        for v, s in srcs:
            el = qa.loc[(s, "elbow_flexion")] if (s, "elbow_flexion") in qa.index and np.isfinite(qa.loc[(s, "elbow_flexion"), "rmse_deg"]) else None
            sh = qa.loc[(s, "shoulder_elevation")] if (s, "shoulder_elevation") in qa.index and np.isfinite(qa.loc[(s, "shoulder_elevation"), "rmse_deg"]) else None
            vals.append([el.rmse_deg if el is not None else np.nan, el.abs_rom_err_deg if el is not None else np.nan,
                         sh.rmse_deg if sh is not None else np.nan, sh.abs_rom_err_deg if sh is not None else np.nan]
                        + [qg.icc21.get((s, f), np.nan) for f, _ in feats])
        bold = bold_best(vals, ["min"] * 4 + ["max"] * len(feats))
        for n, (v, s) in enumerate(srcs):
            el = qa.loc[(s, "elbow_flexion")] if (s, "elbow_flexion") in qa.index and np.isfinite(qa.loc[(s, "elbow_flexion"), "rmse_deg"]) else None
            sh = qa.loc[(s, "shoulder_elevation")] if (s, "shoulder_elevation") in qa.index and np.isfinite(qa.loc[(s, "shoulder_elevation"), "rmse_deg"]) else None
            cells = [f"{el.rmse_deg:.1f} [{el.rmse_ci_lo:.1f}, {el.rmse_ci_hi:.1f}]" if el is not None else "--", f"{el.abs_rom_err_deg:.1f}" if el is not None else "--",
                     f"{sh.rmse_deg:.1f} [{sh.rmse_ci_lo:.1f}, {sh.rmse_ci_hi:.1f}]" if sh is not None else "--", f"{sh.abs_rom_err_deg:.1f}" if sh is not None else "--"]
            cells += [cell(qg.icc21.get((s, f), np.nan), nd=2) for f, _ in feats]
            cells = [f"\\textbf{{{c}}}" if (n, k) in bold and c != "--" else c for k, c in enumerate(cells)]
            rows.append(f"{LAB[v]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace[1.5pt]")
    rows = rows[:-1]
    head = ("& \\multicolumn{2}{c}{Elbow flexion} & \\multicolumn{2}{c}{Shoulder elevation} & \\multicolumn{3}{c}{ICC, wrist} & "
            "\\multicolumn{4}{c}{ICC, fingertip} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-12}\n"
            "Source & RMSE ($^\\circ$) & $|$ROM err.$|$ & RMSE ($^\\circ$) & $|$ROM err.$|$ & " + " & ".join(l for _, l in feats) + " \\\\")
    write_table("res_kin.tex", head, rows, "@{}lcccccccccccc@{}")
    for s, k in (("ft_b_full_pooled", "Ft"), ("ft_b_head_pooled", "FtHead"), ("sapiens2_1b", "Sap"), ("rtmw_x", "Rtmw"), ("seq_sapiens2_1b", "SeqSap"),
                 ("vitpose_b", "Vit"), ("blazepose", "Blaze"), ("sapiens2_0.4b", "SapSmall"), ("ft_b_full_pooled_traj", "FtTraj"), ("seq_rtmw_x", "SeqRtmw"),
                 ("ft_sapiens2_1b_head", "SapFt"), ("ft_sapiens2_0.4b_head", "SapSmallFt")):
        if (s, "elbow_flexion") not in qa.index:
            continue
        for ang, ak in (("elbow_flexion", "Elbow"), ("shoulder_elevation", "Shoulder")):
            macro(f"kin{k}{ak}Rmse", f1(qa.loc[(s, ang), "rmse_deg"]))
            macro(f"kin{k}{ak}Rom", f1(qa.loc[(s, ang), "abs_rom_err_deg"]))
    for f, fk in (("wrist_straightness", "WrStr"), ("wrist_time_to_peak", "WrTtp"), ("wrist_sparc", "WrSparc"), ("wrist_ldlj", "WrLdlj"),
                  ("wrist_n_peaks", "WrNpk"), ("wrist_peak_speed", "WrPks"), ("fingertip_peak_speed", "TipPks"), ("fingertip_straightness", "TipStr"),
                  ("fingertip_time_to_peak", "TipTtp"), ("fingertip_sparc", "TipSparc"), ("fingertip_ldlj", "TipLdlj"), ("fingertip_n_peaks", "TipNpk")):
        # best zero-shot ICC among sources scored on > 50 trials (RTMPose-m hand has a fingertip in few trials)
        z = g[(g.split == "test") & g.source.isin(ZS) & (g.feature == f) & (g.n_paths > 50)].sort_values("icc21", ascending=False).iloc[0]
        macro(f"icc{fk}Zs", f1(z.icc21, 2))
        macro(f"icc{fk}ZsModel", LAB[z.source].replace("$^\\dagger$", ""))
        macro(f"icc{fk}Ft", f1(qg.icc21.get(("ft_b_full_pooled", f), np.nan), 2))
        macro(f"icc{fk}FtTraj", f1(qg.icc21.get(("ft_b_full_pooled_traj", f), np.nan), 2))
        macro(f"icc{fk}Sap", f1(qg.icc21.get(("sapiens2_1b", f), np.nan), 2))
        macro(f"icc{fk}SapFt", f1(qg.icc21.get(("ft_sapiens2_1b_head", f), np.nan), 2))
        macro(f"icc{fk}SapSmallFt", f1(qg.icc21.get(("ft_sapiens2_0.4b_head", f), np.nan), 2))
    for s, k in (("sapiens2_1b", "Sap"), ("ft_sapiens2_1b_head", "SapFt"), ("ft_sapiens2_0.4b_head", "SapSmallFt"), ("ft_b_full_pooled_traj", "FtTraj")):
        if (s, "fingertip_peak_speed") in qg.index:      # jitter: bias of the fingertip's peak speed (mm/s)
            macro(f"biasTipPks{k}", f"{qg.bias[(s, 'fingertip_peak_speed')]:.0f}")
            macro("tipPksVicon", f"{qg.vicon_mean[(s, 'fingertip_peak_speed')]:.0f}")
    # best zero-shot angle RMSE
    for ang, ak in (("elbow_flexion", "Elbow"), ("shoulder_elevation", "Shoulder")):
        z = a[(a.split == "test") & (a.group == "all") & (a.angle == ang) & a.source.isin(ZS)].sort_values("rmse_deg").iloc[0]
        macro(f"kinZs{ak}Rmse", f1(z.rmse_deg))
        macro(f"kinZs{ak}Model", LAB[z.source])
    # effects on all participants (zero-shot) -> supplementary table + macros
    ea = e[(e.split == "all")].set_index(["source", "feature"])
    fe = [("elbow_flexion_rom_deg", "elbow ROM"), ("shoulder_elevation_rom_deg", "shoulder ROM"),
          ("wrist_peak_speed", "wrist peak speed"), ("wrist_time_to_peak", "wrist time to peak"), ("wrist_straightness", "wrist straightness"),
          ("wrist_n_peaks", "wrist speed peaks"), ("wrist_sparc", "wrist SPARC"), ("wrist_ldlj", "wrist LDLJ"),
          ("fingertip_peak_speed", "fingertip peak speed"), ("fingertip_time_to_peak", "fingertip time to peak"),
          ("fingertip_straightness", "fingertip straightness"), ("fingertip_n_peaks", "fingertip speed peaks"), ("fingertip_sparc", "fingertip SPARC"),
          ("fingertip_ldlj", "fingertip LDLJ")]
    es = ["vicon", "vitpose_b", "rtmw_x", "sapiens2_1b", "blazepose"]
    rows = []
    for f, lab in fe:
        c1 = [cell(ea.d_ucp_vs_td.get((s, f), np.nan), nd=2) for s in es]
        c2 = [cell(ea.dz_affected_vs_less_UCP.get((s, f), np.nan), nd=2) for s in es]
        rows.append(f"{lab} & " + " & ".join(c1) + " & " + " & ".join(c2) + " \\\\")
    n = ea.loc[("vicon", "wrist_peak_speed")]
    head = (f"& \\multicolumn{{5}}{{c}}{{$d$, UCP vs.\\ TD ({int(n.n_ucp)} vs.\\ {int(n.n_td)})}} & "
            f"\\multicolumn{{5}}{{c}}{{$d_z$, affected vs.\\ less-affected arm, UCP ({int(n.n_pairs_UCP)} pairs)}} \\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\nFeature & VICON & " + " & ".join(FIGLAB[s] for s in es[1:]) + " & VICON & "
            + " & ".join(FIGLAB[s] for s in es[1:]) + " \\\\")
    write_table("res_effects.tex", head, rows, "@{}lcccccccccc@{}")
    for f, fk in (("elbow_flexion_rom_deg", "ElbowRom"), ("wrist_straightness", "WrStr"), ("fingertip_time_to_peak", "TipTtp"), ("wrist_sparc", "WrSparc"),
                  ("fingertip_sparc", "TipSparc"), ("fingertip_straightness", "TipStr")):
        for s, sk in (("vicon", "Vicon"), ("sapiens2_1b", "Sap"), ("rtmw_x", "Rtmw"), ("blazepose", "Blaze"), ("vitpose_b", "Vit")):
            macro(f"d{fk}{sk}", f1(ea.d_ucp_vs_td.get((s, f), np.nan), 2))
            macro(f"dz{fk}{sk}", f1(ea.dz_affected_vs_less_UCP.get((s, f), np.nan), 2))
    macro("nEffUCP", int(n.n_ucp))
    macro("nEffTD", int(n.n_td))
    # correlation of the per-feature d (video vs VICON) over the 14 features, per source
    for s, sk in (("sapiens2_1b", "Sap"), ("rtmw_x", "Rtmw"), ("vitpose_b", "Vit"), ("blazepose", "Blaze")):
        d = pd.DataFrame({"v": [ea.d_ucp_vs_td.get(("vicon", f), np.nan) for f, _ in fe], "m": [ea.d_ucp_vs_td.get((s, f), np.nan) for f, _ in fe]}).dropna()
        macro(f"dCorr{sk}", f1(d.corr().iloc[0, 1], 2))
        macro(f"dMad{sk}", f1((d.m - d.v).abs().mean(), 2))
    return a, g


# ----------------------------------------------------------------------------- E6
def analyses():
    a, b, c, d, e, f = (read(x) for x in ("E6a_group_limb.csv", "E6b_scaling.csv", "E6c_alignment_shift.csv", "E6d_views.csv",
                                           "E6e_filtering.csv", "E6f_selection_bias.csv"))
    # (a) group x limb, all participants, 3D
    qa = a[(a.split == "all") & (a.role == "active") & (a.view == "3d")].set_index(["variant", "joint", "group", "limb"])
    names = {("TD", "less_affected"): "TdDom", ("TD", "affected"): "TdNon", ("UCP", "less_affected"): "UcpLess", ("UCP", "affected"): "UcpAff"}
    for v, vk in (("sapiens2_1b", "Sap"), ("rtmw_x", "Rtmw"), ("vitpose_b", "Vit")):
        for j, jk in (("wrist", "Wrist"), ("fingertip", "Tip")):
            for (grp, limb), nk in names.items():
                if (v, j, grp, limb) in qa.index:
                    macro(f"limb{vk}{jk}{nk}", f1(qa.loc[(v, j, grp, limb), "value"]))
    macro("nLimbTD", int(qa.loc[("sapiens2_1b", "wrist", "TD", "affected"), "n_participants"]))
    macro("nLimbUCP", int(qa.loc[("sapiens2_1b", "wrist", "UCP", "affected"), "n_participants"]))
    # (b) scaling
    qb = b[(b.split == "test") & (b.view.notna())].set_index(["n_train", "view", "joint"])
    for n, nk in ((4, "Four"), (8, "Eight"), (12, "Twelve"), (17, "All")):
        for j, jk in (("wrist", "Wrist"), ("fingertip", "Tip")):
            macro(f"scale{nk}{jk}ThreeD", f1(qb.loc[(n, "3d", j), "median_mm"]))
            macro(f"scale{nk}{jk}Top", f1(qb.loc[(n, "top", j), "median_mm"]))
    # (c) shift
    qc = c[(c.split == "test") & (c.view == "3d") & (c.joint == "wrist")].set_index(["kind", "variant", "shift_ms"])
    for v, vk in (("sapiens2_1b", "Sap"), ("vitpose_b", "Vit"), ("blazepose", "Blaze"), ("rtmw_x", "Rtmw")):
        for ms, mk in ((0, "Zero"), (33, "A"), (67, "B"), (100, "C"), (200, "D")):
            macro(f"shift{vk}{mk}", f1(qc.loc[("test-time reference shift", v, ms), "median_mm"]))
    macro("shiftFtZero", f1(qc.loc[("training-label shift", "ft_b_full_pooled", 0), "median_mm"]))
    macro("shiftFtB", f1(qc.loc[("training-label shift", "ft_b_full_pooled_shift67", 67), "median_mm"]))
    macro("shiftFtD", f1(qc.loc[("training-label shift", "ft_b_full_pooled_shift200", 200), "median_mm"]))
    # (d) views -> supplementary table
    qd = d[(d.split == "test")].set_index(["variant", "joint", "view"])
    rows = []
    for v in ("vitpose_b", "rtmw_x", "sapiens2_1b", "seq_sapiens2_1b", "ft_b_full_pooled"):
        for j in ("wrist", "fingertip"):
            if (v, j, "3d") not in qd.index:
                continue
            cells = [cell(qd.median_mm.get((v, j, view), np.nan)) for view in ("top", "side", "3d")]
            rows.append(f"{FIGLAB[v]} & {JLAB[j]} & " + " & ".join(cells) + " \\\\")
    write_table("res_views.tex", "Model & Joint & Top 2D & Side 2D & Two-view 3D \\\\", rows, "@{}llccc@{}")
    for v, vk in (("sapiens2_1b", "Sap"), ("ft_b_full_pooled", "Ft"), ("rtmw_x", "Rtmw")):
        for j, jk in (("wrist", "Wrist"), ("fingertip", "Tip")):
            for view, wk in (("top", "Top"), ("side", "Side"), ("3d", "ThreeD")):
                macro(f"view{vk}{jk}{wk}", f1(qd.median_mm.get((v, j, view), np.nan)))
    # (e) filtering
    qe = e[(e.split == "test") & (e.view == "3d")].pivot_table(index=["variant", "joint"], columns="filtering", values="median_mm")
    diff = (qe["raw"] - qe["6 Hz filtered"])
    macro("filterMaxDiffNoBlaze", f1(diff.drop("blazepose", level=0).abs().max()))
    macro("filterBlazeElbowRaw", f1(qe.loc[("blazepose", "elbow"), "raw"]))
    macro("filterBlazeElbowFilt", f1(qe.loc[("blazepose", "elbow"), "6 Hz filtered"]))
    # (f) selection bias -> supplementary table
    rows = []
    fl = {"peak_speed": "peak speed (mm/s)", "time_to_peak": "time to peak (s)", "straightness": "straightness", "n_peaks": "speed peaks",
          "sparc": "SPARC", "ldlj": "LDLJ", "move_duration_s": "trial duration (s)"}
    for grp in ("TD", "UCP"):
        g = f[f.group == grp].set_index("feature")
        for k, lab in fl.items():
            r = g.loc[k]
            nd = {"peak_speed": 0, "time_to_peak": 1, "n_peaks": 1, "move_duration_s": 1}.get(k, 2)
            rows.append(f"{grp if k == 'peak_speed' else ''} & {lab} & {r.mean_pass:.{nd}f} & {r.mean_fail:.{nd}f} & {r.d_pass_vs_fail:+.2f} \\\\")
        rows.append("\\addlinespace[1.5pt]")
        macro(f"selPass{grp}", f"{100 * g.loc['peak_speed'].share_pass:.0f}")
        macro(f"selNPass{grp}", int(g.loc["peak_speed"].n_pass))
        macro(f"selNFail{grp}", int(g.loc["peak_speed"].n_fail))
        for k, kk in (("peak_speed", "Pks"), ("move_duration_s", "Dur"), ("time_to_peak", "Ttp"), ("n_peaks", "Npk"), ("straightness", "Str")):
            macro(f"selD{kk}{grp}", f"{g.loc[k].d_pass_vs_fail:+.2f}")
    write_table("res_selection.tex", "Group & VICON feature & Mean, passing & Mean, failing & $d$ \\\\", rows[:-1], "@{}llccc@{}")
    return a, b, c, d


# ----------------------------------------------------------------------------- splits, calibration
def splits_table():
    sp = splits()
    m = manifest()
    grp = m.groupby("pid").group.first()
    sub = public_ids()
    rows = [f"Test & {', '.join(sub[p] + ' (' + grp[p] + ')' for p in sp['test'])} \\\\"]
    for i, f in enumerate(sp["folds"]):
        rows.append(f"Fold {i + 1} & {', '.join(sub[p] + ' (' + grp[p] + ')' for p in f)} \\\\")
    write_table("splits.tex", "Split & Participants (group) \\\\", rows, "@{}lp{0.78\\linewidth}@{}")
    macro("nTestTD", sum(grp[p] == "TD" for p in sp["test"]))
    macro("nTestUCP", sum(grp[p] == "UCP" for p in sp["test"]))
    macro("nDevPid", sum(len(f) for f in sp["folds"]))
    cal = pd.read_csv(BENCH / "cameras" / "calibration.csv")
    macro("nCalibVideos", len(cal))
    macro("nCalibTop", int((cal.view == "top").sum()))
    macro("nCalibSide", int((cal.view == "side").sum()))
    macro("calibMedPx", f1(cal.reproj_test_median_px.median()))
    macro("calibMinPx", f1(cal.reproj_test_median_px.min()))
    macro("calibMaxPx", f1(cal.reproj_test_median_px.max()))
    macro("calibPnnMedPx", f1(cal.reproj_test_p90_px.median()))
    macro("calibPairsMed", f"{int(cal.n_test_pairs.median()):,}".replace(",", "{,}"))
    macro("calibOk", int(cal.calib_ok.sum()))
    macro("pxPerMmTop", f1(cal[cal.view == "top"].px_per_mm_median.median()))
    macro("pxPerMmSide", f1(cal[cal.view == "side"].px_per_mm_median.median()))
    # participants without a valid video or without valid VICON data
    per = m.groupby("pid").agg(top=("top_valid", "sum"), side=("side_valid", "sum"), vicon=("vicon_valid", "sum"), group=("group", "first"))
    macro("nPidNoVideo", int(((per.top == 0) & (per.side == 0)).sum()))
    macro("nPidNoVicon", int((per.vicon == 0).sum()))
    macro("pidNoVicon", ", ".join(sub[p] for p in per[per.vicon == 0].index))
    # label frames
    n = 0
    for f in (BENCH / "labels").glob("*/*/*.npz"):
        n += len(np.load(f)["frame"])
    macro("nLabelFramesK", round(n / 1000))


# ----------------------------------------------------------------------------- figures
def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK, "axes.titlecolor": INK, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                         "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=250)
    plt.close(fig)


def wrist_as_fingertip(model="vitpose_b"):
    """3D error of a model's wrist keypoint against the VICON fingertip on the test trials (mean over participants of
    the per-participant median, bootstrap CI), for a model without a fingertip keypoint."""
    from evaluate import boot_ci
    m = manifest()
    rows = m[m.vicon_valid & m.top_valid & m.side_valid & ~m.trajectory_qa_failed & m.pid.isin(splits()["test"])]
    per = {}
    for _, r in rows.iterrows():
        f = BENCH / "tracks3d" / f"{r.pid}_r{int(r.aligned_mat_row)}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if model not in z.files:
            continue
        e = np.linalg.norm(z[model][2] - z["vicon"][3], axis=1)        # J4 = shoulder, elbow, wrist, fingertip
        per.setdefault(r.pid, []).append(e[~np.isnan(e)])
    med = pd.Series({p: np.median(np.concatenate(v)) for p, v in per.items() if len(v)})
    lo, hi = boot_ci(med)
    return float(med.mean()), float(lo), float(hi), len(med)


def fig_zs_ft(q1, q2, ang):
    e3 = read("E3_screen.csv")
    e3 = e3[(e3.split == "test") & (e3.group == "all") & (e3.subset == "own")].set_index("source")
    qa = ang[(ang.split == "test") & (ang.group == "all")]
    pairs = [(v, best_with(q2, v)) for v in ZS]
    scr = {v: (None if v.startswith("vitpose") else f"{MODEL_LABEL[v]}, 2 views") for v in ZS}
    scr.update({w: MODEL_LABEL[w] + ", 2 views" for _, w in pairs if w})
    panels = [("Fingertip 3-D (mm)", q2[q2.joint == "fingertip"].set_index("variant")[["median_mm", "median_ci_lo", "median_ci_hi"]]),
              ("Wrist 3-D (mm)", q2[q2.joint == "wrist"].set_index("variant")[["median_mm", "median_ci_lo", "median_ci_hi"]]),
              ("Elbow RMSE (\u00b0)", qa[qa.angle == "elbow_flexion"].set_index("source")[["rmse_deg", "rmse_ci_lo", "rmse_ci_hi"]]),
              ("Shoulder RMSE (\u00b0)", qa[qa.angle == "shoulder_elevation"].set_index("source")[["rmse_deg", "rmse_ci_lo", "rmse_ci_hi"]]),
              ("Screen RMSE (mm)", e3.rename(index={v: k for k, v in scr.items() if v})[["rmse_mm", "rmse_ci_lo", "rmse_ci_hi"]])]
    shown = [n for pr in pairs for n in pr if n]
    wf = wrist_as_fingertip("vitpose_b")
    panels[0][1].loc["vitpose_b"] = wf[:3]                                       # wrist as the fingertip, marked
    panels[4][1].loc["vitpose_b"] = e3.loc["ViTPose+-B wrist, 2 views"][["rmse_mm", "rmse_ci_lo", "rmse_ci_hi"]].values
    macro("zsVitWristAsTipThreeD", f1(wf[0]))
    macro("zsVitWristAsTipCiLo", f1(wf[1]))
    macro("zsVitWristAsTipCiHi", f1(wf[2]))
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 3.6), sharey=True)
    y_zs = {v: 1.0 * i for i, v in enumerate(ZS)}
    vicon_screen = e3.loc["Vicon FIN"]
    for ax, (title, w) in zip(axes, panels):
        w = w.copy()
        w.columns = ["v", "lo", "hi"]
        if title.startswith("Screen"):
            ax.axvline(vicon_screen.rmse_mm, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
            ax.text(vicon_screen.rmse_mm, -0.62, f" Vicon {vicon_screen.rmse_mm:.1f}", fontsize=5.5, color=INK2, va="bottom", ha="left")
        for v, b in pairs:
            for k, (name, dy, alpha, hatch) in enumerate(((v, -0.19, 1.0, None), (b, 0.19, 0.55, "////"))):
                if name is None or name not in w.index or not np.isfinite(w.v.get(name, np.nan)):
                    if k == 0 and name in w.index or (k == 0 and name not in w.index):
                        ax.text(0, y_zs[v] + dy, " n/a", va="center", fontsize=5, color=INK3)
                    continue
                r = w.loc[name]
                ax.barh(y_zs[v] + dy, r.v, height=0.36, color=MCOL[v], alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.5)
                ax.errorbar(r.v, y_zs[v] + dy, xerr=[[r.v - r.lo], [r.hi - r.v]], fmt="none", ecolor=INK2, elinewidth=0.6)
                lab = f" {r.v:.1f}"
                if k == 0 and v == "vitpose_b" and title.split()[0] in ("Fingertip", "Screen"):
                    lab += "\u2021"                                                     # wrist used as the fingertip
                if k == 1 and v in w.index and np.isfinite(w.v.get(v, np.nan)):
                    lab += f" ({(r.v / w.v[v] - 1) * 100:+.0f}%)"
                ax.text(r.hi, y_zs[v] + dy, lab, va="center", fontsize=5.2, color=INK)
        keep = [n for n in shown if n in w.index and not (n == "vitpose_b" and title.startswith("Fingertip"))]
        xmax = np.nanmax(w.hi.reindex(keep).values) * 1.38
        ax.set_xlim(0, xmax)
        # the wrist-as-fingertip bar runs off the axis
        if title.startswith("Fingertip") and "vitpose_b" in w.index:
            for t in ax.texts:
                if t.get_text().endswith("\u2021") and t.get_position()[1] == y_zs["vitpose_b"] - 0.19:
                    t.set_x(xmax * 0.98)
                    t.set_ha("right")
                    t.set_bbox(dict(facecolor="white", lw=0, pad=0.5))
        ax.set_title(title, fontsize=6.5)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(len(ZS) - 0.5, -0.75)
    axes[0].set_yticks([y_zs[v] for v in ZS])
    axes[0].set_yticklabels([FIGLAB[v] for v in ZS])
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=INK3, label="Zero-shot"),
          plt.Rectangle((0, 0), 1, 1, facecolor=INK3, alpha=0.55, hatch="////", edgecolor="white", label="With ViconTraj-MS")]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.03, 1, 1))
    save(fig, "results_zs_ft")


def fig_analyses(a, b, c, d):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), gridspec_kw=dict(width_ratios=(1.15, 0.9, 1.05, 1.1)))
    # (a) group x limb, 3D wrist, all participants
    ax = axes[0]
    qa = a[(a.split == "all") & (a.role == "active") & (a.view == "3d") & (a.joint == "wrist")]
    cats = [("TD", "less_affected"), ("TD", "affected"), ("UCP", "less_affected"), ("UCP", "affected")]
    names = ["TD\ndom.", "TD\nnon-d.", "UCP\nless", "UCP\naff."]
    for k, v in enumerate(["vitpose_b", "rtmw_x", "sapiens2_1b"]):
        w = qa[qa.variant == v].set_index(["group", "limb"]).reindex(cats)
        ax.errorbar(np.arange(4) + (k - 1) * 0.2, w.value, yerr=[w.value - w.ci_lo, w.ci_hi - w.value], fmt=MARK[v], ms=3.6, color=family_color(v),
                    mec="white", mew=0.4, label=FIGLAB[v], capsize=0, elinewidth=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=5.4)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("3-D wrist error (mm)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", handletextpad=0.3)
    ax.set_title("a  Group and arm (zero-shot)", loc="left", fontweight="bold")
    # (b) training size
    ax = axes[1]
    qb = b[(b.split == "test") & (b.view == "3d")]
    for k, j in enumerate(("wrist", "hand", "fingertip")):
        w = qb[qb.joint == j].sort_values("n_train")
        ax.errorbar(w.n_train, w.median_mm, yerr=[w.median_mm - w.median_ci_lo, w.median_ci_hi - w.median_mm], marker="o", ms=3.2, lw=1.4,
                    color=SLOT[k], label=JLAB[j], capsize=0, elinewidth=0.7)
    ax.set_xticks([4, 8, 12, 17])
    ax.set_xlabel("training participants")
    ax.set_ylabel("3-D error (mm), fine-tuned")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    ax.set_title("b  Training size", loc="left", fontweight="bold")
    # (c) alignment shift
    ax = axes[2]
    qc = c[(c.split == "test") & (c.view == "3d") & (c.joint == "wrist")]
    for v in ("blazepose", "rtmpose_wb", "rtmw_x", "sapiens2_1b", "vitpose_b"):
        g = qc[(qc.kind == "test-time reference shift") & (qc.variant == v)].sort_values("shift_ms")
        ax.plot(g.shift_ms, g.median_mm, marker=MARK[v], ms=3.2, lw=1.2, color=family_color(v), mec="white", mew=0.4, label=FIGLAB[v],
                ls=(0, (3, 1.5)) if v == "rtmw_x" else "-")
    tw = qc[qc.kind == "training-label shift"].sort_values("shift_ms")
    ax.plot(tw.shift_ms, tw.median_mm, marker="s", ms=3.4, lw=1.2, ls="--", color=SLOT[4], mec="white", mew=0.4, label="FT, labels shifted\nin training")
    ax.set_xticks([0, 33, 67, 100, 200])
    ax.set_xlabel("timing shift (ms)")
    ax.set_ylabel("3-D wrist error (mm)")
    ax.set_ylim(0, 38)
    ax.legend(loc="upper left", ncol=2, columnspacing=0.6, handletextpad=0.3, fontsize=5.0, labelspacing=0.25)
    ax.set_title("c  Alignment shift", loc="left", fontweight="bold")
    # (d) views
    ax = axes[3]
    qd = d[d.split == "test"].set_index(["variant", "joint", "view"])
    models = ["rtmw_x", "sapiens2_1b", "ft_b_full_pooled"]
    x = np.arange(3) * 3.6
    for k, (view, lab) in enumerate((("top", "top 2-D"), ("side", "side 2-D"), ("3d", "two-view 3-D"))):
        for jj, j in enumerate(("wrist", "fingertip")):
            vals = [qd.median_mm.get((m, j, view), np.nan) for m in models]
            ax.bar(x + jj * 1.0 + (k - 1) * 0.28, vals, width=0.26, color=[family_color(m) for m in models], alpha=(1.0, 0.7, 0.45)[k],
                   edgecolor="white", linewidth=0.4, label=lab if jj == 0 else None)
    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(["RTMW-X", "Sapiens2-\n1B", "ViTPose+-B\nFT full"], fontsize=5.4)
    ax.set_xlim(-0.7, x[-1] + 1.7)
    for i, m in enumerate(models):
        for jj, j in enumerate(("wrist", "fingertip")):
            top = max(qd.median_mm.get((m, j, view), 0) for view in ("top", "side", "3d"))
            ax.text(x[i] + jj * 1.0, top + 1.2, "W" if j == "wrist" else "F", ha="center", va="bottom", fontsize=5.4, color=INK2)
    ax.set_ylabel("median error (mm)")
    ax.set_ylim(0, 44)
    ax.legend(loc="upper right", handlelength=1.0, labelspacing=0.25)
    ax.set_title("d  Views", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=0.5)
    save(fig, "analyses")


def fig_icc(g):
    srcs = [s for s in ("vitpose_b", "rtmpose_wb", "dwpose_l", "rtmw_x", "sapiens2_0.4b", "sapiens2_1b", "blazepose", "seq_sapiens2_1b", "ft_b_head_pooled",
                        "ft_b_full_pooled", "ft_b_full_pooled_traj", "seq_ft_b_full_pooled_traj", "ft_sapiens2_0.4b_head", "ft_sapiens2_1b_head")
            if s in set(g.source)]
    feats = [f"{j}_{f}" for j in ("wrist", "fingertip") for f in ("peak_speed", "time_to_peak", "straightness", "n_peaks", "sparc", "ldlj")]
    flab = [f"{'W' if f.startswith('wrist') else 'F'}: {f.split('_', 1)[1].replace('_', ' ').replace('n peaks', 'speed peaks').replace('sparc', 'SPARC').replace('ldlj', 'LDLJ')}" for f in feats]
    q = g[g.split == "test"].pivot_table(index="source", columns="feature", values="icc21").reindex(index=srcs, columns=feats)
    fig, ax = plt.subplots(figsize=(7.0, 0.6 + 0.2 * len(srcs)))
    im = ax.imshow(q.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(srcs)):
        for j in range(len(feats)):
            v = q.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(flab, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(srcs)))
    ax.set_yticklabels([FIGLAB[s] for s in srcs], fontsize=6.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.set_label("ICC(2,1) vs Vicon", fontsize=6.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "kin_icc")


def copy_e7():
    """fig/qualitative.pdf and .png: the qualitative figure of benchmark/e7_qualitative.py."""
    rep = BENCH / "reports" / "E7"
    for ext in ("pdf", "png"):
        if (rep / f"qualitative.{ext}").exists():
            shutil.copy(rep / f"qualitative.{ext}", FIG / f"qualitative.{ext}")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    setup_style()
    q1, q2 = zero_shot()
    traj(q2)
    finetune(q1, q2)
    ang, agr = kinematics(q2)
    a, b, c, d = analyses()
    splits_table()
    fig_zs_ft(q1, q2, ang)
    fig_analyses(a, b, c, d)
    fig_icc(agr)
    copy_e7()
    with open(TAB / "res_macros.tex", "w") as f:
        f.write("% generated by analysis/make_results.py -- do not edit\n")
        for k, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(MACROS)} macros, tables and figures in", TAB, FIG)


if __name__ == "__main__":
    main()
