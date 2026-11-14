"""E6 analyses: error by group and arm, training size, alignment shift, views, filtering and selection bias.

Reads the E1-E3 results of outputs/benchmark/results/, including the evaluate.py runs with --shift-ms and
--no-filter and the fine-tuned runs of finetune.py --n-train / --label-shift-ms, and writes there:
    E6a_group_limb.csv        error by group (TD, UCP) and arm (affected = paretic / non-dominant; active, resting)
    E6b_scaling.csv           ViTPose+-B full fine-tune, both views, on 4, 8, 12 and all 17 training participants
    E6c_alignment_shift.csv   reference shifted at test time by 0-200 ms, and training on labels shifted by 67 / 200 ms
    E6d_views.csv             top 2D, side 2D and two-view 3D
    E6e_filtering.csv         zero-shot models: raw keypoints against raw markers, and the 6 Hz filtered default
    E6f_selection_*.csv       VICON fingertip reach features of the VICON-valid trials that pass both video tests
                              against those that fail them, per group
E6a-e follow the protocol of evaluate.py: median per participant, mean over participants with a 1000-sample
bootstrap 95 % CI over participants. E6f compares trials (Cohen's d). A table whose inputs are missing is skipped.

    python benchmark/e6_analyses.py
"""
import argparse

import numpy as np
import pandas as pd

from benchlib import MODEL_LABEL, limb_of, manifest, markers, splits
from e4_kinematics import FEATURES, FS, cohen_d, endpoint_features
from evaluate import RES, VARIANT_LABEL, boot_ci, ref_track


def participant_mean(df, by, val="median_mm"):
    """Per participant median of the trial medians, then mean over participants with a bootstrap CI."""
    p = df.groupby(["pid"] + by)[val].median().reset_index()
    out = []
    for k, g in p.groupby(by):
        lo, hi = boot_ci(g[val])
        out.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), n_participants=g.pid.nunique(), value=g[val].mean(),
                        ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out)


def splitwise(df, fn):
    """fn on the test participants and on all participants, stacked with a 'split' column."""
    test = set(splits()["test"])
    return pd.concat([fn(df[df.pid.isin(test)]).assign(split="test"), fn(df).assign(split="all")], ignore_index=True)


def e6a():
    out = []
    for task, f in (("2d", "E1_2d_paths.csv"), ("3d", "E2_3d_paths.csv")):
        p = pd.read_csv(RES / f)
        p = p[p.labels == ("visible" if task == "2d" else "all")]
        # pid: 7-character prefix + study number; pid.str[7] is its first digit (1xx = TD, 2xx = UCP)
        p["group"] = np.where(p.pid.str[7] == "2", "UCP", "TD")
        p["limb"] = [limb_of(a, s) for a, s in zip(p.pid, p.side)]
        r = splitwise(p, lambda d: participant_mean(d, ["group", "limb", "role", "view", "variant", "joint"]))
        out.append(r.assign(task=task))
    a = pd.concat(out, ignore_index=True)
    a["model"] = a.variant.map(lambda v: VARIANT_LABEL.get(v, MODEL_LABEL.get(v, v)))
    a.to_csv(RES / "E6a_group_limb.csv", index=False, float_format="%.4f")


def e6b():
    rows = []
    for n, run in ((4, "ft_b_full_pooled_n4"), (8, "ft_b_full_pooled_n8"), (12, "ft_b_full_pooled_n12"), (17, "ft_b_full_pooled")):
        for f, lab in (("E1_2d.csv", "visible"), ("E2_3d.csv", "all")):
            s = pd.read_csv(RES / f)
            s = s[(s.variant == run) & (s.labels == lab) & (s.group == "all") & (s.role == "active")]
            rows.append(s.assign(n_train=n))
        e3 = pd.read_csv(RES / "E3_screen.csv")
        lab = MODEL_LABEL.get(run, run)
        rows.append(e3[e3.source.str.startswith(lab + ",") & (e3.group == "all") & (e3.subset == "own")].assign(n_train=n))
    pd.concat(rows, ignore_index=True).to_csv(RES / "E6b_scaling.csv", index=False, float_format="%.4f")


def e6c():
    rows = []
    for ms in (0, 33, 67, 100, 200):
        sfx = "" if ms == 0 else f"_shift{ms}ms"
        for f, lab in (("E1_2d", "visible"), ("E2_3d", "all")):
            fp = RES / f"{f}{sfx}.csv"
            if not fp.exists():
                continue
            s = pd.read_csv(fp)
            s = s[(s.labels == lab) & (s.group == "all") & (s.role == "active")
                  & s.variant.isin(["vitpose_b", "rtmpose_wb", "rtmw_x", "sapiens2_1b", "blazepose"])]
            rows.append(s.assign(shift_ms=ms, kind="test-time reference shift"))
    for ms, run in ((0, "ft_b_full_pooled"), (67, "ft_b_full_pooled_shift67"), (200, "ft_b_full_pooled_shift200")):
        for f, lab in (("E1_2d", "visible"), ("E2_3d", "all")):
            s = pd.read_csv(RES / f"{f}.csv")
            s = s[(s.variant == run) & (s.labels == lab) & (s.group == "all") & (s.role == "active")]
            rows.append(s.assign(shift_ms=ms, kind="training-label shift"))
    pd.concat(rows, ignore_index=True).to_csv(RES / "E6c_alignment_shift.csv", index=False, float_format="%.4f")


def e6d():
    e1, e2 = pd.read_csv(RES / "E1_2d.csv"), pd.read_csv(RES / "E2_3d.csv")
    q = pd.concat([e1[e1.labels == "visible"], e2])
    q = q[(q.group == "all") & (q.role == "active") & ~q.variant.isin(["blazepose_topswap", "blazepose_noswap"])]
    q[["split", "view", "model", "variant", "joint", "n_participants", "n_paths", "median_mm", "median_ci_lo", "median_ci_hi",
       "coverage"]].to_csv(RES / "E6d_views.csv", index=False, float_format="%.4f")


def e6e():
    rows = []
    for sfx, filt in (("", "6 Hz filtered"), ("_nofilter", "raw")):
        for f, lab in (("E1_2d", "visible"), ("E2_3d", "all")):
            fp = RES / f"{f}{sfx}.csv"
            if fp.exists():
                s = pd.read_csv(fp)
                rows.append(s[(s.labels == lab) & (s.group == "all") & (s.role == "active")].assign(filtering=filt))
    e = pd.concat(rows, ignore_index=True)
    e = e[~e.variant.str.startswith(("ft_", "seq_"))]
    e.to_csv(RES / "E6e_filtering.csv", index=False, float_format="%.4f")


def e6f():
    m = manifest()
    m = m[m.vicon_valid & ~m.trajectory_qa_failed]
    rec = []
    for _, r in m.iterrows():
        v = markers(r.pid)[0].get(int(r.aligned_mat_row))
        if v is None:
            continue
        step = int(round(v["fps"] / FS))
        x = ref_track(v, str(r.hand)[0], "fingertip")[::step]
        rec.append(dict(pid=r.pid, row=int(r.aligned_mat_row), group="UCP" if r.motor_impaired else "TD",
                        limb=limb_of(r.pid, r.hand), video_valid=bool(r.top_valid and r.side_valid), top_valid=bool(r.top_valid),
                        move_duration_s=r.move_duration_s, **endpoint_features(x, v["t"][::step])))
    d = pd.DataFrame(rec)
    d.to_csv(RES / "E6f_selection_paths.csv", index=False, float_format="%.5f")
    out = []
    for group, g in d.groupby("group"):
        for f in FEATURES + ["move_duration_s"]:
            a, b = g[g.video_valid][f], g[~g.video_valid][f]
            out.append(dict(group=group, feature=f, n_pass=int(a.notna().sum()), n_fail=int(b.notna().sum()),
                            n_participants_fail=g[~g.video_valid].pid.nunique(), mean_pass=a.mean(), mean_fail=b.mean(),
                            d_pass_vs_fail=cohen_d(a, b), share_pass=g.video_valid.mean()))
    pd.DataFrame(out).to_csv(RES / "E6f_selection_bias.csv", index=False, float_format="%.4f")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    for name, fn in (("a", e6a), ("b", e6b), ("c", e6c), ("d", e6d), ("e", e6e), ("f", e6f)):
        try:
            fn()
            print(f"E6{name} done", flush=True)
        except FileNotFoundError as e:
            print(f"E6{name} skipped: missing {e.filename}", flush=True)


if __name__ == "__main__":
    main()
