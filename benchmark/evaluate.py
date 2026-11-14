"""Task 1, 2D and 3D upper-limb pose: E1 (keypoints per view) and E2 (two-view 3D joints) against VICON.

Trials (paths) of the visits that pass the trajectory QA of e3_screen.py. Per participant, over the frames of all its
trials: median, mean and P90 error (mm), PCK@20 mm (a labelled frame without an estimate is a miss) and coverage;
summaries per split, group, view, model, arm and joint. Writes outputs/benchmark/results/E1_2d*.csv and E2_3d*.csv
(summary, _participants, _paths, _skipped).

    python benchmark/evaluate.py --task t1|t2|all                       # t1 = E1, t2 = E2
    python benchmark/evaluate.py --shift-ms 67 --out-suffix _shift67ms  # E6c
    python benchmark/evaluate.py --no-filter --out-suffix _nofilter     # E6e
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from benchlib import (BENCH, JOINT_NAMES, KP, MODEL_LABEL, aux_file, clip_stem, frame_clock, manifest,
                      marker_index, markers, model_file, model_joints, proxy, set_lr_swap, splits, valid_rows)
from calibrate import CAM, camera_id, interp_markers
from kinematics import lowpass_runs
from vicon_eval import interp_track, project, px_per_mm, smooth

VIS_PX = 8.0
PCK_MM = 20.0
CUTOFF_HZ = 6.0
RES = BENCH / "results"
# (variant, model, mirror argument of benchlib.model_joints); two extra BlazePose variants with fixed side rules
VARIANTS = [(m, m, "auto") for m in KP] + [("blazepose_topswap", "blazepose", "view"), ("blazepose_noswap", "blazepose", False)]
N_MAIN = len(KP)                                       # VARIANTS[:N_MAIN]: one per model, the only ones in E2
VARIANT_LABEL = {**MODEL_LABEL, "blazepose": "BlazePose+Hands (L/R from ViTPose+)",
                 "blazepose_topswap": "BlazePose+Hands (L/R swapped in top view)",
                 "blazepose_noswap": "BlazePose+Hands (own L/R)"}


def cameras():
    c = pd.read_csv(CAM / "calibration.csv")
    return {r.camera: np.load(CAM / f"{r.camera}.npz")["P"] for r in c.itertuples() if r.calib_ok}


# Set by main(): shift_s (s) is added to the reference time (E6c, simulated misalignment);
# filter=False scores raw keypoints against raw markers (E6e)
OPT = dict(shift_s=0.0, filter=True)


def ref_track(v, side, joint):
    """VICON joint proxy (N, 3) in mm, 6 Hz low-passed (raw when OPT['filter'] is False)."""
    x = proxy(v, side, joint)
    if not OPT["filter"]:
        return x
    return np.stack([lowpass_runs(x[:, d], v["fps"], CUTOFF_HZ) for d in range(3)], 1)


def model_tracks(z, model, side, view, mirror):
    """Keypoints (T, 5, 2) px of one side, Hampel filter and 6 Hz low-pass (raw when OPT['filter'] is False)."""
    mj = model_joints(z, model, side, view, mirror)
    if not OPT["filter"]:
        return mj
    fps = float(z["fps"])
    return np.stack([smooth(mj[:, j], fps) for j in range(mj.shape[1])], 1)


def path_stats(e, **key):
    """Summary of the errors of one trial (NaN = labelled frame without an estimate)."""
    got = ~np.isnan(e)
    return dict(key, n_labels=len(e), coverage=got.mean() if len(e) else np.nan,
                median_mm=np.median(e[got]) if got.any() else np.nan, mean_mm=np.mean(e[got]) if got.any() else np.nan,
                pck20=np.mean(np.where(got, e, np.inf) <= PCK_MM) if len(e) else np.nan)


def at(t_src, x, t):
    """x (N, D) linearly interpolated at the times t (NaN outside t_src)."""
    return np.stack([np.interp(t, t_src, x[:, d], left=np.nan, right=np.nan) for d in range(x.shape[1])], 1)


def triangulate(Pa, xa, Pb, xb):
    """Vectorised linear two-view triangulation; NaN where either view is missing."""
    out = np.full((len(xa), 3), np.nan)
    ok = ~np.isnan(xa).any(1) & ~np.isnan(xb).any(1)
    if not ok.any():
        return out
    a, b = xa[ok], xb[ok]
    A = np.stack([a[:, :1] * Pa[2] - Pa[0], a[:, 1:] * Pa[2] - Pa[1],
                  b[:, :1] * Pb[2] - Pb[0], b[:, 1:] * Pb[2] - Pb[1]], 1)            # (N, 4, 4)
    h = np.linalg.svd(A)[2][:, -1]
    out[ok] = h[:, :3] / h[:, 3:]
    return out


def role_of(side, hand):
    return "active" if side == str(hand)[0] else "resting"


# ----------------------------------------------------------------------------- E1 (--task t1)
def t1(cams):
    """E1 errors per frame of the movement window, in mm through the local camera scale (px_per_mm), on trials with
    vicon_valid and <view>_valid and a calibrated camera. Label sets: 'all' frames, and 'visible' blob frames
    (10 Hz) where every marker of the joint has a white blob within VIS_PX px of its projection."""
    acc = defaultdict(list)          # (pid, view, variant, role, joint, labels) -> [err_mm with NaN = missed]
    acc_px = defaultdict(list)
    paths, skipped, prec = [], [], []
    for view in ("top", "side"):
        for _, r in valid_rows(view).iterrows():
            pid, row = r.pid, int(r.aligned_mat_row)
            cid = camera_id(pid, int(r.visit), view, r[f"{view}_source_video"])
            if cid not in cams:
                skipped.append(dict(pid=pid, row=row, view=view, reason="camera failed calibration"))
                continue
            stem, P = clip_stem(r, view), cams[cid]
            zs = {m: dict(np.load(model_file(m, pid, view, stem))) for m in KP if model_file(m, pid, view, stem).exists()}
            if not zs or not aux_file("blobs", pid, view, stem).exists():
                skipped.append(dict(pid=pid, row=row, view=view, reason="missing model output or blobs"))
                continue
            bz = np.load(aux_file("blobs", pid, view, stem))
            set_lr_swap(zs, pid, view, r[f"{view}_source_video"])
            v = markers(pid)[0][row]
            fr = next(iter(zs.values()))["frame"]
            s0, n0 = int(r[f"{view}_start_frame"]), int(r[f"{view}_n_frames"])
            win = (fr >= s0) & (fr < s0 + n0)
            t = frame_clock(pid, row, view, fr)
            # visibility at the blob frames
            bfr, bxy = bz["frame"], np.split(bz["xy"], np.cumsum(bz["n"])[:-1])
            bpos = {f: i for i, f in enumerate(bfr)}
            Xm = interp_markers(v, frame_clock(pid, row, view, bfr))                     # (B, 24, 3)
            tracks = {(name, side): model_tracks(zs[m], m, side, view, mir)
                      for name, m, mir in VARIANTS if m in zs for side in "LR"}
            for side in "LR":
                role = role_of(side, r.hand)
                for j, jn in enumerate(JOINT_NAMES):
                    ref = at(v["t"], ref_track(v, side, jn), t + OPT["shift_s"])          # (T, 3)
                    has = win & ~np.isnan(ref).any(1)
                    if not has.any():
                        continue
                    proj = np.full((len(t), 2), np.nan)
                    proj[has] = project(P, ref[has])
                    scale = np.full(len(t), np.nan)
                    scale[has] = px_per_mm(P, ref[has])
                    vis = np.zeros(len(t), bool)
                    mi = marker_index(v, side, jn)
                    for i in np.flatnonzero(has):
                        b = bpos.get(fr[i])
                        if b is None or len(bxy[b]) == 0 or np.isnan(Xm[b, mi]).any():
                            continue
                        pm = project(P, Xm[b, mi])
                        vis[i] = (np.linalg.norm(pm[:, None] - bxy[b][None], axis=2).min(1) <= VIS_PX).all()
                    for name, m, _ in VARIANTS:
                        if (name, side) not in tracks or KP[m][side][j] == -1:
                            continue
                        e_px = np.linalg.norm(tracks[(name, side)][:, j] - proj, axis=1)
                        e_mm = e_px / scale
                        for lab, sel in (("all", has), ("visible", has & vis)):
                            key = (pid, view, name, role, jn, lab)
                            acc[key].append(e_mm[sel].astype(np.float32))
                            acc_px[key].append(e_px[sel].astype(np.float32))
                            prec.append(path_stats(e_mm[sel], pid=pid, row=row, view=view, variant=name, side=side,
                                                   hand=r.hand, role=role, joint=jn, labels=lab))
            paths.append(dict(pid=pid, row=row, view=view))
    return acc, acc_px, pd.DataFrame(paths), pd.DataFrame(skipped), pd.DataFrame(prec)


# ----------------------------------------------------------------------------- E2 (--task t2)
def t2(cams):
    """E2 errors of the two-view triangulation against the 3D reference, on trials valid in VICON and both views
    with both cameras calibrated, at every third VICON sample (33 Hz)."""
    acc = defaultdict(list)
    paths, skipped, prec = [], [], []
    m = manifest()
    rows = m[m.vicon_valid & m.top_valid & m.side_valid & ~m.trajectory_qa_failed]
    for _, r in rows.iterrows():
        pid, row = r.pid, int(r.aligned_mat_row)
        cid = {view: camera_id(pid, int(r.visit), view, r[f"{view}_source_video"]) for view in ("top", "side")}
        if not all(c in cams for c in cid.values()):
            skipped.append(dict(pid=pid, row=row, reason="camera failed calibration"))
            continue
        zs = {view: {mm: dict(np.load(f)) for mm in KP if (f := model_file(mm, pid, view, clip_stem(r, view))).exists()}
              for view in ("top", "side")}
        if not zs["top"] or not zs["side"]:
            skipped.append(dict(pid=pid, row=row, reason="missing model output"))
            continue
        for view in zs:
            set_lr_swap(zs[view], pid, view, r[f"{view}_source_video"])
        v = markers(pid)[0][row]
        tv = v["t"][::3]
        clocks = {view: frame_clock(pid, row, view, next(iter(zs[view].values()))["frame"]) for view in zs}
        for side in "LR":
            role = role_of(side, r.hand)
            refs = {jn: ref_track(v, side, jn)[::3] for jn in JOINT_NAMES}
            for name, mm, mir in VARIANTS[:N_MAIN]:
                if mm not in zs["top"] or mm not in zs["side"]:
                    continue
                tr = {view: model_tracks(zs[view][mm], mm, side, view, mir) for view in zs}
                for j, jn in enumerate(JOINT_NAMES):
                    if KP[mm][side][j] == -1:
                        continue
                    ref = refs[jn]
                    has = ~np.isnan(ref).any(1)
                    xa = interp_track(clocks["top"], tr["top"][:, j], tv - OPT["shift_s"])
                    xb = interp_track(clocks["side"], tr["side"][:, j], tv - OPT["shift_s"])
                    X3 = triangulate(cams[cid["top"]], xa, cams[cid["side"]], xb)
                    e = np.linalg.norm(X3 - ref, axis=1)
                    acc[(pid, "3d", name, role, jn, "all")].append(e[has].astype(np.float32))
                    prec.append(path_stats(e[has], pid=pid, row=row, view="3d", variant=name, side=side, hand=r.hand,
                                           role=role, joint=jn, labels="all"))
        paths.append(dict(pid=pid, row=row, view="3d"))
    return acc, pd.DataFrame(paths), pd.DataFrame(skipped), pd.DataFrame(prec)


# ----------------------------------------------------------------------------- aggregation
def participant_table(acc, acc_px=None):
    """Metrics of each participant from the pooled frame errors of its trials (NaN = missed)."""
    rec = []
    for key, parts in acc.items():
        e = np.concatenate(parts)
        if len(e) == 0:
            continue
        got = ~np.isnan(e)
        d = dict(zip(["pid", "view", "variant", "role", "joint", "labels"], key), n_labels=len(e),
                 coverage=got.mean(), pck20=np.mean(np.where(got, e, np.inf) <= PCK_MM),
                 median_mm=np.median(e[got]) if got.any() else np.nan,
                 mean_mm=np.mean(e[got]) if got.any() else np.nan,
                 p90_mm=np.percentile(e[got], 90) if got.any() else np.nan)
        if acc_px is not None:
            ep = np.concatenate(acc_px[key])
            d["median_px"] = np.nanmedian(ep) if got.any() else np.nan
        rec.append(d)
    p = pd.DataFrame(rec)
    # pid: 7-character prefix + study number; pid.str[7] is its first digit (1xx = TD, 2xx = UCP)
    p["group"] = np.where(p.pid.str[7] == "2", "UCP", "TD")
    return p


def boot_ci(x, n=1000, seed=2027):
    """95 % percentile bootstrap CI of the mean of x (n resamples, NaN dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def summary(p, paths):
    """Mean over participants of the per-participant metrics, with the bootstrap CI of the median error."""
    test = set(splits()["test"])
    rec = []
    keys = ["view", "variant", "role", "joint", "labels"]
    for split, ps in (("test", p[p.pid.isin(test)]), ("all", p)):
        for group in ("all", "TD", "UCP"):
            q = ps if group == "all" else ps[ps.group == group]
            for k, g in q.groupby(keys):
                lo, hi = boot_ci(g.median_mm)
                pp = paths[paths.pid.isin(g.pid) & (paths.view == k[0])]
                d = dict(zip(keys, k), split=split, group=group, model=VARIANT_LABEL[k[1]],
                         n_participants=g.pid.nunique(), n_paths=len(pp), n_labels=int(g.n_labels.sum()),
                         median_mm=g.median_mm.mean(), median_ci_lo=lo, median_ci_hi=hi,
                         mean_mm=g.mean_mm.mean(), p90_mm=g.p90_mm.mean(), pck20=g.pck20.mean(),
                         coverage=g.coverage.mean())
                if "median_px" in g:
                    d["median_px"] = g.median_px.mean()
                rec.append(d)
    cols = ["split", "group", "view", "model", "variant", "role", "joint", "labels", "n_participants", "n_paths",
            "n_labels", "median_mm", "median_ci_lo", "median_ci_hi", "mean_mm", "p90_mm", "pck20", "coverage"]
    out = pd.DataFrame(rec)
    return out[[c for c in cols + ["median_px"] if c in out]]


def show(s, labels):
    for split in ("test", "all"):
        q = s[(s.split == split) & (s.group == "all") & (s.role == "active") & (s.labels == labels)]
        if len(q):
            print(f"{split} split, active arm, {labels} labels, median mm (mean over participants):")
            piv = q.pivot_table(index=["view", "model"], columns="joint", values="median_mm", sort=False)
            print(piv.reindex(columns=JOINT_NAMES).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["t1", "t2", "all"])
    ap.add_argument("--models", default=None, help="comma-separated variants (default: all)")
    ap.add_argument("--shift-ms", type=float, default=0.0, help="E6c: shift the reference in time (ms)")
    ap.add_argument("--no-filter", action="store_true", help="E6e: raw keypoints vs raw markers")
    ap.add_argument("--participants", default=None, help="comma-separated subset")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    global N_MAIN, valid_rows, manifest                                  # rebound for --models / --participants
    OPT["shift_s"], OPT["filter"] = args.shift_ms / 1000.0, not args.no_filter
    if args.models:
        keep = args.models.split(",")
        VARIANTS[:] = [v for v in VARIANTS if v[0] in keep]
        N_MAIN = sum(1 for v in VARIANTS if v[2] == "auto")
    if args.participants:
        pids, _vr, _mf = args.participants.split(","), valid_rows, manifest
        valid_rows = lambda view, p=None: _vr(view, pids)
        manifest = lambda: _mf()[_mf().pid.isin(pids)]
    sfx = args.out_suffix
    RES.mkdir(parents=True, exist_ok=True)
    cams = cameras()
    if args.task in ("t1", "all"):
        acc, acc_px, paths, skipped, prec = t1(cams)
        prec.to_csv(RES / f"E1_2d_paths{sfx}.csv", index=False, float_format="%.4f")
        p = participant_table(acc, acc_px)
        p.to_csv(RES / f"E1_2d_participants{sfx}.csv", index=False, float_format="%.4f")
        s = summary(p, paths)
        s.to_csv(RES / f"E1_2d{sfx}.csv", index=False, float_format="%.4f")
        skipped.to_csv(RES / f"E1_2d_skipped{sfx}.csv", index=False)
        print(f"E1: {len(paths)} clips scored ({paths.pid.nunique()} participants), {len(skipped)} skipped")
        show(s, "visible")
    if args.task in ("t2", "all"):
        acc, paths, skipped, prec = t2(cams)
        prec.to_csv(RES / f"E2_3d_paths{sfx}.csv", index=False, float_format="%.4f")
        p = participant_table(acc)
        p.to_csv(RES / f"E2_3d_participants{sfx}.csv", index=False, float_format="%.4f")
        s = summary(p, paths)
        s.to_csv(RES / f"E2_3d{sfx}.csv", index=False, float_format="%.4f")
        skipped.to_csv(RES / f"E2_3d_skipped{sfx}.csv", index=False)
        print(f"E2: {len(paths)} paths scored ({paths.pid.nunique()} participants), {len(skipped)} skipped")
        show(s, "all")


if __name__ == "__main__":
    main()
