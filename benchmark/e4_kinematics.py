"""Task 2, kinematic features (E4): joint angles and reach features from the two-view 3D joints against VICON.

Active arm of the trials (paths) of E2: valid in VICON and both views, both cameras calibrated, trajectory QA passed.
3D tracks at 50 Hz on the VICON clock; angle RMSE and range-of-motion error per trial, agreement of the reach features
with VICON (ICC(2,1), Bland-Altman), group and arm effect sizes. Writes outputs/benchmark/tracks3d/<pid>_r<row>.npz
(cache) and results/E4_kin_paths.csv, E4_kin_angles.csv, E4_kin_agreement.csv and E4_kin_effects.csv.

    python benchmark/e4_kinematics.py
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from benchlib import BENCH, KP, MODEL_LABEL, clip_stem, frame_clock, limb_of, manifest, markers, model_file, \
    set_lr_swap, splits
from calibrate import camera_id
from evaluate import RES, boot_ci, cameras, model_tracks, ref_track, triangulate
from kinematics import ldlj, sparc
from vicon_eval import interp_track

FS = 50.0                                              # Hz of the 3D tracks (every second VICON sample)
J4 = ["shoulder", "elbow", "wrist", "fingertip"]       # joints of the 3D tracks
JIDX = {"shoulder": 0, "elbow": 1, "wrist": 2, "hand": 3, "fingertip": 4}     # benchlib.JOINT_NAMES index
TRK = BENCH / "tracks3d"
FEATURES = ["peak_speed", "time_to_peak", "straightness", "n_peaks", "sparc", "ldlj"]


def tracks_for_path(r, cams):
    """{'t': (N,) s, 'vicon' and each model: (4, N, 3) mm in J4 order} of one trial, active arm, at FS on the VICON
    clock. Cached in TRK; models missing from the cache are added."""
    f = TRK / f"{r.pid}_r{int(r.aligned_mat_row)}.npz"
    pid, row, side = r.pid, int(r.aligned_mat_row), str(r.hand)[0]
    out = {}
    if f.exists():
        z = np.load(f)
        out = {k: z[k] for k in z.files}
    avail = [m for m in KP if all(model_file(m, pid, view, clip_stem(r, view)).exists() for view in ("top", "side"))]
    if out and all(m in out for m in avail):
        return out
    v = markers(pid)[0][row]
    step = int(round(v["fps"] / FS))
    t = v["t"][::step]
    out.update({"t": t, "vicon": np.stack([ref_track(v, side, j)[::step] for j in J4])})
    cid = {view: camera_id(pid, int(r.visit), view, r[f"{view}_source_video"]) for view in ("top", "side")}
    zs = {view: {m: dict(np.load(fp)) for m in KP if (fp := model_file(m, pid, view, clip_stem(r, view))).exists()}
          for view in ("top", "side")}
    for view in zs:
        set_lr_swap(zs[view], pid, view, r[f"{view}_source_video"])
    for m in avail:
        if m in out:
            continue
        tr, ck = {}, {}
        for view in ("top", "side"):
            z = zs[view][m]
            ck[view] = frame_clock(pid, row, view, z["frame"])
            tr[view] = model_tracks(z, m, side, view, "auto")
        X = np.full((len(J4), len(t), 3), np.nan)
        for a, j in enumerate(J4):
            if KP[m][side][JIDX[j]] == -1:
                continue
            xa = interp_track(ck["top"], tr["top"][:, JIDX[j]], t)
            xb = interp_track(ck["side"], tr["side"][:, JIDX[j]], t)
            X[a] = triangulate(cams[cid["top"]], xa, cams[cid["side"]], xb)
        out[m] = X
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def angle(a, b, c):
    """Angle ABC (deg) per sample."""
    u, w = a - b, c - b
    cos = np.sum(u * w, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def angles(X):
    """Elbow flexion (180 - angle shoulder-elbow-wrist) and shoulder elevation (upper arm against the downward
    vertical, VICON z up), deg, from (4, N, 3) joints in J4 order."""
    elbow = 180.0 - angle(X[0], X[1], X[2])
    up = X[1] - X[0]
    elev = np.degrees(np.arccos(np.clip(-up[:, 2] / np.linalg.norm(up, axis=-1), -1, 1)))
    return {"elbow_flexion": elbow, "shoulder_elevation": elev}


def fill_gaps(x, max_gap=int(0.2 * FS)):
    """Linear interpolation over interior gaps of at most max_gap samples (0.2 s)."""
    out = x.copy()
    ok = ~np.isnan(x).any(1)
    if ok.sum() < 2:
        return out
    idx = np.arange(len(x))
    for d in range(x.shape[1]):
        out[:, d] = np.interp(idx, idx[ok], x[ok, d])
    # keep NaN in gaps longer than max_gap and at both ends
    gaps = np.flatnonzero(np.diff(np.r_[0, (~ok).astype(int), 0]))
    for s, e in zip(gaps[::2], gaps[1::2]):
        if e - s > max_gap or s == 0 or e == len(x):
            out[s:e] = np.nan
    return out


def endpoint_features(p, t):
    """Reach features of an (N, 3) mm track: peak speed (mm/s), time to peak speed (s), straightness (start-end
    distance / distance travelled), number of speed peaks (prominence >= 10 % of the peak), SPARC and LDLJ
    (kinematics.py). NaN when less than 80 % of the samples exist."""
    if np.mean(~np.isnan(p).any(1)) < 0.8:
        return {f: np.nan for f in FEATURES}
    p = fill_gaps(p)
    ok = ~np.isnan(p).any(1)
    p, t = p[ok], t[ok]
    if len(p) < 20:
        return {f: np.nan for f in FEATURES}
    sp = np.linalg.norm(np.gradient(p, t, axis=0), axis=1)
    i = int(np.argmax(sp))
    length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    pk, _ = find_peaks(sp, prominence=0.1 * sp.max())
    return dict(peak_speed=sp.max(), time_to_peak=t[i] - t[0], straightness=np.linalg.norm(p[-1] - p[0]) / max(length, 1e-6),
                n_peaks=max(len(pk), 1), sparc=sparc(sp, FS), ldlj=ldlj(sp, t))


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single measures (Shrout & Fleiss)."""
    x = np.c_[a, b]
    x = x[~np.isnan(x).any(1)]
    n, k = x.shape
    if n < 5:
        return np.nan
    gm = x.mean()
    msr = k * np.sum((x.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((x.mean(0) - gm) ** 2) / (k - 1)
    mse = np.sum((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2) / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def cohen_d(a, b):
    """Cohen's d of a against b with the pooled SD (NaN dropped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cams = cameras()
    m = manifest()
    rows = m[m.vicon_valid & m.top_valid & m.side_valid & ~m.trajectory_qa_failed]
    recs = []
    for k, (_, r) in enumerate(rows.iterrows()):
        cid = [camera_id(r.pid, int(r.visit), v, r[f"{v}_source_video"]) for v in ("top", "side")]
        if not all(c in cams for c in cid):
            continue
        tr = tracks_for_path(r, cams)
        t = tr["t"]
        ref_ang = angles(tr["vicon"])
        base = dict(pid=r.pid, row=int(r.aligned_mat_row), visit=r.visit, hand=r.hand, group="UCP" if r.motor_impaired else "TD",
                    limb=limb_of(r.pid, r.hand))
        srcs = {"vicon": tr["vicon"]} | {s: tr[s] for s in KP if s in tr}
        for s, X in srcs.items():
            d = dict(base, source=s)
            ang = angles(X)
            for a, y in ang.items():
                x0 = ref_ang[a]
                ok = ~np.isnan(y) & ~np.isnan(x0)
                d[f"{a}_rmse_deg"] = np.sqrt(np.mean((y[ok] - x0[ok]) ** 2)) if ok.sum() > 20 else np.nan
                d[f"{a}_rom_deg"] = np.nanpercentile(y, 95) - np.nanpercentile(y, 5) if np.isfinite(y).sum() > 20 else np.nan
                d[f"{a}_coverage"] = ok.mean()
            for ep, a in (("fingertip", 3), ("wrist", 2)):
                for fn, val in endpoint_features(X[a], t).items():
                    d[f"{ep}_{fn}"] = val
            recs.append(d)
        if k % 50 == 0:
            print(f"{k}/{len(rows)} paths", flush=True)
    P = pd.DataFrame(recs)
    v = P[P.source == "vicon"].set_index(["pid", "row"])
    for a in ("elbow_flexion", "shoulder_elevation"):
        P[f"{a}_rom_err_deg"] = P[f"{a}_rom_deg"].values - v.loc[list(zip(P.pid, P.row)), f"{a}_rom_deg"].values
    P.to_csv(RES / "E4_kin_paths.csv", index=False, float_format="%.5f")

    test = set(splits()["test"])
    label = lambda s: "VICON" if s == "vicon" else MODEL_LABEL.get(s, s)
    feats = [f"{ep}_{f}" for ep in ("fingertip", "wrist") for f in FEATURES]
    # angles: per participant median over trials, mean over participants with a bootstrap CI
    arec = []
    for split, Q in (("test", P[P.pid.isin(test)]), ("all", P)):
        for group in ("all", "TD", "UCP"):
            q = Q if group == "all" else Q[Q.group == group]
            for s, g in q[q.source != "vicon"].groupby("source"):
                for a in ("elbow_flexion", "shoulder_elevation"):
                    pp = g.groupby("pid")[[f"{a}_rmse_deg", f"{a}_rom_err_deg"]].median()
                    lo, hi = boot_ci(pp[f"{a}_rmse_deg"])
                    arec.append(dict(split=split, group=group, source=s, model=label(s), angle=a, n_participants=len(pp),
                                     n_paths=len(g), rmse_deg=pp[f"{a}_rmse_deg"].mean(), rmse_ci_lo=lo, rmse_ci_hi=hi,
                                     rom_err_deg=pp[f"{a}_rom_err_deg"].mean(), abs_rom_err_deg=pp[f"{a}_rom_err_deg"].abs().mean()))
    pd.DataFrame(arec).to_csv(RES / "E4_kin_angles.csv", index=False, float_format="%.4f")
    # feature agreement with VICON over trials
    grec = []
    for split, Q in (("test", P[P.pid.isin(test)]), ("all", P)):
        vv = Q[Q.source == "vicon"].set_index(["pid", "row"])
        for s, g in Q[Q.source != "vicon"].groupby("source"):
            g = g.set_index(["pid", "row"])
            for f in feats:
                ref = vv[f].reindex(g.index)
                x, y = ref.values.astype(float), g[f].values.astype(float)
                ok = ~np.isnan(x) & ~np.isnan(y)
                if ok.sum() < 5:
                    continue
                dlt = y[ok] - x[ok]
                grec.append(dict(split=split, source=s, model=label(s), feature=f, n_paths=int(ok.sum()), icc21=icc21(x[ok], y[ok]),
                                 bias=dlt.mean(), loa_lo=dlt.mean() - 1.96 * dlt.std(ddof=1), loa_hi=dlt.mean() + 1.96 * dlt.std(ddof=1),
                                 mae=np.abs(dlt).mean(), vicon_mean=x[ok].mean(), r=np.corrcoef(x[ok], y[ok])[0, 1]))
    pd.DataFrame(grec).to_csv(RES / "E4_kin_agreement.csv", index=False, float_format="%.5f")
    # group effects from each source (participant means); limb from data/Behavioral_Data_Hand_Limb.csv
    # (affected = paretic in UCP, non-dominant in TD)
    erec = []
    for split, Q in (("test", P[P.pid.isin(test)]), ("all", P)):
        for s, g in Q.groupby("source"):
            pm = g.groupby(["pid", "group"])[feats + ["elbow_flexion_rom_deg", "shoulder_elevation_rom_deg"]].mean().reset_index()
            lm = g.groupby(["pid", "group", "limb"])[feats + ["elbow_flexion_rom_deg", "shoulder_elevation_rom_deg"]].mean().reset_index()
            for f in feats + ["elbow_flexion_rom_deg", "shoulder_elevation_rom_deg"]:
                d = dict(split=split, source=s, model=label(s), feature=f,
                         d_ucp_vs_td=cohen_d(pm[pm.group == "UCP"][f], pm[pm.group == "TD"][f]),
                         n_ucp=int(pm[pm.group == "UCP"][f].notna().sum()), n_td=int(pm[pm.group == "TD"][f].notna().sum()))
                for grp in ("UCP", "TD"):
                    w = lm[lm.group == grp].pivot_table(index="pid", columns="limb", values=f)
                    if {"affected", "less_affected"} <= set(w.columns):
                        diff = (w["affected"] - w["less_affected"]).dropna()
                        d[f"dz_affected_vs_less_{grp}"] = diff.mean() / diff.std(ddof=1) if len(diff) > 2 else np.nan
                        d[f"n_pairs_{grp}"] = len(diff)
                erec.append(d)
    pd.DataFrame(erec).to_csv(RES / "E4_kin_effects.csv", index=False, float_format="%.4f")
    print(f"E4: {P[['pid', 'row']].drop_duplicates().shape[0]} paths, {P.pid.nunique()} participants")


if __name__ == "__main__":
    main()
