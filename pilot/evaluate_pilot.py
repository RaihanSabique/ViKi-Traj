"""Pilot analysis of one participant (supplement), step 3: VICON-game offset, then joint and fingertip errors.

Reads data/<pid>/<pid>_reaching_plain.mat (export_trials.m: the aligned trial table provided with the data, made
outside alignment/vicon), outputs/<pid>/alignment/ (align_trials.py), outputs/<pid>/models/ (run_models.py) and, to
fit the cameras, the visit videos. Writes outputs/<pid>/eval/:
    vicon_sync.csv                per trial: VICON-game lag (s) from the game trajectory and from the video; group lag
    calibration.csv, cameras.npz  per video: DLT camera (3x4, VICON mm -> image px) and its reprojection error
    joint_errors.csv              per model, filter, joint, space (3D, Top, Side) and arm: error statistics (mm)
    joint_errors_frames.parquet   every compared sample
    trajectory_errors.csv         per trial and source: fingertip error on the screen against the game trajectory (mm)
    trajectory_samples.parquet    the mapped trajectories
Every VICON time is corrected by the lag of its group (VICON recording) in vicon_sync.csv.

    python pilot/evaluate_pilot.py --participant PID [--reuse_cameras] [--resync]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import savgol_filter

from align_trials import VISITS, load_trials

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))                    # src/ modules
from blobs import detect_candidates                                                           # noqa: E402
from vicon_eval import (apply_affine3, dlt, fit_affine3, fit_robust, interp_track, project,  # noqa: E402
                        px_per_mm, refine_P, smooth)

DATA = ROOT / "data"
OUT = ROOT / "outputs"
JOINTS = ["shoulder", "elbow", "wrist", "hand", "finger"]
THR = {"vit": 0.3, "rtm": 0.3, "mp": 0.5}
MODEL_NAMES = {"vit": "ViTPose+", "rtm": "RTMPose", "mp": "MediaPipe"}
# index of each joint per model and side (-1 = not provided); mp hand/finger come from the
# hand landmarker (21 points: 9 = middle MCP, 8 = index tip)
KP = {
    "vit": {"L": [5, 7, 9, -1, -1], "R": [6, 8, 10, -1, -1]},
    "rtm": {"L": [5, 7, 9, 91 + 9, 91 + 8], "R": [6, 8, 10, 112 + 9, 112 + 8]},
    "mp": {"L": [11, 13, 15, "h9", "h8"], "R": [12, 14, 16, "h9", "h8"]},
}


# ----------------------------------------------------------------------------- data
def load_vicon(participant):
    """{row: VICON data of the trial}: marker labels, positions X (3, markers, samples) in mm, sample times t and
    trial start rel (s, VICON clock)."""
    m = sio.loadmat(DATA / participant / f"{participant}_reaching_plain.mat",
                    squeeze_me=True, struct_as_record=False)
    out = {}
    for i, r in enumerate(np.atleast_1d(m["trials"])):
        out[i + 1] = dict(labels=list(r.labels), X=np.asarray(r.position_data_interpolated, float),
                          t=np.asarray(r.time, float), rel=float(r.relative_start_sec))
    return out


def vicon_proxy(v, side, joint):
    """(N, 3) VICON position of a joint proxy for one side ('L'/'R'): shoulder SHO (acromion, 3-5 cm above the joint
    centre), elbow ELB (lateral epicondyle), wrist WRA/WRB midpoint, hand HND (middle-finger knuckle), finger FIN
    (index tip)."""
    g = lambda n: v["X"][:, v["labels"].index(side + n), :].T
    return {"shoulder": lambda: g("SHO"), "elbow": lambda: g("ELB"),
            "wrist": lambda: (g("WRA") + g("WRB")) / 2, "hand": lambda: g("HND"),
            "finger": lambda: g("FIN")}[joint]()


# BlazePose labels left and right as if the person faced the camera. The top camera films the participant from behind,
# so there its left arm is the participant's right (active-wrist error against VICON 455 px as labelled, 34 px
# mirrored), and the labels are mirrored in that view. joint_errors also scores the unmirrored labels (mp_unmirrored).
MP_MIRRORED_VIEWS = {"Top"}


def model_joints(z, model, side):
    """(T, 5, 2) image positions of JOINTS for one model and side; NaN if absent/low score."""
    if model == "mp" and z.get("view") in MP_MIRRORED_VIEWS and not z.get("raw_mp", False):
        side = {"L": "R", "R": "L"}[side]
    T = len(z["frame"])
    out = np.full((T, len(JOINTS), 2), np.nan)
    k, s = z[f"{model}_k"], z[f"{model}_s"]
    for j, idx in enumerate(KP[model][side]):
        if idx == -1:
            continue
        if isinstance(idx, str):                       # MediaPipe hand landmarker
            out[:, j] = z[f"mp_hand_{side}"][:, int(idx[1:])]
        else:
            ok = s[:, idx] >= THR[model]
            out[ok, j] = k[ok, idx]
    if z.get("smooth", False):
        fps = float(z["fps"])
        for j in range(len(JOINTS)):
            out[:, j] = smooth(out[:, j], fps)
    return out


# ----------------------------------------------------------------------------- cameras
def triangulate(Pa, xa, Pb, xb):
    """Linear two-view triangulation of matched (N, 2) points."""
    out = np.full((len(xa), 3), np.nan)
    for i, (a, b) in enumerate(zip(xa, xb)):
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        A = np.stack([a[0] * Pa[2] - Pa[0], a[1] * Pa[2] - Pa[1],
                      b[0] * Pb[2] - Pb[0], b[1] * Pb[2] - Pb[1]])
        h = np.linalg.svd(A)[2][-1]
        out[i] = h[:3] / h[3]
    return out


# ----------------------------------------------------------------------------- context
class Context:
    """Trials, VICON data, video timing and model outputs of one participant, with the VICON clock correction of
    eval/vicon_sync.csv when that file exists."""

    def __init__(self, participant):
        self.p = participant
        adir = OUT / participant / "alignment"
        self.seg = pd.read_csv(adir / "trial_segments.csv")
        self.summ = pd.read_csv(adir / "alignment_summary.csv")
        self.trials = load_trials(participant).set_index("row")
        self.vicon = load_vicon(participant)
        self.t0 = {v: g.start.min() for v, g in self.trials.groupby("visit")}
        self.mdir = OUT / participant / "models"
        f = OUT / participant / "eval" / "vicon_sync.csv"
        self.sync = None
        if f.exists():
            g = pd.read_csv(f).drop_duplicates(["visit", "group"])
            self.sync = {(r.visit, r.group): r.group_lag_s for r in g.itertuples()}

    def line(self, visit, view):
        """Drift slope, intercept (s) and file name of the video of one visit and view: video time = slope x visit
        clock + intercept."""
        r = self.summ[(self.summ.visit == visit) & (self.summ.view == view)].iloc[0]
        return r.drift_slope, r.intercept_s, r.video

    def vicon_clock(self, row, corrected=True):
        """Visit-clock time (s since the visit's first trial start) of each VICON sample, corrected by the lag of the
        trial's group when vicon_sync.csv exists."""
        tr, v = self.trials.loc[row], self.vicon[row]
        c = (tr.start - self.t0[tr.visit]).total_seconds() + (v["t"] - v["rel"])
        if corrected and self.sync is not None:
            c = c - self.sync.get((tr.visit, tr.group), 0.0)
        return c

    def model(self, row, view):
        """Model outputs of one trial and view (run_models.py), with the visit-clock time of each frame (clock)."""
        tr = self.trials.loc[row]
        z = dict(np.load(self.mdir / f"{tr.visit}_{view}_T{row:02d}.npz"))
        a, b, _ = self.line(tr.visit, view)
        z["clock"] = (z["frame"] / float(z["fps"]) - b) / a
        z["view"] = view
        return z


# ----------------------------------------------------------------------------- VICON sync
def estimate_vicon_sync(ctx, search=3.0, step=0.01):
    """Per trial: the lag (s) at which the VICON fingertip best matches the game trajectory (affine fit within the
    trial), and an independent video-only lag (VICON wrist speed against the side-view ViTPose+ wrist speed). The group
    lag (median game lag of the trials of one VICON recording) is the correction: a VICON sample at raw visit-clock
    time r belongs to game time r - lag."""
    rows = []
    for row, tr in ctx.trials.iterrows():
        v, vc = ctx.vicon[row], ctx.vicon_clock(row, corrected=False)
        side = tr.hand[0]
        c0 = (tr.start - ctx.t0[tr.visit]).total_seconds() + tr.rmov_t
        fin = vicon_proxy(v, side, "finger")
        best = (np.inf, 0.0)
        for lag in np.arange(-search, search + 1e-9, step):
            X = interp_track(vc, fin, c0 + lag)
            ok = ~np.isnan(X).any(1)
            if ok.sum() < 0.6 * len(ok):
                continue
            M = fit_affine3(X[ok], tr.rmov_xy[ok])
            best = min(best, (np.median(np.linalg.norm(apply_affine3(M, X[ok]) - tr.rmov_xy[ok], axis=1)), lag))
        z = ctx.model(row, "Side")
        z["smooth"] = True
        w2, t2 = model_joints(z, "vit", side)[:, 2], z["clock"]
        sp_v = np.linalg.norm(np.gradient(w2, t2, axis=0), axis=1)
        sp_w = np.linalg.norm(np.gradient(vicon_proxy(v, side, "wrist"), vc, axis=0), axis=1)
        bv = (-np.inf, 0.0)
        for lag in np.arange(-search, search + 1e-9, step):
            sw = np.interp(t2 + lag, vc, sp_w, left=np.nan, right=np.nan)
            ok = ~np.isnan(sw) & ~np.isnan(sp_v)
            if ok.sum() > 100:
                bv = max(bv, (np.corrcoef(sw[ok], sp_v[ok])[0, 1], lag))
        rows.append(dict(visit=tr.visit, group=tr.group, row=row, game_lag_s=best[1],
                         game_fit_median_px=best[0], video_lag_s=bv[1], video_r=bv[0]))
    df = pd.DataFrame(rows)
    df["group_lag_s"] = df.groupby(["visit", "group"]).game_lag_s.transform("median")
    return df


# ----------------------------------------------------------------------------- calibration
def calibrate(ctx, visit, view, frame_step=10):
    """DLT camera P (3x4, VICON mm -> image px) of the video of one visit and view, and its fit statistics. P starts
    from the pose models' elbows and wrists against the VICON markers and is refitted on VICON markers matched to
    white image blobs with a shrinking radius, so the camera does not depend on a pose model. Fitted on the even
    trials and tested on the odd ones; the returned P is refitted on all trials."""
    rows = ctx.trials[ctx.trials.visit == visit].index.tolist()
    a, b, video = ctx.line(visit, view)
    cap = cv2.VideoCapture(str(DATA / ctx.p / video))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    sc = W / 1920.0
    # 1) initial camera from the pose models' elbows / wrists (median of the 3 models)
    X, x = [], []
    for row in rows:
        z, v, vc = ctx.model(row, view), ctx.vicon[row], ctx.vicon_clock(row)
        for side in "LR":
            stack = np.stack([model_joints(z, m, side) for m in KP])            # (M, T, 5, 2)
            fused = np.nanmedian(stack, 0)
            for j in (1, 2):
                vx = interp_track(vc, vicon_proxy(v, side, JOINTS[j]), z["clock"])
                ok = ~np.isnan(vx).any(1) & ~np.isnan(fused[:, j]).any(1)
                X.append(vx[ok][::3]); x.append(fused[ok, j][::3])
    X, x = np.concatenate(X), np.concatenate(x)
    P = fit_robust(X, x, thr=40 * sc)
    # 2) refit on VICON markers <-> white blobs, shrinking the match radius
    samples = []                                       # (row, all-marker 3-D, blob 2-D)
    for row in rows:
        z, v, vc = ctx.model(row, view), ctx.vicon[row], ctx.vicon_clock(row)
        inside = (z["clock"] >= vc[0]) & (z["clock"] <= vc[-1])
        for f, c in zip(z["frame"][inside][::frame_step], z["clock"][inside][::frame_step]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, im = cap.read()
            if not ok:
                continue
            k = np.searchsorted(vc, c)
            Xm = v["X"][:, :, min(k, v["X"].shape[2] - 1)].T                 # (24, 3)
            samples.append((row, Xm, detect_candidates(im)[:, :2]))

    def match(P, radius, which):
        Xs, xs, rs = [], [], []
        for row, Xm, bl in samples:
            if which(row) is False or len(bl) == 0:
                continue
            ok = ~np.isnan(Xm).any(1)
            pr = project(P, Xm[ok])
            D = np.linalg.norm(pr[:, None] - bl[None], axis=2)
            i = D.argmin(1)
            mutual = D.argmin(0)[i] == np.arange(len(pr))
            good = (D[np.arange(len(pr)), i] < radius) & mutual
            Xs.append(Xm[ok][good]); xs.append(bl[i[good]]); rs.append(D[np.arange(len(pr)), i][good])
        return np.concatenate(Xs), np.concatenate(xs), np.concatenate(rs)

    train = lambda row: row % 2 == 0
    rmin = max(10 * sc, 8.0)                           # small markers in the 720p top view
    for radius in (40 * sc, 25 * sc, 15 * sc, rmin):
        Xs, xs, _ = match(P, radius, train)
        P = refine_P(dlt(Xs, xs), Xs, xs)
    _, _, err_test = match(P, rmin, lambda row: row % 2 == 1)
    _, _, err_train = match(P, rmin, train)
    Xs, xs, _ = match(P, rmin, lambda row: True)
    P_all = refine_P(dlt(Xs, xs), Xs, xs)
    stats = dict(visit=visit, view=view, video=video, n_init_pairs=len(X), n_blob_pairs=len(Xs),
                 reproj_train_median_px=np.median(err_train), reproj_test_median_px=np.median(err_test),
                 reproj_test_p90_px=np.percentile(err_test, 90),
                 px_per_mm_median=float(np.median(px_per_mm(P_all, Xs))))
    return P_all, stats


# ----------------------------------------------------------------------------- joint errors
def joint_errors(ctx, cams):
    """Per-sample errors for every model, joint, side and space. `filtered` = model tracks
    low-passed at 6 Hz (primary); unfiltered kept for reference. `mp_unmirrored` = MediaPipe
    in the top view without the left/right mirroring (shows the raw failure)."""
    recs = []
    variants = [(m, m, False) for m in KP] + [("mp_unmirrored", "mp", True)]
    for row, tr in ctx.trials.iterrows():
        v, vc = ctx.vicon[row], ctx.vicon_clock(row)
        active = tr.hand[0]
        c = vc[::3]                                                     # ~33 Hz VICON samples
        for filt in (True, False):
            zs = {}
            for view in ("Top", "Side"):
                zs[view] = ctx.model(row, view)
                zs[view]["smooth"] = filt
            for side in "LR":
                role = "active" if side == active else "resting"
                for name, m, raw in variants:
                    if raw:
                        for z in zs.values():
                            z["raw_mp"] = True
                    mj = {view: model_joints(zs[view], m, side) for view in zs}
                    for z in zs.values():
                        z["raw_mp"] = False
                    for j, jn in enumerate(JOINTS):
                        if KP[m][side][j] == -1:
                            continue
                        ref = vicon_proxy(v, side, jn)
                        base = dict(row=row, visit=tr.visit, hand=tr.hand, side=side, role=role,
                                    model=name, joint=jn, filtered=filt)
                        for view, z in zs.items():
                            if raw and view not in MP_MIRRORED_VIEWS:
                                continue
                            P = cams[(tr.visit, view)]
                            inside = (z["clock"] >= vc[0]) & (z["clock"] <= vc[-1])
                            Xv = interp_track(vc, ref, z["clock"][inside])
                            pred = mj[view][inside, j]
                            ok = ~np.isnan(Xv).any(1) & ~np.isnan(pred).any(1)
                            if not ok.any():
                                continue
                            e = np.linalg.norm(project(P, Xv[ok]) - pred[ok], axis=1)
                            recs.append(pd.DataFrame(dict(base, space=view, err_px=e,
                                                          err_mm=e / px_per_mm(P, Xv[ok]))))
                        if raw:
                            continue
                        # 3-D: triangulate top + side at the VICON sample times
                        xa = interp_track(zs["Top"]["clock"], mj["Top"][:, j], c)
                        xb = interp_track(zs["Side"]["clock"], mj["Side"][:, j], c)
                        X3 = triangulate(cams[(tr.visit, "Top")], xa, cams[(tr.visit, "Side")], xb)
                        ok = ~np.isnan(X3).any(1) & ~np.isnan(ref[::3]).any(1)
                        if ok.any():
                            recs.append(pd.DataFrame(dict(base, space="3D", err_px=np.nan,
                                                          err_mm=np.linalg.norm(X3[ok] - ref[::3][ok], axis=1))))
    return pd.concat(recs, ignore_index=True)


# ----------------------------------------------------------------------------- trajectories
def screen_sources(ctx, cams, row):
    """Visit-clock times of the game samples of one trial, and the fingertip sources at those times: VICON FIN and,
    per model, the two-view triangulated and the top-view 2-D fingertip."""
    tr = ctx.trials.loc[row]
    v, vc = ctx.vicon[row], ctx.vicon_clock(row)
    side = tr.hand[0]
    c = (tr.start - ctx.t0[tr.visit]).total_seconds() + tr.rmov_t
    src = {"Vicon (3-D)": interp_track(vc, vicon_proxy(v, side, "finger"), c)}
    zs = {view: ctx.model(row, view) for view in ("Top", "Side")}
    for z in zs.values():
        z["smooth"] = True
    for m in KP:
        j = 4 if KP[m][side][4] != -1 else 2                  # ViTPose+: wrist as stand-in
        tag = MODEL_NAMES[m] + ("" if j == 4 else " [wrist]")
        xa = interp_track(zs["Top"]["clock"], model_joints(zs["Top"], m, side)[:, j], c)
        xb = interp_track(zs["Side"]["clock"], model_joints(zs["Side"], m, side)[:, j], c)
        src[f"{tag} (3-D)"] = triangulate(cams[(tr.visit, "Top")], xa, cams[(tr.visit, "Side")], xb)
        src[f"{tag} (top 2-D)"] = xa
    return c, src


def trajectory_errors(ctx, cams):
    """Per trial and source: error (mm) of the fingertip mapped to the screen against the game trajectory, ratio of the
    trajectory lengths and speed correlation. The mapping is fitted on the other trials of the visit with the same
    hand: an affine for 3-D sources, a RANSAC homography for top-view 2-D sources. Also returns the mapped samples and
    the screen scale per visit (game px per mm)."""
    recs, samples, scale = [], [], {}
    for visit, tv in ctx.trials.groupby("visit"):
        per = {row: screen_sources(ctx, cams, row) for row in tv.index}
        game = {row: tv.loc[row, "rmov_xy"] for row in tv.index}
        # screen scale from the VICON fingertip -> screen affine over the whole visit
        X = np.concatenate([per[r][1]["Vicon (3-D)"] for r in tv.index])
        y = np.concatenate([game[r] for r in tv.index])
        ok = ~np.isnan(X).any(1)
        M = fit_affine3(X[ok], y[ok])
        scale[visit] = float(np.linalg.svd(M[:3].T, compute_uv=False).mean())   # game px per mm
        for name in per[tv.index[0]][1]:
            for row in tv.index:          # leave one trial out, same hand only (a keypoint
                                          # sits differently on each hand; the marker does not)
                same = [r for r in tv.index if r != row and tv.loc[r, "hand"] == tv.loc[row, "hand"]]
                tr_X = np.concatenate([per[r][1][name] for r in same])
                tr_y = np.concatenate([game[r] for r in same])
                ok = ~np.isnan(tr_X).any(1)
                if ok.sum() < 50:
                    continue
                Xq, yq = per[row][1][name], game[row]
                if tr_X.shape[1] == 3:
                    pred = np.full_like(yq, np.nan)
                    good = ~np.isnan(Xq).any(1)
                    pred[good] = apply_affine3(fit_affine3(tr_X[ok], tr_y[ok]), Xq[good])
                else:
                    H, _ = cv2.findHomography(tr_X[ok], tr_y[ok], cv2.RANSAC, 30.0)
                    pred = np.full_like(yq, np.nan)
                    good = ~np.isnan(Xq).any(1)
                    if H is None or not good.any():
                        continue
                    pred[good] = cv2.perspectiveTransform(Xq[good][None].astype(np.float64), H)[0]
                e = np.linalg.norm(pred - yq, axis=1) / scale[visit]
                good = ~np.isnan(e)
                if good.sum() < 10:
                    continue
                # trajectory-level kinematics on a ~30 Hz subsample of the game clock
                k = np.arange(0, len(yq), 4)
                k = k[good[k]]
                c = per[row][0]
                sg = lambda xy: savgol_filter(xy[k], 9, 2, axis=0) if len(k) > 9 else xy[k]
                sp = lambda xy: np.linalg.norm(np.gradient(sg(xy), c[k], axis=0), axis=1)
                plen = lambda xy: np.linalg.norm(np.diff(sg(xy), axis=0), axis=1).sum()
                recs.append(dict(visit=visit, row=row, hand=ctx.trials.loc[row, "hand"], source=name,
                                 coverage=good.mean(), rmse_mm=np.sqrt(np.nanmean(e ** 2)),
                                 median_mm=np.nanmedian(e), p90_mm=np.nanpercentile(e, 90),
                                 path_ratio=plen(pred) / plen(yq),
                                 speed_r=np.corrcoef(sp(np.nan_to_num(pred)), sp(yq))[0, 1]))
                samples.append(pd.DataFrame(dict(visit=visit, row=row, source=name, t=per[row][0] - per[row][0][0],
                                                 game_x=yq[:, 0], game_y=yq[:, 1],
                                                 x=pred[:, 0], y=pred[:, 1], err_mm=e)))
    return pd.DataFrame(recs), pd.concat(samples, ignore_index=True), scale


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="XXXXXXX115")
    ap.add_argument("--reuse_cameras", action="store_true", help="load eval/cameras.npz instead of re-fitting")
    ap.add_argument("--resync", action="store_true", help="re-estimate the VICON-game offsets")
    args = ap.parse_args()
    ctx = Context(args.participant)
    odir = OUT / args.participant / "eval"
    odir.mkdir(parents=True, exist_ok=True)
    if not (odir / "vicon_sync.csv").exists() or args.resync:
        sync = estimate_vicon_sync(ctx)
        sync.to_csv(odir / "vicon_sync.csv", index=False, float_format="%.4f")
        ctx = Context(args.participant)                      # reload with the correction
        g = sync.groupby(["visit", "group"]).agg(group_lag_s=("group_lag_s", "first"),
                                                 video_lag_s=("video_lag_s", "median"))
        print("VICON-game offset per group (s):\n", g.round(3).to_string())
    cams, cal = {}, []
    if args.reuse_cameras and (odir / "cameras.npz").exists():
        cz = np.load(odir / "cameras.npz")
        cams = {tuple(k.split("_")): cz[k] for k in cz.files}
        print(f"reusing cameras from {odir / 'cameras.npz'}")
    for visit in ([] if cams else sorted(VISITS.values())):
        for view in ("Top", "Side"):
            P, st = calibrate(ctx, visit, view)
            cams[(visit, view)] = P
            cal.append(st)
            print(f"[calib {visit} {view}] {st['n_blob_pairs']} marker-blob pairs; reprojection "
                  f"median train {st['reproj_train_median_px']:.1f} px / test {st['reproj_test_median_px']:.1f} px; "
                  f"{st['px_per_mm_median']:.2f} px/mm")
    if cal:
        pd.DataFrame(cal).to_csv(odir / "calibration.csv", index=False, float_format="%.4f")
        np.savez(odir / "cameras.npz", **{f"{k[0]}_{k[1]}": P for k, P in cams.items()})

    je = joint_errors(ctx, cams)
    je.to_parquet(odir / "joint_errors_frames.parquet")
    summ = (je.groupby(["model", "filtered", "joint", "space", "role"]).err_mm
            .agg(n="size", median="median", mean="mean", p90=lambda x: np.percentile(x, 90)).reset_index())
    summ["model"] = summ.model.map({**MODEL_NAMES, "mp_unmirrored": "MediaPipe (unmirrored)"})
    summ.to_csv(odir / "joint_errors.csv", index=False, float_format="%.2f")
    for filt in (True, False):
        piv = summ[(summ.role == "active") & (summ.filtered == filt)].pivot_table(
            index=["space", "joint"], columns="model", values="median")
        print(f"\nMedian error (mm), active arm, {'6 Hz filtered' if filt else 'unfiltered'}:\n", piv.round(1).to_string())

    te, ts, scale = trajectory_errors(ctx, cams)
    te.to_csv(odir / "trajectory_errors.csv", index=False, float_format="%.4f")
    ts.to_parquet(odir / "trajectory_samples.parquet")
    print(f"\ngame px per mm: {scale}")
    print(te.groupby("source")[["rmse_mm", "median_mm", "p90_mm", "coverage", "path_ratio", "speed_r"]]
          .median().sort_values("rmse_mm").round(2).to_string())


if __name__ == "__main__":
    main()
