"""Task 3, end-effector trajectory on the screen (E3): fingertip estimates against the game trajectory.

Each source (VICON fingertip; per model the fingertip, ViTPose+ the wrist, in the top view and from both views) is
mapped to the screen leave one trial out, from the other trials of the same participant, visit and hand. Visits whose
median VICON fingertip RMSE exceeds QA_MAX_MM fail the trajectory QA (qa_vicon_trajectory.csv; dropped from every
experiment). Writes outputs/benchmark/results/E3_screen.csv, E3_screen_paths.csv and E3_screen_participants.csv.

    python benchmark/e3_screen.py [--participants PID,PID] [--out-suffix SFX]
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import savgol_filter

from benchlib import ALIGNED, KP, MODEL_LABEL, QA_FILE, clip_stem, frame_clock, manifest, markers, model_file, set_lr_swap, splits
from calibrate import camera_id
from evaluate import RES, boot_ci, cameras, model_tracks, ref_track, triangulate
from vicon_eval import apply_affine3, fit_affine3, interp_track

FINGER, WRIST = 4, 2                                   # benchlib.JOINT_NAMES index
MIN_FIT = 50                                           # samples needed to fit a screen mapping
QA_MAX_MM = 20.0                                       # trajectory QA: largest median VICON fingertip RMSE of a visit


def game_paths(pid):
    """{row: (absolute game time (s), (N, 2) game px)}: the game trajectory of every trial of a participant."""
    P = np.atleast_1d(sio.loadmat(ALIGNED / f"{pid}_reaching_paths_plain.mat", squeeze_me=True,
                                  struct_as_record=False)["paths"])
    out = {}
    for p in P:
        t = np.atleast_1d(np.asarray(p.RMov_time_step_sec, float))
        xy = np.atleast_2d(np.asarray(p.RMov_path, float))
        n = min(len(t), len(xy))
        out[int(p.row)] = (pd.Timestamp(str(p.trial_start_time)).value / 1e9 + t[:n], xy[:n])
    return out


def sources_for_path(r, cams, game_t):
    """Estimates of one trial at the game times: ({source: (kind '3d' | '2d', model, (N, 3 | 2))}, two_view).
    The two-view sources need both views valid and both cameras calibrated."""
    pid, row, side = r.pid, int(r.aligned_mat_row), str(r.hand)[0]
    out = {}
    v = markers(pid)[0].get(row)
    if v is not None:
        out["Vicon FIN"] = ("3d", "vicon", interp_track(v["t"], ref_track(v, side, "fingertip"), game_t))
    tracks = {}
    for view in ("top", "side"):
        if not r[f"{view}_valid"]:
            continue
        zs = {m: dict(np.load(f)) for m in KP if (f := model_file(m, pid, view, clip_stem(r, view))).exists()}
        set_lr_swap(zs, pid, view, r[f"{view}_source_video"])
        for m, z in zs.items():
            tracks[(view, m)] = (frame_clock(pid, row, view, z["frame"]), model_tracks(z, m, side, view, "auto"))
    cid = {view: camera_id(pid, int(r.visit), view, r[f"{view}_source_video"]) for view in ("top", "side")
           if isinstance(r[f"{view}_source_video"], str)}
    two_view = r.top_valid and r.side_valid and len(cid) == 2 and all(c in cams for c in cid.values())
    for m in KP:
        j = FINGER if KP[m][side][FINGER] != -1 else (WRIST if m.startswith("vitpose") else None)
        if j is None:
            continue
        name = MODEL_LABEL[m] + (" wrist" if j == WRIST else "")
        top = tracks.get(("top", m))
        xa = interp_track(top[0], top[1][:, j], game_t) if top else None
        if xa is not None:
            out[f"{name}, top"] = ("2d", m, xa)
        side_tr = tracks.get(("side", m))
        if two_view and xa is not None and side_tr:
            xb = interp_track(side_tr[0], side_tr[1][:, j], game_t)
            out[f"{name}, 2 views"] = ("3d", m, triangulate(cams[cid["top"]], xa, cams[cid["side"]], xb))
    return out, two_view


def path_metrics(pred, game_xy, game_t, scale):
    """Metrics of one trial on the screen (game px, scale in game px per mm): RMSE, median and P90 error (mm),
    coverage (share of game samples with an estimate), length ratio (estimate / game trajectory) and speed
    correlation, the last two on every fourth sample (about 30 Hz) after Savitzky-Golay smoothing."""
    e = np.linalg.norm(pred - game_xy, axis=1) / scale
    good = ~np.isnan(e)
    if good.sum() < 10:
        return None
    k = np.arange(0, len(game_xy), 4)
    k = k[good[k]]
    if len(k) < 12:
        return None
    sg = lambda xy: savgol_filter(xy[k], 9, 2, axis=0)
    sp = lambda xy: np.linalg.norm(np.gradient(sg(xy), game_t[k], axis=0), axis=1)
    plen = lambda xy: np.linalg.norm(np.diff(sg(xy), axis=0), axis=1).sum()
    return dict(coverage=good.mean(), rmse_mm=np.sqrt(np.nanmean(e ** 2)), median_mm=np.nanmedian(e),
                p90_mm=np.nanpercentile(e, 90), path_ratio=plen(pred) / plen(game_xy),
                speed_r=np.corrcoef(sp(pred), sp(game_xy))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=None, help="comma-separated subset (testing)")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    cams = cameras()
    m = manifest()
    rows = m[m.vicon_valid & m.top_valid]                    # model outputs exist only for these clips
    if args.participants:
        rows = rows[rows.pid.isin(args.participants.split(","))]
    per = {}                                                   # row key -> (row, game_t, game_xy, sources, two_view)
    for pid, g in rows.groupby("pid"):
        gp = game_paths(pid)
        for _, r in g.iterrows():
            t, xy = gp[int(r.aligned_mat_row)]
            src, two = sources_for_path(r, cams, t)
            per[(pid, int(r.aligned_mat_row))] = (r, t, xy, src, two)
        print(f"[{pid}] {len(g)} paths", flush=True)

    # game px per mm of each visit, from the VICON fingertip -> screen affine (the iPads differ between
    # visits: 7-9.4 px/mm); visits without enough VICON samples use the median over visits
    scales, by_visit = {}, defaultdict(list)
    for k, val in per.items():
        by_visit[(k[0], val[0].visit)].append(k)
    for v, keys in by_visit.items():
        X = np.concatenate([per[k][3]["Vicon FIN"][2] for k in keys])
        y = np.concatenate([per[k][2] for k in keys])
        ok = ~np.isnan(X).any(1)
        if ok.sum() > 500:
            scales[v] = float(np.linalg.svd(fit_affine3(X[ok], y[ok])[:3].T, compute_uv=False).mean())
    sc = np.array(list(scales.values()))
    print(f"game px per mm per visit: median {np.median(sc):.3f}, IQR {np.percentile(sc, 25):.3f}-{np.percentile(sc, 75):.3f}")

    # leave one trial out: the source -> screen mapping of a trial is fitted on the other trials of the same
    # participant, visit and hand (3D sources: affine; top-view 2D sources: homography)
    recs = []
    groups = defaultdict(list)                               # (pid, visit, hand) -> keys
    for k, (r, *_rest) in per.items():
        groups[(r.pid, r.visit, r.hand)].append(k)
    for gkeys in groups.values():
        names = set().union(*[per[k][3].keys() for k in gkeys])
        for name in names:
            for k in gkeys:
                r, t, xy, src, two = per[k]
                if name not in src:
                    continue
                kind, model, est = src[name]
                others = [o for o in gkeys if o != k and name in per[o][3]]
                if not others:
                    continue
                Xf = np.concatenate([per[o][3][name][2] for o in others])
                yf = np.concatenate([per[o][2] for o in others])
                ok = ~np.isnan(Xf).any(1)
                if ok.sum() < MIN_FIT:
                    continue
                pred = np.full_like(xy, np.nan)
                good = ~np.isnan(est).any(1)
                if not good.any():
                    continue
                if kind == "3d":
                    pred[good] = apply_affine3(fit_affine3(Xf[ok], yf[ok]), est[good])
                else:
                    H, _ = cv2.findHomography(Xf[ok], yf[ok], cv2.LMEDS)   # least median of squares, no threshold
                    if H is None:
                        continue
                    pred[good] = cv2.perspectiveTransform(est[good][None].astype(np.float64), H)[0]
                scale = scales.get((r.pid, r.visit), float(np.median(sc)))
                mt = path_metrics(pred, xy, t, scale)
                if mt is None:
                    continue
                recs.append(dict(pid=r.pid, row=int(r.aligned_mat_row), visit=r.visit, hand=r.hand,
                                 group="UCP" if r.motor_impaired else "TD", source=name, kind=kind,
                                 mapping="affine" if kind == "3d" else "homography", two_view_valid=two,
                                 game_px_per_mm=scale, **mt))
    paths = pd.DataFrame(recs)
    # trajectory QA: visits above QA_MAX_MM (broken game-VICON alignment) go to QA_FILE (benchlib.manifest drops them)
    qa = (paths[paths.source == "Vicon FIN"].groupby(["pid", "visit"]).rmse_mm
          .agg(vicon_trajectory_rmse_mm="median", n_paths="size").reset_index())
    qa["failed"] = qa.vicon_trajectory_rmse_mm > QA_MAX_MM
    if not args.participants:
        qa.to_csv(QA_FILE, index=False, float_format="%.3f")
    bad = set(map(tuple, qa.loc[qa.failed, ["pid", "visit"]].values))
    print(f"trajectory QA: {len(bad)} visit(s) failed (VICON vs game trajectory > {QA_MAX_MM} mm): {sorted(bad)}")
    paths = paths[[(p, v) not in bad for p, v in zip(paths.pid, paths.visit)]].copy()
    # subsets: 'own' every trial of a source, 'common' the trials with an estimate from every source
    n_src = paths.groupby(["pid", "row"]).source.transform("nunique")
    paths["common"] = n_src == paths.source.nunique()
    paths.to_csv(RES / f"E3_screen_paths{args.out_suffix}.csv", index=False, float_format="%.4f")

    metrics = ["rmse_mm", "median_mm", "p90_mm", "coverage", "path_ratio", "speed_r"]
    part = []
    for subset, q in (("own", paths), ("common", paths[paths.common])):
        p = q.groupby(["pid", "group", "source", "mapping"])[metrics].median().reset_index()
        p["n_paths"] = q.groupby(["pid", "group", "source", "mapping"]).size().values
        p["subset"] = subset
        part.append(p)
    part = pd.concat(part)
    part.to_csv(RES / f"E3_screen_participants{args.out_suffix}.csv", index=False, float_format="%.4f")

    test = set(splits()["test"])
    rec = []
    for split, ps in (("test", part[part.pid.isin(test)]), ("all", part)):
        for group in ("all", "TD", "UCP"):
            q = ps if group == "all" else ps[ps.group == group]
            for (subset, source, mapping), g in q.groupby(["subset", "source", "mapping"]):
                lo, hi = boot_ci(g.rmse_mm)
                rec.append(dict(split=split, group=group, subset=subset, source=source, mapping=mapping,
                                n_participants=g.pid.nunique(), n_paths=int(g.n_paths.sum()),
                                rmse_ci_lo=lo, rmse_ci_hi=hi, **g[metrics].mean().to_dict()))
    s = pd.DataFrame(rec)
    s.to_csv(RES / f"E3_screen{args.out_suffix}.csv", index=False, float_format="%.4f")
    for split in ("test", "all"):
        q = s[(s.split == split) & (s.group == "all") & (s.subset == "own")].sort_values("rmse_mm")
        print(f"\n{split} split, all paths of each source (mean over participants of the per-path median):")
        print(q[["source", "mapping", "n_participants", "n_paths", "rmse_mm", "median_mm", "p90_mm", "coverage",
                 "path_ratio", "speed_r"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
