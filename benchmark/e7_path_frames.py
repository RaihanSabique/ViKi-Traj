"""E7 qualitative figure of one test trial: VICON and both videos at three moments, with the errors of one model.

By default the lowest-error trial of the TD test participant with the lowest 3D fingertip and wrist error (choose).
Rows: VICON 3D with the model's two-view triangulation, then the top and side frames with the projected VICON joints
and the keypoints (faces blurred in the side view). Writes outputs/benchmark/reports/E7/fig_E7_path_frames.{png,pdf}
and table_E7_path_frames.csv.

    python benchmark/e7_path_frames.py [--model ft_b_full_pooled] [--pid XXXXXXX110] [--row N]
"""
import argparse

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402

from benchlib import BENCH, JOINT_NAMES, MODEL_LABEL, clip_stem, frame_clock, manifest, markers, model_file, set_lr_swap, \
    splits                                                               # noqa: E402
from calibrate import camera_id                                          # noqa: E402
from e4_kinematics import angles                                         # noqa: E402
from e7_qualitative import HEAD, frame_at                                # noqa: E402
from evaluate import RES, at, cameras, model_tracks, ref_track           # noqa: E402
from trial_overlays import FaceBlur                                      # noqa: E402
from vicon_eval import project, px_per_mm                                # noqa: E402

OUT = BENCH / "reports" / "E7"
MOMENTS = (0.15, 0.50, 0.85)                                            # fractions of the movement window
TCOL = ["#2a78d6", "#eb6834", "#1baf7a"]                                # one colour per moment
INK, INK2 = "#0b0b0b", "#52514e"
J = {j: i for i, j in enumerate(JOINT_NAMES)}                           # shoulder, elbow, wrist, hand, fingertip
ARM = ["shoulder", "elbow", "wrist", "hand", "fingertip"]
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": INK2})


def choose(model, pid=None, row=None):
    """(manifest row, per-participant error, trial errors) of the trial to show: unless given, the TD test
    participant with the lowest mean of its 3D fingertip and wrist errors and its trial with the lowest mean."""
    p = pd.read_csv(RES / "E2_3d_paths.csv")
    m = manifest()
    td = set(m[~m.motor_impaired.astype(bool)].pid)
    q = p[(p.variant == model) & (p.role == "active") & p.joint.isin(["wrist", "fingertip"]) & p.pid.isin(splits()["test"])
          & p.pid.isin(td)]
    per = q.groupby(["pid", "joint"]).median_mm.median().unstack().mean(1)
    pid = pid or per.idxmin()
    pp = q[q.pid == pid].pivot_table(index="row", columns="joint", values="median_mm").dropna()
    row = int(row if row is not None else pp.mean(1).idxmin())
    return m[(m.pid == pid) & (m.aligned_mat_row == row)].iloc[0], per, pp.loc[row]


def angle2d(a, b, c):
    """Angle ABC (deg) of 2D points."""
    u, w = a - b, c - b
    return np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1)))


def panel_text(e, two_d):
    a = " (2-D)" if two_d else ""
    return (f"Fingertip {e['fingertip']:.1f} mm\nWrist {e['wrist']:.1f} mm\n"
            f"Elbow{a} {e['elbow']:.1f}°\nShoulder{a} {e['shoulder']:.1f}°")


def free_corner(pts, shape):
    """Axes-fraction anchor of the corner farthest from the drawn keypoints (pts in crop px)."""
    u = pts[~np.isnan(pts).any(1)] / [shape[1], shape[0]]
    u[:, 1] = 1 - u[:, 1]
    best = max(((0.02, 0.97), (0.98, 0.97), (0.02, 0.03), (0.98, 0.03)),
               key=lambda c: np.min(np.linalg.norm(u - c, axis=1)) if len(u) else 0)
    return best


def draw_streams(fig, gs, r, model, cams, blur, rows=(0, 1, 2), fs=1.0, label="VICON", view3d=(22, -58, 0), errors3d=True, zoom3d=1.0, nticks3d=None,
                 labelpad3d=-6, moments=MOMENTS, colors=TCOL, cell=None, row_labels=True, where=("row 1", "rows 2-3"),
                 vicon_col=None, traj_label=None):
    """Three rows (VICON 3D, top video, side video) x the moments into the gridspec rows `rows`; returns (records,
    legend handles, axes). view3d is (elev, azim, roll); cell(kind, c) returns the SubplotSpec of panel kind 0-2 at
    moment c; vicon_col colours the VICON arm and gives the model arm the moment colour. Used by make_fig_dataset.py."""
    pid, row, side = r.pid, int(r.aligned_mat_row), str(r.hand)[0]
    v = markers(pid)[0][row]
    tv = v["t"]
    times = tv[0] + np.array(moments) * (tv[-1] - tv[0])
    cell = cell or (lambda kind, c: gs[rows[kind], c])
    ref = {j: ref_track(v, side, j) for j in ARM}
    tr = np.load(BENCH / "tracks3d" / f"{pid}_r{row}.npz")
    t3, vic3, mod3 = tr["t"], tr["vicon"], tr[model]                    # (4, N, 3) in e4_kinematics.J4 order
    rec, axes = [], {}
    F = lambda x: x * fs                                                 # noqa: E731
    box_kw = dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9)
    # ---------------------------------------------------------------- row 1: VICON 3D
    tip = np.array([at(tv, ref["fingertip"], [t])[0] for t in times])
    allpts = np.concatenate([np.array([at(tv, ref[j], [t])[0] for j in ARM for t in times]), ref["fingertip"]])
    allpts = allpts[~np.isnan(allpts).any(1)]
    lo, hi = allpts.min(0) - 60, allpts.max(0) + 60
    for c, (t, col) in enumerate(zip(times, colors)):
        ax = fig.add_subplot(cell(0, c), projection="3d")
        ax.patch.set_alpha(0)                                              # tick labels may overlap the next cell
        axes[(0, c)] = ax
        hx = int(np.clip(np.searchsorted(tv, t), 0, len(tv) - 1))
        mk = v["X"][hx]
        mk = mk[~np.isnan(mk).any(1)]
        inb = np.all((mk > lo - 250) & (mk < hi + 250), 1)
        ax.scatter(*mk[inb].T, s=F(5), color="#a9a8a3", depthshade=False)
        ax.plot(*ref["fingertip"].T, color="#c9c8c3", lw=F(1))
        vc, mc = (vicon_col, col) if vicon_col else (col, INK)                # VICON arm / model arm colours
        if len(times) > 1:
            for cc, (pt, tc) in enumerate(zip(tip, colors)):
                ax.scatter(*pt, s=F(18 if cc != c else 34), color=tc, edgecolor="white", linewidth=0.5, depthshade=False, zorder=4)
        sk = np.array([at(tv, ref[j], [t])[0] for j in ARM])
        ax.plot(*sk.T, color=vc, lw=F(2.6), solid_capstyle="round")
        ax.scatter(*sk.T, s=F(26), color=vc, edgecolor="white", linewidth=0.6, depthshade=False)
        i3 = int(np.argmin(np.abs(t3 - t)))
        X = mod3[:, i3]
        ax.plot(*X.T, color=mc, lw=F(1.6 if vicon_col else 1.4), ls=(0, (3, 2)))
        ax.scatter(*X.T, s=F(22), facecolor="none", edgecolor=mc, linewidth=1.0, depthshade=False)
        av, am = angles(vic3[:, i3:i3 + 1]), angles(mod3[:, i3:i3 + 1])
        e = dict(fingertip=np.linalg.norm(X[3] - vic3[3, i3]), wrist=np.linalg.norm(X[2] - vic3[2, i3]),
                 elbow=abs(am["elbow_flexion"][0] - av["elbow_flexion"][0]),
                 shoulder=abs(am["shoulder_elevation"][0] - av["shoulder_elevation"][0]))
        rec.append(dict(row_kind=f"{label} 3-D", moment=moments[c], t_s=t - tv[0], **e))
        if errors3d:
            ax.text2D(0.99, 0.95, panel_text(e, False), transform=ax.transAxes, va="top", ha="right", fontsize=F(7.5), color=INK,
                      bbox=dict(box_kw, ec=col, lw=1.2))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        try:
            ax.set_box_aspect(hi - lo, zoom=zoom3d)
        except TypeError:
            ax.set_box_aspect(hi - lo)
        ax.view_init(elev=view3d[0], azim=view3d[1], roll=view3d[2])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.fill = False
            a.pane.set_edgecolor("#e4e3df")
            a.set_tick_params(labelsize=F(6), colors=INK2, pad=0)
            if nticks3d:
                a.set_major_locator(MaxNLocator(nticks3d))
        ax.set_xlabel("x (mm)", fontsize=F(6.5), labelpad=labelpad3d)
        ax.set_ylabel("y (mm)", fontsize=F(6.5), labelpad=labelpad3d)
        ax.set_zlabel("z (mm)", fontsize=F(6.5), labelpad=labelpad3d)
        if len(times) > 1:
            ax.set_title(f"Moment {c + 1}: t = {t - tv[0]:.1f} s ({int(moments[c] * 100)} % of the trial)", fontsize=F(9), color=INK, pad=2)
    # ---------------------------------------------------------------- rows 2-3: videos
    for rr, view in ((1, "top"), (2, "side")):
        video = r[f"{view}_source_video"]
        P = cams[camera_id(pid, int(r.visit), view, video)]
        z = dict(np.load(model_file(model, pid, view, clip_stem(r, view))))
        set_lr_swap({model: z}, pid, view, video)
        k2 = model_tracks(z, model, side, view, "auto")
        ft = frame_clock(pid, row, view, z["frame"])
        idx = [int(np.argmin(np.abs(ft - t))) for t in times]
        face = None
        if view == "side":
            fw = model_file("rtmpose_wb", pid, view, clip_stem(r, view))
            face = dict(np.load(fw)) if fw.exists() else None
        box = np.load(BENCH / "boxes" / pid / view / f"{clip_stem(r, view)}.npz")
        bb = np.array([box["box"][int(np.searchsorted(box["frame"], z["frame"][i]))] for i in idx])
        x0, y0, x1, y1 = bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()
        w, h = (x1 - x0) * 1.2, (y1 - y0) * 1.12
        w, h = max(w, h * 4 / 3), max(h, w * 3 / 4)                     # 4:3 crop, the same for the three moments
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for c, (i, col) in enumerate(zip(idx, colors)):
            f, tf = int(z["frame"][i]), ft[i]
            img = frame_at(video, f)
            g3 = np.array([at(tv, ref[j], [tf])[0] for j in ARM])
            g2 = project(P, np.nan_to_num(g3))
            g2[np.isnan(g3).any(1)] = np.nan
            scale = px_per_mm(P, np.nan_to_num(g3))
            k = k2[i]
            down = project(P, g3[J["shoulder"]][None] - [0, 0, 200.0])[0] - g2[J["shoulder"]]   # VICON -z in the image
            down /= np.linalg.norm(down)

            def elev(p):
                u = p[J["elbow"]] - p[J["shoulder"]]
                return np.degrees(np.arccos(np.clip(u @ down / np.linalg.norm(u), -1, 1)))
            e = dict(fingertip=np.linalg.norm(k[J["fingertip"]] - g2[J["fingertip"]]) / scale[J["fingertip"]],
                     wrist=np.linalg.norm(k[J["wrist"]] - g2[J["wrist"]]) / scale[J["wrist"]],
                     elbow=abs(angle2d(*k[[J["shoulder"], J["elbow"], J["wrist"]]]) - angle2d(*g2[[J["shoulder"], J["elbow"], J["wrist"]]])),
                     shoulder=abs(elev(k) - elev(g2)))
            rec.append(dict(row_kind=f"{view} video 2-D", moment=moments[c], t_s=tf - tv[0], frame=f, **e))
            if view == "side":
                heads = [v["labels"].index(hd) for hd in HEAD if hd in v["labels"]]
                hx = int(np.clip(np.searchsorted(tv, tf), 0, len(tv) - 1))
                hp = project(P, v["X"][hx, heads]) if heads else None
                fk = None
                if face is not None:
                    fi = int(np.searchsorted(face["frame"], f))
                    if fi < len(face["frame"]) and face["frame"][fi] == f:
                        fk = face["k"][fi, :5][face["s"][fi, :5] > 0.3]          # nose, eyes, ears
                img = blur.apply(img, blur.regions(img, hp, fk))
            xa, ya = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0))
            xb, yb = int(min(cx + w / 2, img.shape[1])), int(min(cy + h / 2, img.shape[0]))
            ax = fig.add_subplot(cell(rr, c))
            axes[(rr, c)] = ax
            ax.imshow(img[ya:yb, xa:xb, ::-1])
            G, K = g2 - [xa, ya], k - [xa, ya]
            ax.plot(G[:, 0], G[:, 1], color="white", lw=F(1.0), alpha=0.9)
            ax.plot(K[:, 0], K[:, 1], color=col, lw=F(2.2), solid_capstyle="round")
            ax.scatter(K[:, 0], K[:, 1], s=F(28), color=col, edgecolor="white", linewidth=0.7, zorder=3)
            ax.scatter(G[:, 0], G[:, 1], s=F(80), facecolor="none", edgecolor="white", linewidth=1.8, zorder=4)
            ax.scatter(G[:, 0], G[:, 1], s=F(80), facecolor="none", edgecolor=INK, linewidth=0.5, zorder=4)
            cx_, cy_ = free_corner(np.concatenate([G, K]), img[ya:yb, xa:xb].shape)
            ax.text(cx_, cy_, panel_text(e, True), transform=ax.transAxes, va="top" if cy_ > 0.5 else "bottom",
                    ha="left" if cx_ < 0.5 else "right", fontsize=F(7.5), color=INK, bbox=dict(box_kw, ec=col, lw=1.2))
            ax.set_axis_off()
    if row_labels:
        for rr, lab in enumerate((f"{label} (3-D)", "Top video (2-D)", "Side video (2-D)")):
            bb = axes[(rr, 0)].get_position()
            fig.text(bb.x0 - 0.022, (bb.y0 + bb.y1) / 2, lab, rotation=90, va="center", ha="center", fontsize=F(10), color=INK)
    one = colors[0] if len(times) == 1 else INK2                          # one moment: the model's colour in the legend
    hs = [Line2D([], [], color=vicon_col or INK2, lw=F(2.6), marker="o", ms=F(5), mec="white", label=f"{label} joints (3-D" + (f", {where[0]})" if where[0] else ")")),
          Line2D([], [], color=one if vicon_col else INK, lw=F(1.4), ls=(0, (3, 2)), marker="o", mfc="none", ms=F(5),
                 label=f"{MODEL_LABEL[model]}, two-view 3-D"),
          Line2D([], [], color="#c9c8c3", lw=F(1), label=traj_label or f"fingertip trajectory ({label})"),
          Line2D([], [], ls="", marker="o", ms=F(9), mfc="none", mec=INK, mew=1.2, label=f"{label} joints projected" + (f" ({where[1]})" if where[1] else "")),
          Line2D([], [], color=one, lw=F(2.2), marker="o", ms=F(5), mec="white", label=f"{MODEL_LABEL[model]} keypoints" + (f" ({where[1]})" if where[1] else ""))]
    if len(times) > 1:
        hs += [Line2D([], [], ls="", marker="s", ms=F(8), color=col, label=f"moment {c + 1}") for c, col in enumerate(colors)]
    return rec, hs, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ft_b_full_pooled")
    ap.add_argument("--pid")
    ap.add_argument("--row", type=int)
    args = ap.parse_args()
    model = args.model
    r, per, trial_err = choose(model, args.pid, args.row)
    pid, row = r.pid, int(r.aligned_mat_row)
    print("TD test children, mean 3-D fingertip / wrist error (mm):", per.round(2).to_dict())
    print(f"chosen {pid} row {row} ({r.hand} hand, visit {r.visit}, trial {r.path}); trial medians {trial_err.round(2).to_dict()}")
    v = markers(pid)[0][row]
    fig = plt.figure(figsize=(11.0, 10.0))
    gs = fig.add_gridspec(3, 3, hspace=0.08, wspace=0.04, left=0.05, right=0.99, top=0.92, bottom=0.075, height_ratios=(1.08, 1, 1))
    rec, hs, _ = draw_streams(fig, gs, r, model, cameras(), FaceBlur())
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.52, 0.0))
    fig.suptitle(f"{pid} (TD test child with the lowest error), {r.hand.lower()} hand, visit {r.visit}, trial {r.path} "
                 f"({v['t'][-1] - v['t'][0]:.1f} s movement trial)\nVICON vs {MODEL_LABEL[model]} at three moments of the trial",
                 fontsize=10, x=0.02, ha="left", y=0.995)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_E7_path_frames.{ext}", dpi=200)
    t = pd.DataFrame(rec).assign(pid=pid, row=row, model=model)
    t.to_csv(OUT / "table_E7_path_frames.csv", index=False, float_format="%.3f")
    print(t.round(2).to_string())
    print("wrote", OUT / "fig_E7_path_frames.png")


if __name__ == "__main__":
    main()
