"""E7 qualitative figure: projected VICON joints and model keypoints on test frames, with two failure cases.

Tiles: a TD and a UCP test trial at the peak VICON fingertip speed (top and side view), the top-view frame with the
largest RTMPose-WB fingertip error, and BlazePose under the fixed top-view side swap. White rings are the low-passed
VICON joints projected into the view, dots the unfiltered keypoints of the active arm; heads are blurred
(src/trial_overlays.FaceBlur). Writes outputs/benchmark/reports/E7/qualitative.{png,pdf}.

    python benchmark/e7_qualitative.py
"""
import argparse

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

from benchlib import BENCH, JOINT_NAMES, ROOT, clip_stem, frame_clock, manifest, markers, model_file, model_joints, \
    set_lr_swap, splits                                                  # noqa: E402
from calibrate import camera_id                                          # noqa: E402
from evaluate import at, cameras, ref_track                              # noqa: E402
from trial_overlays import FaceBlur                                      # noqa: E402
from vicon_eval import project                                           # noqa: E402

OUT = BENCH / "reports" / "E7"
COL = {"sapiens2_1b": "#eb6834", "rtmpose_wb": "#2a78d6", "blazepose": "#6250d6"}      # one colour per model
LAB = {"sapiens2_1b": "Sapiens2-1B", "rtmpose_wb": "RTMPose-WB", "blazepose": "BlazePose"}
HEAD = ("LFHD", "RFHD", "LBHD", "RBHD")                                  # VICON head markers


def frame_at(video, f):
    """Frame f (BGR) of a source video, path relative to the repository root."""
    cap = cv2.VideoCapture(str(ROOT / video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    cap.release()
    return img


def tile(r, view, f, models, cams, blur, mirror="auto"):
    """Face-blurred RGB crop of frame f around the participant, with the projected VICON joints (5, 2) and
    {model: keypoints (5, 2)} of the active arm in crop px."""
    pid, row, side = r.pid, int(r.aligned_mat_row), str(r.hand)[0]
    video = r[f"{view}_source_video"]
    P = cams[camera_id(pid, int(r.visit), view, video)]
    img = frame_at(video, f)
    v = markers(pid)[0][row]
    t = frame_clock(pid, row, view, np.array([f]))
    truth = np.array([at(v["t"], ref_track(v, side, j), t)[0] for j in JOINT_NAMES])
    ok = ~np.isnan(truth).any(1)
    gt = np.full((5, 2), np.nan)
    gt[ok] = project(P, truth[ok])
    heads = [v["labels"].index(h) for h in HEAD if h in v["labels"]]
    hx = np.searchsorted(v["t"], t[0]).clip(0, len(v["t"]) - 1)
    hp = project(P, v["X"][hx, heads]) if heads else None
    kps = {}
    face = []
    for m in models:
        z = dict(np.load(model_file(m, pid, view, clip_stem(r, view))))
        set_lr_swap({m: z}, pid, view, video)
        i = int(np.searchsorted(z["frame"], f))
        # every per-frame array cut to this one frame
        kps[m] = model_joints({k: (a[i:i + 1] if hasattr(a, "ndim") and a.ndim and len(a) == len(z["frame"]) else a)
                               for k, a in z.items()}, m, side, view, mirror)[0]
        if m == "rtmpose_wb":
            face.append(z["k"][i, :5][z["s"][i, :5] > 0.3])             # nose, eyes, ears
    img = blur.apply(img, blur.regions(img, hp, np.concatenate(face) if face else None))
    box = np.load(BENCH / "boxes" / pid / view / f"{clip_stem(r, view)}.npz")
    b = box["box"][int(np.searchsorted(box["frame"], f))]
    cx, cy, s = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 0.65
    x0, y0 = int(max(cx - s, 0)), int(max(cy - s, 0))
    x1, y1 = int(min(cx + s, img.shape[1])), int(min(cy + s, img.shape[0]))
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), gt - [x0, y0], {m: k - [x0, y0] for m, k in kps.items()}


def draw(ax, im, gt, kps, title):
    ax.imshow(im)
    for m, k in kps.items():
        ax.plot(k[:, 0], k[:, 1], "-", color=COL[m], lw=2, alpha=0.9)
        ax.scatter(k[:, 0], k[:, 1], s=36, color=COL[m], edgecolor="white", linewidth=0.8, zorder=3, label=LAB[m])
    ax.scatter(gt[:, 0], gt[:, 1], s=90, facecolor="none", edgecolor="white", linewidth=2, zorder=4)
    ax.scatter(gt[:, 0], gt[:, 1], s=90, facecolor="none", edgecolor="black", linewidth=0.6, zorder=4, label="VICON (projected)")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def peak_frame(r, view):
    """Source-video frame of the movement window nearest the peak VICON fingertip speed of the trial."""
    v = markers(r.pid)[0][int(r.aligned_mat_row)]
    x = ref_track(v, str(r.hand)[0], "fingertip")
    sp = np.linalg.norm(np.gradient(x, v["t"], axis=0), axis=1)
    tp = v["t"][int(np.nanargmax(sp))]
    s0, n0 = int(r[f"{view}_start_frame"]), int(r[f"{view}_n_frames"])
    fr = np.arange(s0, s0 + n0)
    return int(fr[np.argmin(np.abs(frame_clock(r.pid, int(r.aligned_mat_row), view, fr) - tp))])


def worst_frame(r, view, model, cams):
    """Frame of the clip with the largest image distance between the model fingertip and the projected
    VICON fingertip (active arm)."""
    pid, row, side = r.pid, int(r.aligned_mat_row), str(r.hand)[0]
    video = r[f"{view}_source_video"]
    P = cams[camera_id(pid, int(r.visit), view, video)]
    z = dict(np.load(model_file(model, pid, view, clip_stem(r, view))))
    set_lr_swap({model: z}, pid, view, video)
    k = model_joints(z, model, side, view, "auto")[:, JOINT_NAMES.index("fingertip")]
    v = markers(pid)[0][row]
    s0, n0 = int(r[f"{view}_start_frame"]), int(r[f"{view}_n_frames"])
    inside = (z["frame"] >= s0) & (z["frame"] < s0 + n0)
    truth = at(v["t"], ref_track(v, side, "fingertip"), frame_clock(pid, row, view, z["frame"]))
    e = np.full(len(k), np.nan)
    ok = inside & ~np.isnan(truth).any(1) & ~np.isnan(k).any(1)
    e[ok] = np.linalg.norm(project(P, truth[ok]) - k[ok], axis=1)
    return int(z["frame"][np.nanargmax(e)])


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cams = cameras()
    blur = FaceBlur()
    m = manifest()
    test = splits()["test"]
    ok = m[m.vicon_valid & m.top_valid & m.side_valid & m.pid.isin(test)]
    # the third valid trial of one TD and one UCP test participant
    td = ok[(~ok.motor_impaired) & (ok.pid == "XXXXXXX110")].iloc[2]
    ucp = ok[ok.motor_impaired & (ok.pid == "XXXXXXX203")].iloc[2]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for c, (r, name) in enumerate(((td, "TD"), (ucp, "UCP"))):
        for rr, view in enumerate(("top", "side")):
            im, gt, kps = tile(r, view, peak_frame(r, view), ["sapiens2_1b", "rtmpose_wb"], cams, blur)
            draw(axes[rr, c], im, gt, kps, f"{name} child, {view} view (peak fingertip speed)")
    # failure 1: top-view frame with the largest RTMPose-WB fingertip error
    p = __import__("pandas").read_csv(BENCH / "results" / "E1_2d_paths.csv")
    q = p[(p.variant == "rtmpose_wb") & (p.view == "top") & (p.joint == "fingertip") & (p.role == "active") & (p.labels == "all")
          & p.pid.isin(test)].sort_values("median_mm", ascending=False).iloc[0]
    r = m[(m.pid == q.pid) & (m.aligned_mat_row == q.row)].iloc[0]
    im, gt, kps = tile(r, "top", worst_frame(r, "top", "rtmpose_wb", cams), ["rtmpose_wb", "sapiens2_1b"], cams, blur)
    draw(axes[0, 2], im, gt, kps, "Failure: largest RTMPose-WB fingertip error (top)")
    # failure 2: BlazePose with the fixed top-view swap on a visit filmed from the front
    im, gt, kps = tile(td, "top", peak_frame(td, "top"), ["blazepose"], cams, blur, mirror="view")
    draw(axes[1, 2], im, gt, kps, "Failure: BlazePose sides under a fixed swap")
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 2].get_legend_handles_labels()
    fig.legend(h[:2] + h2[:1] + h[-1:], l[:2] + l2[:1] + l[-1:], loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"qualitative.{ext}", dpi=200)
    print("wrote", OUT / "qualitative.png")


if __name__ == "__main__":
    main()
