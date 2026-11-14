"""Demo videos of one trial: projected VICON joints and model keypoints over the released clips.

Reads only the released dataset. Writes demo_best_model.mp4 (Sapiens2-1B + ViconTraj-MS on the top and side views,
with the trial's median 3D error per joint from tracks3d) and demo_all_models.mp4 (the 8 models zero-shot and with
ViconTraj-MS on one view), each with a GIF copy.

    python scripts/render_demo.py [--root release/ViKi-Traj] [--trial sub-03 ses-1 run-06] [--view top]
        [--out docs/media]
"""
import argparse
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DERIV = "derivatives/vikitraj"
JOINTS = ("shoulder", "elbow", "wrist", "hand", "fingertip")         # drawn as one chain in this order
JOINTS_3D = ("shoulder", "elbow", "wrist", "fingertip")             # joints of tracks3d
BEST = ("ft_sapiens2_1b_head", "Sapiens2-1B + ViconTraj-MS", "#008300")
MODELS = [  # zero-shot output, name, colour (MCOL of paper/make_results.py), ViconTraj-MS output
    ("vitpose_b", "ViTPose+-B", "#2a78d6", "ft_b_full_pooled_traj"),
    ("rtmpose_wb", "RTMPose-WB", "#eb6834", "seq_rtmpose_wb"),
    ("rtmpose_hand", "RTMPose-m hand", "#8c5a2b", "seq_rtmpose_hand"),
    ("dwpose_l", "DWPose-L", "#eda100", "seq_dwpose_l"),
    ("rtmw_x", "RTMW-X", "#6250d6", "seq_rtmw_x"),
    ("sapiens2_0.4b", "Sapiens2-0.4B", "#1baf7a", "ft_sapiens2_0.4b_head"),
    ("sapiens2_1b", "Sapiens2-1B", "#008300", "ft_sapiens2_1b_head"),
    ("blazepose", "BlazePose+Hands", "#e87ba4", "seq_blazepose"),
]
PANEL = {"top": (540, 720), "side": (960, 720)}       # px of a view in demo_best_model
TILE = {"top": (352, 470), "side": (470, 352)}        # px of a tile in demo_all_models
INK, INK2, WHITE = (11, 11, 11), (82, 81, 78), (255, 255, 255)
GAP, TITLE_H, LEGEND_H = 8, 34, 46
TRAIL_S = 2.0               # fingertip trail, s: long enough to show beside the dot at the drag speed (about 35 mm/s)
LIGHT = 0.3                 # zero-shot colour: share of white mixed into the model colour
CRF = 20                    # x264 quality
GIF_FPS, GIF_BOX, GIF_COLOURS = 10, (960, 540), 96          # GIF frame fits in GIF_BOX px
SUB = 16                    # cv2 drawing with 4 fractional bits (sub-pixel positions)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", na_values="n/a", keep_default_na=False)


def trial(root, sub, ses, run):
    m = read_tsv(root / DERIV / "manifest.tsv")
    row = m[(m.participant_id == sub) & (m.session == ses) & (m.run == run)]
    if row.empty:
        raise SystemExit(f"{sub} {ses} {run} is not in the manifest")
    return row.iloc[0]


def frame_table(root, rel, n):
    """A labels2d or keypoints2d table indexed by clip frame 0 .. n-1 (NaN rows for frames without an image)."""
    return read_tsv(root / rel).set_index("frame").reindex(range(n))


def joints_xy(table, side):
    """(n, 5, 2) image px of the arm's joints, NaN where missing."""
    return np.stack([table[[f"{side}_{j}_x", f"{side}_{j}_y"]].to_numpy(float) for j in JOINTS], 1)


def thresholds(root, model, side):
    """Score threshold per joint as the sidecar states it ('... only where score >= 0.3'); None = every position."""
    meta = json.loads((root / DERIV / "keypoints2d" / model / "task-reaching_keypoints.json").read_text())
    found = [re.search(r"score >= (\d+(?:\.\d+)?)", meta[f"{side}_{j}_score"]["Description"]) for j in JOINTS]
    return [float(f.group(1)) if f else None for f in found]


def model_xy(root, row, view, model, n):
    """Keypoints of the active arm, NaN where n/a, below the score threshold or without output for the clip."""
    rel = row[f"{view}_labels"].replace("labels2d/", f"keypoints2d/{model}/").replace("_labels.tsv", "_keypoints.tsv")
    if not (root / rel).exists():
        return np.full((n, len(JOINTS), 2), np.nan)
    table = frame_table(root, rel, n)
    xy = joints_xy(table, row.hand)
    for i, (j, thr) in enumerate(zip(JOINTS, thresholds(root, model, row.hand))):
        if thr is not None:
            xy[~(table[f"{row.hand}_{j}_score"].to_numpy(float) >= thr), i] = np.nan
    return xy


def median_error(root, row, source):
    """Median distance (mm) between the source's two-view 3D joints and VICON, per joint of tracks3d."""
    sub, ses, run = row.participant_id, row.session, row.run
    t = read_tsv(root / DERIV / "tracks3d" / sub / ses / f"{sub}_{ses}_task-reaching_{run}_tracks3d.tsv")
    pos = {s: g.set_index(["time", "joint"])[["x", "y", "z"]] for s, g in t.groupby("source")}
    d = (pos[source] - pos["vicon"]).dropna()
    err = pd.Series(np.linalg.norm(d.to_numpy(), axis=1), index=d.index.get_level_values("joint"))
    return err.groupby(level=0).median().reindex(JOINTS_3D)


# ----------------------------------------------------------------------------- frames and crops
def clip_frames(path, n):
    """The first n images of a clip as RGB arrays (the last image repeats if the clip is shorter)."""
    cap = cv2.VideoCapture(str(path))
    img = None
    for _ in range(n):
        ok, bgr = cap.read()
        if ok:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yield img
    cap.release()


def image_shape(path):
    cap = cv2.VideoCapture(str(path))
    shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return shape


def crop_box(xy, size, shape, margin=0.15):
    """Fixed crop (x0, y0, w, h) around all points of the trial plus a margin, with the aspect ratio of size."""
    pts = xy.reshape(-1, 2)
    lo, hi = np.nanmin(pts, 0), np.nanmax(pts, 0)
    w, h = (hi - lo) * (1 + 2 * margin)
    aspect = size[0] / size[1]
    w, h = max(w, h * aspect), max(h, w / aspect)
    s = min(1.0, shape[1] / w, shape[0] / h)
    w, h = w * s, h * s
    x0 = np.clip((lo[0] + hi[0] - w) / 2, 0, shape[1] - w)
    y0 = np.clip((lo[1] + hi[1] - h) / 2, 0, shape[0] - h)
    return x0, y0, w, h


def crop(img, box, size):
    x0, y0, w, h = (int(round(v)) for v in box)
    return cv2.resize(img[y0:y0 + h, x0:x0 + w], size, interpolation=cv2.INTER_AREA)


def to_panel(xy, box, size):
    return (xy - np.array(box[:2])) * (size[0] / box[2])


# ----------------------------------------------------------------------------- drawing
def rgb(hex_colour, light=0.0):
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    return tuple(int(v) for v in c + (255 - c) * light)


def ip(p):
    """A point in the fixed-point coordinates of cv2 drawing (shift 4)."""
    return int(round(p[0] * SUB)), int(round(p[1] * SUB))


def line(img, p, q, colour, width, dash=0.0):
    if not dash:
        cv2.line(img, ip(p), ip(q), colour, width, cv2.LINE_AA, 4)
        return
    n = max(1, int(np.hypot(*(q - p)) / dash))
    for i in range(0, n, 2):
        cv2.line(img, ip(p + (q - p) * i / n), ip(p + (q - p) * (i + 1) / n), colour, width, cv2.LINE_AA, 4)


def rings(img, xy, r=9):
    """VICON joints: white rings with a dark outline."""
    for p in xy[~np.isnan(xy).any(1)]:
        cv2.circle(img, ip(p), r * SUB, INK, 4, cv2.LINE_AA, 4)
        cv2.circle(img, ip(p), r * SUB, WHITE, 2, cv2.LINE_AA, 4)


def chain(img, xy, colour, solid=True):
    """Model joints joined in JOINTS order: solid with filled dots, or dashed with hollow dots."""
    ok = ~np.isnan(xy).any(1)
    for a in range(len(xy) - 1):
        if ok[a] and ok[a + 1]:
            line(img, xy[a], xy[a + 1], colour, 3 if solid else 2, dash=0.0 if solid else 6.0)
    for p in xy[ok]:
        if solid:
            cv2.circle(img, ip(p), 6 * SUB, WHITE, -1, cv2.LINE_AA, 4)
            cv2.circle(img, ip(p), 5 * SUB, colour, -1, cv2.LINE_AA, 4)
        else:
            cv2.circle(img, ip(p), 4 * SUB, colour, 2, cv2.LINE_AA, 4)


def trail(img, pts, colour, width=4):
    """Polyline of the last positions (oldest first), fading out with age."""
    alpha = np.zeros(img.shape[:2], np.uint8)
    for i in range(1, len(pts)):
        if not np.isnan(pts[i - 1:i + 1]).any():
            cv2.line(alpha, ip(pts[i - 1]), ip(pts[i]), int(230 * i / len(pts)), width, cv2.LINE_AA, 4)
    a = alpha[..., None] / 255.0
    img[:] = (img * (1 - a) + np.array(colour) * a).astype(np.uint8)


@lru_cache(None)
def font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def label(draw, xy, text, size=19):
    """White text on a translucent dark box, top-left corner at xy."""
    x, y = xy
    _, _, r, b = draw.textbbox((x + 8, y + 5), text, font=font(size))
    draw.rounded_rectangle((x, y, r + 8, b + 7), 4, fill=(0, 0, 0, 150))
    draw.text((x + 8, y + 5), text, font=font(size), fill=WHITE)


def error_box(errors, colour, size=18):
    """RGBA image of a box listing the median 3D error per joint."""
    title = "Median 3D error, this trial"
    w, h = int(font(size - 2).getlength(title)) + 26, 36 + (size + 7) * len(errors)
    img = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), 6, fill=(255, 255, 255, 230), outline=colour, width=2)
    draw.text((13, 9), title, font=font(size - 2), fill=INK2)
    for i, (joint, e) in enumerate(errors.items()):
        y = 33 + (size + 7) * i
        draw.text((13, y), joint.capitalize(), font=font(size), fill=INK)
        draw.text((w - 13, y), f"{e:.1f} mm", font=font(size), fill=INK, anchor="ra")
    return img


def legend(canvas, items, right="", size=18):
    """Legend line at the bottom: items ((symbol, colour), text) from the left, symbol one of ring, solid, dashed,
    trail; right-aligned text at the end."""
    y, x, places = canvas.shape[0] - LEGEND_H / 2, 16, []
    for (kind, colour), text in items:
        c = np.array([x + 20.0, y])
        if kind == "ring":
            rings(canvas, c[None])
        elif kind == "trail":
            trail(canvas, np.linspace(c - [20, 0], c + [20, 0], 15), colour)
        else:
            chain(canvas, np.array([c - [18, 0], c + [18, 0]]), colour, kind == "solid")
        places.append((x + 48, text))
        x += 48 + font(size).getlength(text) + 36
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for x, text in places:
        draw.text((x, y), text, font=font(size), fill=INK, anchor="lm")
    draw.text((canvas.shape[1] - 16, y), right, font=font(size), fill=INK, anchor="rm")
    canvas[:] = np.asarray(pil)


def free_corner(panels, box, pad=10):
    """Canvas (x, y) of the panel corner, other than the top-left one, where a box covers the fewest points.

    panels: (origin, size, points in panel px) in order of preference."""
    bw, bh = box
    best = None
    for (ox, oy), (w, h), pts in panels:
        for x, y in ((w - bw - pad, h - bh - pad), (pad, h - bh - pad), (w - bw - pad, pad)):
            inside = ((pts[:, 0] > x - pad) & (pts[:, 0] < x + bw + pad)
                      & (pts[:, 1] > y - pad) & (pts[:, 1] < y + bh + pad)).sum()
            if best is None or inside < best[0]:
                best = inside, (ox + x, oy + y)
    return best[1]


# ----------------------------------------------------------------------------- video files
def write_mp4(path, size, fps, frames):
    """H.264 MP4 (yuv420p, moov atom first) from RGB images of the given (width, height)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in frames:
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit(f"ffmpeg failed on {path}")


def to_gif(mp4):
    """GIF copy of an MP4: denoised in time so that the background stays still between frames (smaller file), one
    palette weighted to the moving pixels (keeps the model colours apart)."""
    vf = (f"fps={GIF_FPS},scale={GIF_BOX[0]}:{GIF_BOX[1]}:force_original_aspect_ratio=decrease:flags=lanczos,"
          "hqdn3d=4:4:20:20,split[a][b];"
          f"[a]palettegen=max_colors={GIF_COLOURS}:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf,
                    str(mp4.with_suffix(".gif"))], check=True)


# ----------------------------------------------------------------------------- demos
def render_best(root, row, out):
    """demo_best_model.mp4: the best model on the top and side views, every frame of the trial."""
    model, name, colour = BEST[0], BEST[1], rgb(BEST[2])
    errors = median_error(root, row, model)
    print(f"median 3D error of {name} (mm): " + ", ".join(f"{j} {e:.2f}" for j, e in errors.items()))
    n, fps = int(max(row.top_n_frames, row.side_n_frames)), float(row.top_fps)
    resting = "right" if row.hand == "left" else "left"
    views, x = {}, GAP
    for v in ("top", "side"):
        lab = frame_table(root, row[f"{v}_labels"], n)
        gt, video = joints_xy(lab, row.hand), root / row[f"{v}_video"]
        box = crop_box(gt, PANEL[v], image_shape(video))
        views[v] = dict(origin=(x, GAP), box=box, frames=clip_frames(video, n), gt=to_panel(gt, box, PANEL[v]),
                        kp=to_panel(model_xy(root, row, v, model, n), box, PANEL[v]),
                        arms=to_panel(np.concatenate([gt, joints_xy(lab, resting)]), box, PANEL[v]).reshape(-1, 2))
        x += PANEL[v][0] + GAP
    size = (x, GAP + PANEL["top"][1] + LEGEND_H)
    box_img = error_box(errors, colour)
    corner = free_corner([(views[v]["origin"], PANEL[v], views[v]["arms"]) for v in ("side", "top")], box_img.size)
    n_trail = int(round(TRAIL_S * fps))
    background = np.full((size[1], size[0], 3), 255, np.uint8)
    legend(background, [(("ring", None), "VICON joints projected into the view"),
                        (("solid", colour), f"{name} keypoints"),
                        (("trail", rgb(BEST[2], LIGHT)), f"model fingertip, last {TRAIL_S:g} s")],
           right=f"{row.participant_id} {row.session} {row.run}")

    def frames():
        for k in range(n):
            canvas = background.copy()
            for v, d in views.items():
                img = crop(next(d["frames"]), d["box"], PANEL[v])
                trail(img, d["kp"][max(0, k - n_trail + 1):k + 1, -1], rgb(BEST[2], LIGHT))
                chain(img, d["kp"][k], colour)
                rings(img, d["gt"][k])
                (ox, oy), (w, h) = d["origin"], PANEL[v]
                canvas[oy:oy + h, ox:ox + w] = img
            pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil, "RGBA")
            for v, d in views.items():
                label(draw, (d["origin"][0] + 8, d["origin"][1] + 8), f"{name}, {v} view, {k / fps:.2f} s")
            pil.paste(box_img, corner, box_img)
            yield np.asarray(pil)

    write_mp4(out / "demo_best_model.mp4", size, fps, frames())


def render_all(root, row, view, out):
    """demo_all_models.mp4: the 8 models zero-shot and with ViconTraj-MS on one view, 2 x 4 tiles."""
    n, fps = int(row[f"{view}_n_frames"]), float(row[f"{view}_fps"])
    lab = frame_table(root, row[f"{view}_labels"], n)
    video = root / row[f"{view}_video"]
    tw, th = TILE[view]
    box = crop_box(joints_xy(lab, row.hand), TILE[view], image_shape(video))
    gt = to_panel(joints_xy(lab, row.hand), box, TILE[view])
    size = (4 * tw + 5 * GAP, 2 * (TITLE_H + th) + 2 * GAP + LEGEND_H)
    tiles, titles = [], []
    background = np.full((size[1], size[0], 3), 255, np.uint8)
    for i, (zero_shot, name, hex_colour, ft) in enumerate(MODELS):
        x, y = GAP + (i % 4) * (tw + GAP), GAP + (i // 4) * (TITLE_H + th + GAP) + TITLE_H
        tiles.append(((x, y), rgb(hex_colour), rgb(hex_colour, LIGHT),
                      to_panel(model_xy(root, row, view, zero_shot, n), box, TILE[view]),
                      to_panel(model_xy(root, row, view, ft, n), box, TILE[view])))
        cv2.circle(background, ip((x + 9, y - TITLE_H / 2)), 7 * SUB, rgb(hex_colour), -1, cv2.LINE_AA, 4)
        titles.append(((x + 24, y - TITLE_H / 2), name))
    pil = Image.fromarray(background)
    draw = ImageDraw.Draw(pil)
    for xy, name in titles:
        draw.text(xy, name, font=font(20), fill=INK, anchor="lm")
    background = np.array(pil)
    legend(background, [(("ring", None), "VICON joints projected into the view"),
                        (("dashed", rgb("#52514e", LIGHT)), "zero-shot"),
                        (("solid", INK2), "with ViconTraj-MS")])
    trial_id = f"{row.participant_id} {row.session} {row.run}, {view} view"

    def frames():
        for k, frame in enumerate(clip_frames(video, n)):
            img = crop(frame, box, TILE[view])
            canvas = background.copy()
            for (x, y), colour, light, zs, ft in tiles:
                tile = img.copy()
                chain(tile, zs[k], light, solid=False)
                chain(tile, ft[k], colour)
                rings(tile, gt[k])
                canvas[y:y + th, x:x + tw] = tile
            pil = Image.fromarray(canvas)
            ImageDraw.Draw(pil).text((size[0] - 16, size[1] - LEGEND_H / 2), f"{trial_id}, {k / fps:.2f} s",
                                     font=font(18), fill=INK, anchor="rm")
            yield np.asarray(pil)

    write_mp4(out / "demo_all_models.mp4", size, fps, frames())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT / "release" / "ViKi-Traj", help="released dataset")
    ap.add_argument("--trial", nargs=3, default=["sub-03", "ses-1", "run-06"], metavar=("SUB", "SES", "RUN"))
    ap.add_argument("--view", choices=("top", "side"), default="top", help="view of demo_all_models")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    args = ap.parse_args()
    row = trial(args.root, *args.trial)
    if row[["top_labels", "side_labels"]].isna().any():
        raise SystemExit("the trial needs labels2d of both views")
    args.out.mkdir(parents=True, exist_ok=True)
    render_best(args.root, row, args.out)
    render_all(args.root, row, args.view, args.out)
    for name in ("demo_best_model", "demo_all_models"):
        to_gif(args.out / f"{name}.mp4")
        for suffix in (".mp4", ".gif"):
            path = args.out / f"{name}{suffix}"
            print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
