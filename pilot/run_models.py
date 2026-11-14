"""Pilot analysis of one participant (supplement), step 2: ViTPose+, RTMPose and MediaPipe on every aligned trial.

Reads outputs/<pid>/alignment/trial_segments.csv (align_trials.py) and decodes the frames of each trial, +/- pad s,
from its visit video in data/<pid>/. Writes outputs/<pid>/models/<visit>_<view>_T<row>.npz (existing files are skipped):
    frame, t_video, fps      video frame index and time (s) of every processed frame, frame rate
    box                      participant box (x0, y0, x1, y1)
    vit_k, vit_s             ViTPose+-B, COCO-17 keypoints (T, 17, 2) in image px and scores
    rtm_k, rtm_s             RTMPose-WB (RTMPose-Body2d), COCO-WholeBody-133 keypoints (T, 133, 2) and scores
    mp_k, mp_s               MediaPipe BlazePose heavy, 33 landmarks and visibility
    mp_hand_L, mp_hand_R     MediaPipe hand landmarker, 21 points per hand (NaN if absent)

    python pilot/run_models.py --participant PID [--video STEM]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))                    # src/ modules
from trial_models import Detector, MediaPipe, RTMPose, ViTPose, participant_boxes, read_range  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"


def reference_box(det, cap, n_frames, n=30):
    """Median of the largest confident person over n frames spread through the video; participant_boxes keeps the
    person that overlaps it, as an experimenter is sometimes in view."""
    frames = []
    for f in np.linspace(0, n_frames - 2, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            frames.append(im)
    boxes = []
    for b, s in det.people(frames):
        if len(b):
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            boxes.append(b[np.argmax(area * s)])
    return np.median(np.array(boxes), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="XXXXXXX115")
    ap.add_argument("--video", default=None, help="only this video stem")
    ap.add_argument("--pad", type=float, default=1.0)
    args = ap.parse_args()
    seg = pd.read_csv(OUT / args.participant / "alignment" / "trial_segments.csv")
    odir = OUT / args.participant / "models"
    odir.mkdir(parents=True, exist_ok=True)
    det, vit, rtm, mpp = Detector(), ViTPose(), RTMPose(), MediaPipe()
    for video, sv in seg.groupby("video"):
        if args.video and not video.startswith(args.video):
            continue
        cap = cv2.VideoCapture(str(DATA / args.participant / video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ref = reference_box(det, cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        print(f"{video}: reference participant box {np.round(ref).astype(int)}")
        for _, r in tqdm(sv.iterrows(), total=len(sv), desc=video):
            out = odir / f"{r.visit}_{r['view']}_T{r.row:02d}.npz"
            if out.exists():
                continue
            f0 = max(int(r.start_frame - args.pad * fps), 0)
            f1 = int(r.end_frame + args.pad * fps)
            frames, t = read_range(cap, f0, f1)
            boxes = participant_boxes(det, frames, ref)
            vk, vs = vit(frames, boxes)
            rk, rs = rtm(frames, boxes)
            mk, ms, hl, hr = mpp(frames, t)
            np.savez_compressed(out, frame=np.arange(f0, f0 + len(frames)), t_video=t, box=boxes,
                                vit_k=vk, vit_s=vs, rtm_k=rk, rtm_s=rs, mp_k=mk, mp_s=ms,
                                mp_hand_L=hl, mp_hand_R=hr, fps=fps)


if __name__ == "__main__":
    main()
