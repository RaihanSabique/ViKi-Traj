"""Person detector and pose model wrappers: RT-DETR, ViTPose+-B, RTMPose-WB (ONNX), MediaPipe BlazePose and hands.

Each pose model maps the BGR frames of a clip to keypoints (T, K, 2) in image px and scores (T, K); MediaPipe also
returns the 21 hand landmarks of each side.
"""
import cv2
import numpy as np
import torch

from common import MODELS, box_to_affine, invert_affine_points

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def iou(a, b):
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


class Detector:
    def __init__(self):
        from transformers import AutoProcessor, RTDetrForObjectDetection
        rid = "PekingU/rtdetr_r50vd_coco_o365"
        self.proc = AutoProcessor.from_pretrained(rid)
        self.model = RTDetrForObjectDetection.from_pretrained(rid).to(DEV).eval()

    @torch.no_grad()
    def people(self, frames):
        """Per frame: person boxes (x0, y0, x1, y1) with score >= 0.3, and their scores."""
        H, W = frames[0].shape[:2]
        out = []
        for i in range(0, len(frames), 16):
            batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[i:i + 16]]
            inp = self.proc(images=batch, return_tensors="pt").to(DEV)
            res = self.proc.post_process_object_detection(self.model(**inp), threshold=0.3,
                                                          target_sizes=[(H, W)] * len(batch))
            for r in res:
                keep = r["labels"] == 0
                out.append((r["boxes"][keep].cpu().numpy(), r["scores"][keep].cpu().numpy()))
        return out


def participant_boxes(det, frames, ref):
    """(T, 4) participant box per frame: the person with the largest IoU (> 0.3) with the reference box ref, gaps
    interpolated, 5-frame median filter; ref itself in every frame when no person matches."""
    from scipy.ndimage import median_filter
    boxes = np.full((len(frames), 4), np.nan, np.float32)
    for i, (b, s) in enumerate(det.people(frames)):
        if len(b):
            ov = np.array([iou(x, ref) for x in b])
            if ov.max() > 0.3:
                boxes[i] = b[np.argmax(ov)]
    good = ~np.isnan(boxes[:, 0])
    if not good.any():
        return np.tile(ref, (len(frames), 1)).astype(np.float32)
    for c in range(4):
        boxes[:, c] = np.interp(np.arange(len(boxes)), np.flatnonzero(good), boxes[good, c])
    return median_filter(boxes, size=(5, 1), mode="nearest")


class ViTPose:
    def __init__(self):
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        rid = "usyd-community/vitpose-plus-base"
        self.proc = AutoProcessor.from_pretrained(rid)
        self.model = VitPoseForPoseEstimation.from_pretrained(rid).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, frames, boxes):
        T = len(frames)
        k, s = np.zeros((T, 17, 2), np.float32), np.zeros((T, 17), np.float32)
        xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
        for i in range(0, T, 32):
            imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[i:i + 32]]
            bx = [[xywh[i + j].tolist()] for j in range(len(imgs))]
            inp = self.proc(imgs, boxes=bx, return_tensors="pt").to(DEV)
            inp["dataset_index"] = torch.zeros(len(imgs), dtype=torch.long, device=DEV)  # COCO expert
            res = self.proc.post_process_pose_estimation(self.model(**inp), boxes=bx)
            for j, r in enumerate(res):
                k[i + j] = r[0]["keypoints"].cpu().numpy()
                s[i + j] = r[0]["scores"].cpu().numpy()
        return k, s


class RTMPose:
    def __init__(self):
        import onnxruntime as ort
        onnx = MODELS / "rtmpose_body2d/rtmpose_body2d-onnx-float/rtmpose_body2d.onnx"
        self.sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])

    def __call__(self, frames, boxes):
        T = len(frames)
        k, s = np.zeros((T, 133, 2), np.float32), np.zeros((T, 133), np.float32)
        for i in range(T):
            M = box_to_affine(boxes[i], 192, 256, pad=1.25)
            crop = cv2.warpAffine(frames[i], M, (192, 256), flags=cv2.INTER_LINEAR)
            # Qualcomm AI Hub export: takes 0-255 BGR (the graph normalises itself); SimCC bins are half pixels
            px, py = self.sess.run(None, {"image": crop.astype(np.float32)[None].transpose(0, 3, 1, 2)})
            pts = np.stack([px[0].argmax(-1), py[0].argmax(-1)], -1).astype(np.float32) / 2.0
            k[i] = invert_affine_points(M, pts)
            s[i] = np.minimum(px[0].max(-1), py[0].max(-1))
        return k, s


class MediaPipe:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        self.vision, self.BaseOptions = vision, BaseOptions

    def __call__(self, frames, t):
        import mediapipe as mp
        v = self.vision
        pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(MODELS / "mediapipe/pose_landmarker_heavy.task")),
            running_mode=v.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.3, min_tracking_confidence=0.3))
        hands = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(MODELS / "mediapipe/hand_landmarker.task")),
            running_mode=v.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))
        T = len(frames)
        H, W = frames[0].shape[:2]
        k = np.full((T, 33, 2), np.nan, np.float32)
        s = np.zeros((T, 33), np.float32)
        hand = {"L": np.full((T, 21, 2), np.nan, np.float32), "R": np.full((T, 21, 2), np.nan, np.float32)}
        for i in range(T):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            ms = int(round(t[i] * 1000))
            r = pose.detect_for_video(img, ms)
            if r.pose_landmarks:
                k[i] = [[p.x * W, p.y * H] for p in r.pose_landmarks[0]]
                s[i] = [p.visibility for p in r.pose_landmarks[0]]
            rh = hands.detect_for_video(img, ms)
            # Each hand goes to the nearer pose wrist (15 left, 16 right) within 150 px; the handedness label
            # assumes a mirrored selfie view, which these cameras are not.
            for lm in rh.hand_landmarks or []:
                h = np.array([[p.x * W, p.y * H] for p in lm], np.float32)
                d = {side: np.linalg.norm(h[0] - k[i, w]) for side, w in (("L", 15), ("R", 16))}
                side = min(d, key=lambda q: d[q] if np.isfinite(d[q]) else np.inf)
                if np.isfinite(d[side]) and d[side] <= 150:
                    hand[side][i] = h
        pose.close()
        hands.close()
        return k, s, hand["L"], hand["R"]


def read_range(cap, f0, f1):
    """Frames f0 .. f1 - 1 of an open video and their container times (s)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f0))
    frames, t = [], []
    for _ in range(f0, f1):
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    return frames, np.asarray(t)
