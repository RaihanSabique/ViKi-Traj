"""Face blurring: an ellipse over the head (projected head markers and face keypoints) and over every YuNet face."""
import cv2
import numpy as np

from common import MODELS


class FaceBlur:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(MODELS / "face/face_detection_yunet_2023mar.onnx"),
                                             "", (320, 320), 0.6)

    def regions(self, img, head_pts, face_kp):
        """Ellipses (cx, cy, rx, ry) in px around head_pts and face_kp ((N, 2) or None) and around each YuNet face."""
        out = []
        H, W = img.shape[:2]
        pts = [p for p in (head_pts, face_kp) if p is not None and len(p)]
        if pts:
            p = np.concatenate(pts)
            c = p.mean(0)
            r = max(np.linalg.norm(p - c, axis=1).max() * 1.35, 0.06 * W)
            out.append((c[0], c[1], r, r * 1.15))
        self.det.setInputSize((W, H))
        _, faces = self.det.detect(img)
        for f in faces if faces is not None else []:
            x, y, w, h = f[:4]
            out.append((x + w / 2, y + h / 2, w * 0.8, h * 0.9))
        return out

    @staticmethod
    def apply(img, regions):
        """img with the regions pixelated and blurred (feathered edges)."""
        if not regions:
            return img
        H, W = img.shape[:2]
        small = cv2.resize(img, (W // 24, H // 24), interpolation=cv2.INTER_AREA)
        pix = cv2.GaussianBlur(cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST), (0, 0), 6)
        mask = np.zeros((H, W), np.uint8)
        for cx, cy, rx, ry in regions:
            if np.isfinite([cx, cy, rx, ry]).all():
                cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
        m = cv2.GaussianBlur(mask, (0, 0), 5)[..., None] / 255.0
        return (img * (1 - m) + pix * m).astype(np.uint8)
