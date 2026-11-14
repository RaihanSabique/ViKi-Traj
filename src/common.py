"""Model weight directory, Hugging Face cache location and the person-box crop affine of the top-down pose models."""
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))
# Weights are cached in the repository, but the login token stays where `hf auth login` puts it.
os.environ.setdefault("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))


def box_to_affine(box, out_w, out_h, pad=1.25):
    """Affine that maps an (x0,y0,x1,y1) box, padded and aspect-fixed, onto an out_w x out_h crop."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * pad, (y1 - y0) * pad
    aspect = out_w / out_h
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    s = out_w / w
    M = np.array([[s, 0, out_w / 2 - s * cx], [0, s, out_h / 2 - s * cy]], np.float32)
    return M


def invert_affine_points(M, pts):
    """Map (N, 2) crop points back to the image through the inverse of M."""
    Mi = cv2.invertAffineTransform(M)
    pts = np.asarray(pts, np.float32)
    return pts @ Mi[:, :2].T + Mi[:, 2]
