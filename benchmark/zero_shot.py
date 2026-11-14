"""Run the zero-shot and the fine-tuned pose models on the valid top and side clips of one participant.

The frames of every valid clip (benchlib.valid_rows) are decoded from the source video with PAD_S s on each side,
so the filters have room at the movement edges. Per clip it writes, under outputs/benchmark/:
    boxes/<pid>/<view>/<clip>.npz           frame, fps, box (T, 4): RT-DETR participant box, shared by the models
    models/<model>/<pid>/<view>/<clip>.npz  frame, fps, k (T, K, 2) image px, s (T, K) scores; BlazePose adds
                                            hand_L, hand_R; RTMPose-m hand has hand_<L|R> and hand_s_<L|R> only
    blobs/<pid>/<view>/<clip>.npz           frame, n, xy (M, 2): white marker-blob candidates on every
                                            BLOB_STEP-th frame of the movement window
Existing files are kept, so a later call runs only the missing models. --models takes FACTORY names, or ft_<run>
for a run of finetune.py / finetune_sapiens.py. Weights are read from models/ (Hugging Face cache, ONNX files,
MediaPipe tasks, Sapiens2 checkpoints); the Sapiens2 code from third_party/sapiens2.

    python benchmark/zero_shot.py --participant XXXXXXX103 [--models vitpose_b,rtmpose_wb,blazepose] [--no-blobs]
    python benchmark/zero_shot.py --participant XXXXXXX103 --models ft_b_full_pooled_traj,ft_sapiens2_1b_head --no-blobs
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from benchlib import ROOT, aux_file, clip_stem, model_file, valid_rows, VIEWS

PAD_S = 0.5                   # s decoded before and after each clip
BLOB_STEP = 3                 # blobs at 10 Hz: calibration and visibility labels
NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
cv2.setNumThreads(NCPU)

from blobs import detect_candidates                                    # noqa: E402
import trial_models as tm                                              # noqa: E402


ORT_GPU = os.environ.get("BENCH_ORT_GPU") == "1"     # onnxruntime-gpu on PYTHONPATH (slurm/bench_onnx_gpu.sbatch)


def ort_session(path):
    """ONNX Runtime session: CUDA when BENCH_ORT_GPU=1 (CUDA 12 / cuDNN 9 wheels preloaded), else CPU
    with the thread count of the Slurm task."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = NCPU
    providers = ["CPUExecutionProvider"]
    if ORT_GPU:
        ort.preload_dlls(cuda=True, cudnn=True)
        providers = ["CUDAExecutionProvider"] + providers
    sess = ort.InferenceSession(str(path), so, providers=providers)
    if ORT_GPU and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider not available for {path}")
    return sess


class RTMPoseCPU(tm.RTMPose):
    """RTMPose-WB: trial_models.RTMPose (COCO-WholeBody, ONNX export of Qualcomm AI Hub) on an ort_session."""
    def __init__(self):
        self.sess = ort_session(tm.MODELS / "rtmpose_body2d/rtmpose_body2d-onnx-float/rtmpose_body2d.onnx")


class ViTPoseHF(tm.ViTPose):
    """ViTPose+-L or -H from Hugging Face, otherwise as trial_models.ViTPose (COCO expert)."""
    def __init__(self, rid):
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.proc = AutoProcessor.from_pretrained(rid)
        self.model = VitPoseForPoseEstimation.from_pretrained(rid).to(tm.DEV).eval()


class SimCC:
    """Top-down SimCC model of the RTMPose family (DWPose, RTMW, RTMPose-m hand) exported to ONNX, run in batches.
    Input: the box padded by 1.25 and aspect-fixed, RGB, ImageNet mean / std (mmdeploy pipeline.json: to_rgb).
    Keypoint: arg-max of the x and y SimCC vectors (2 bins per px); score: mean of the two maxima."""
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, onnx, w, h, batch=16):
        self.sess = ort_session(onnx)
        self.inp = self.sess.get_inputs()[0].name
        self.w, self.h, self.batch = w, h, batch

    def run(self, frames, boxes):
        """boxes (T, 4) xyxy, NaN rows skipped -> k (T, K, 2), s (T, K); (None, None) if no row has a box."""
        from common import box_to_affine, invert_affine_points
        T = len(frames)
        ok = np.flatnonzero(~np.isnan(boxes).any(1))
        k = s = None
        for i in range(0, len(ok), self.batch):
            idx = ok[i:i + self.batch]
            Ms, X = [], []
            for j in idx:
                M = box_to_affine(boxes[j], self.w, self.h, pad=1.25)
                c = cv2.cvtColor(cv2.warpAffine(frames[j], M, (self.w, self.h), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                X.append(((c.astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1))
                Ms.append(M)
            px, py = self.sess.run(None, {self.inp: np.stack(X)})
            if k is None:
                K = px.shape[1]
                k, s = np.full((T, K, 2), np.nan, np.float32), np.zeros((T, K), np.float32)
            loc = np.stack([px.argmax(-1), py.argmax(-1)], -1).astype(np.float32) / 2.0
            sc = (px.max(-1) + py.max(-1)) / 2
            for n, j in enumerate(idx):
                k[j] = invert_affine_points(Ms[n], loc[n])
                s[j] = sc[n]
        return k, s

    def __call__(self, frames, boxes):
        k, s = self.run(frames, boxes)
        if k is None:
            raise RuntimeError("no valid box in the clip")
        return k, s


ONNX = tm.MODELS / "onnx"
HAND_ONNX = next(ONNX.glob("rtmpose-m_simcc-hand5*/**/end2end.onnx"), None)


def hand_boxes(wb_k, wb_s, side, thr=0.3):
    """Square hand box per frame from a COCO-WholeBody result: the confident hand keypoints,
    else the wrist pushed along the forearm. NaN when neither is available."""
    h0 = 91 if side == "L" else 112                  # first keypoint of the 21-point hand block
    wr, el = (9, 7) if side == "L" else (10, 8)
    T = len(wb_k)
    out = np.full((T, 4), np.nan, np.float32)
    for t in range(T):
        fore = np.linalg.norm(wb_k[t, wr] - wb_k[t, el]) if min(wb_s[t, wr], wb_s[t, el]) >= thr else np.nan
        hk = wb_k[t, h0:h0 + 21][wb_s[t, h0:h0 + 21] >= thr]
        if len(hk) >= 5:
            c = (hk.min(0) + hk.max(0)) / 2
            size = max(np.ptp(hk, 0).max(), 0.5 * fore if np.isfinite(fore) else 0) * 1.2
        elif np.isfinite(fore):
            c = wb_k[t, wr] + 0.45 * (wb_k[t, wr] - wb_k[t, el])
            size = 0.9 * fore
        else:
            continue
        if size > 4:
            out[t] = [c[0] - size / 2, c[1] - size / 2, c[0] + size / 2, c[1] + size / 2]
    return out


class HandRTMPose(SimCC):
    """RTMPose-m hand (21 keypoints) on hand crops placed from the RTMW-X whole-body result."""
    def __init__(self):
        super().__init__(HAND_ONNX, 256, 256)

    def __call__(self, frames, wb):
        out = {}
        for side in "LR":
            k, s = self.run(frames, hand_boxes(wb["k"], wb["s"], side))
            if k is None:
                k, s = np.full((len(frames), 21, 2), np.nan, np.float32), np.zeros((len(frames), 21), np.float32)
            out[f"hand_{side}"], out[f"hand_s_{side}"] = k, s
        return out


class Sapiens2:
    """Sapiens2 pose (308 keypoints, 768 x 1024 crops), fp16 on GPU, no flip test. Only the KEEP heatmaps are
    decoded; the other keypoints are NaN with score 0."""
    KEEP = [5, 6, 7, 8, 25, 32, 41, 46, 53, 62]          # shoulders, elbows, wrists, index tips, middle MCPs
    POSE = ROOT / "third_party" / "sapiens2" / "sapiens" / "pose"

    def __init__(self, size, batch=8):
        import sys
        import torch
        sys.path.insert(0, str(ROOT / "third_party" / "sapiens2"))
        from sapiens.pose.datasets import UDPHeatmap
        from sapiens.pose.models import init_model
        cfg = self.POSE / f"configs/keypoints308/shutterstock_goliath_3po/sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py"
        self.model = init_model(str(cfg), str(tm.MODELS / "sapiens2" / "pose" / f"sapiens2_{size}_pose.safetensors"), device=tm.DEV)
        codec = dict(self.model.cfg.codec)
        codec.pop("type")
        self.codec, self.batch, self.torch = UDPHeatmap(**codec), batch, torch

    def __call__(self, frames, boxes):
        torch, m = self.torch, self.model
        T = len(frames)
        k, s = np.full((T, 308, 2), np.nan, np.float32), np.zeros((T, 308), np.float32)
        for i in range(0, T, self.batch):
            ins, metas = [], []
            for j in range(i, min(i + self.batch, T)):
                d = m.data_preprocessor(m.pipeline(dict(img=frames[j], bbox=boxes[j][None].astype(np.float32),
                                                        bbox_score=np.ones(1, np.float32))))
                ins.append(d["inputs"])
                metas.append(d["data_samples"]["meta"])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=tm.DEV == "cuda"):
                pred = m(torch.cat(ins, 0))[:, self.KEEP].float().cpu().numpy()
            for n, meta in enumerate(metas):
                kp, sc = self.codec.decode(pred[n])
                k[i + n, self.KEEP] = kp[0] / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
                s[i + n, self.KEEP] = sc[0]
        return k, s


class FineTuned:
    """ViTPose+ fine-tuned on the joint set by finetune.py: the training crop (box padded by 1.25, 192 x 256,
    RGB, ImageNet mean / std), DARK sub-pixel peak, averaged with the prediction on the mirrored crop."""
    def __init__(self, run, batch=64):
        import torch
        from finetune import FLIP, MEAN, STD, OUT, build_model, decode, forward
        self.flip = FLIP
        import json
        cfg = json.loads((OUT / run / "config.json").read_text())
        self.model = build_model(cfg.get("backbone", "base"))
        self.model.load_state_dict(torch.load(OUT / run / "model.pt", map_location="cpu"))
        self.model = self.model.to(tm.DEV).eval()
        self.torch, self.mean, self.std, self.decode, self.forward, self.batch = torch, MEAN, STD, decode, forward, batch

    def __call__(self, frames, boxes):
        from common import box_to_affine, invert_affine_points
        torch = self.torch
        T = len(frames)
        k, s = np.full((T, 10, 2), np.nan, np.float32), np.zeros((T, 10), np.float32)
        for i in range(0, T, self.batch):
            idx = [j for j in range(i, min(i + self.batch, T)) if not np.isnan(boxes[j]).any()]
            if not idx:
                continue
            Ms = [box_to_affine(boxes[j], 192, 256, pad=1.25) for j in idx]
            X = np.stack([((cv2.cvtColor(cv2.warpAffine(frames[j], M, (192, 256), flags=cv2.INTER_LINEAR),
                                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - self.mean) / self.std).transpose(2, 0, 1)
                          for j, M in zip(idx, Ms)])
            x = torch.from_numpy(X).to(tm.DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=tm.DEV == "cuda"):
                hm, hf = self.forward(self.model, x).float(), self.forward(self.model, x.flip(-1)).float()
            p, sc = self.decode(hm)
            pf, sf = self.decode(hf)                                     # flip test: mirrored crop, sides swapped back
            pf[..., 0] = X.shape[-1] - 1 - pf[..., 0]
            p, sc = (p + pf[:, self.flip]) / 2, (sc + sf[:, self.flip]) / 2
            p, sc = p.cpu().numpy(), sc.cpu().numpy()
            for n, j in enumerate(idx):
                k[j] = invert_affine_points(Ms[n], p[n])
                s[j] = sc[n]
        return k, s


class FineTunedSapiens:
    """Sapiens2 with the head fine-tuned by finetune_sapiens.py: 768 x 1024 crops (box padded by 1.25, as in training),
    the model's UDP decoding of the ten joint-set channels, no flip test; output in the Sapiens2 308-keypoint layout."""
    def __init__(self, run, batch=8):
        import json
        import torch
        from finetune import OUT
        from finetune_sapiens import H, KEEP, MEAN, STD, W, sapiens_model
        cfg = json.loads((OUT / run / "config.json").read_text())
        self.model, self.codec = sapiens_model(cfg["size"], device=tm.DEV)
        self.model.decode_head.load_state_dict(torch.load(OUT / run / "head.pt", map_location="cpu"))
        self.model.eval()
        self.torch, self.batch, self.keep, self.mean, self.std, self.w, self.h = torch, batch, KEEP, MEAN, STD, W, H

    def __call__(self, frames, boxes):
        from common import box_to_affine, invert_affine_points
        torch = self.torch
        T = len(frames)
        k, s = np.full((T, 308, 2), np.nan, np.float32), np.zeros((T, 308), np.float32)
        for i in range(0, T, self.batch):
            idx = [j for j in range(i, min(i + self.batch, T)) if not np.isnan(boxes[j]).any()]
            if not idx:
                continue
            Ms = [box_to_affine(boxes[j], self.w, self.h, pad=1.25) for j in idx]
            X = np.stack([((cv2.cvtColor(cv2.warpAffine(frames[j], M, (self.w, self.h), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                            .astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1) for j, M in zip(idx, Ms)])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=tm.DEV == "cuda"):
                pred = self.model(torch.from_numpy(X).to(tm.DEV))[:, self.keep].float().cpu().numpy()
            for n, j in enumerate(idx):
                kp, sc = self.codec.decode(pred[n])
                k[j, self.keep] = invert_affine_points(Ms[n], kp[0])
                s[j, self.keep] = sc[0]
        return k, s


def run_model(name, net, frames, t, boxes, prior=None):
    """Arrays of one model on one clip; BlazePose takes the frame times, RTMPose-m hand the RTMW-X result."""
    if name == "blazepose":
        k, s, hl, hr = net(frames, t)
        return dict(k=k, s=s, hand_L=hl, hand_R=hr)
    if name == "rtmpose_hand":
        return net(frames, prior("rtmw_x"))
    k, s = net(frames, boxes)
    return dict(k=k, s=s)


FACTORY = {
    "vitpose_b": tm.ViTPose, "rtmpose_wb": RTMPoseCPU, "blazepose": tm.MediaPipe,
    "vitpose_l": lambda: ViTPoseHF("usyd-community/vitpose-plus-large"),
    "vitpose_h": lambda: ViTPoseHF("usyd-community/vitpose-plus-huge"),
    "dwpose_l": lambda: SimCC(ONNX / "dw-ll_ucoco_384.onnx", 288, 384),
    "rtmw_x": lambda: SimCC(ONNX / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122" / "end2end.onnx", 288, 384),
    "rtmpose_hand": HandRTMPose,
    "sapiens2_0.4b": lambda: Sapiens2("0.4b"),
    "sapiens2_1b": lambda: Sapiens2("1b", batch=4),
}
# top-down models that need the person box (rtmpose_hand needs rtmw_x, run it after)
BOX_MODELS = {"vitpose_b", "rtmpose_wb", "vitpose_l", "vitpose_h", "dwpose_l", "rtmw_x", "sapiens2_0.4b", "sapiens2_1b"}


def save(path, **arrays):
    """Write atomically, so a cancelled job never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def reference_box(det, cap, rows, view, n=40):
    """Median of the largest confident person (area x score) over frames spread across the valid trials of the video."""
    mids = np.concatenate([np.linspace(r[f"{view}_start_frame"], r[f"{view}_start_frame"] + r[f"{view}_n_frames"] - 1, 4)
                           for _, r in rows.iterrows()]).astype(int)
    pick = mids[np.linspace(0, len(mids) - 1, min(n, len(mids))).astype(int)]
    frames = []
    for f in pick:
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
    ap.add_argument("--participant", required=True)
    ap.add_argument("--models", default=",".join(FACTORY))
    ap.add_argument("--no-blobs", action="store_true", help="skip the blob detection (e.g. a BlazePose-only CPU job)")
    ap.add_argument("--max-clips", type=int, default=None, help="testing: at most this many clips per video")
    args = ap.parse_args()
    pid, names = args.participant, args.models.split(",")
    need_boxes = any(n in BOX_MODELS or n.startswith("ft_") for n in names)
    need_blobs = not args.no_blobs
    nets, det = {}, None
    for view in VIEWS:
        rows = valid_rows(view, [pid])
        for video, g in rows.groupby(f"{view}_source_video"):
            todo = [(r, clip_stem(r, view)) for _, r in g.iterrows()]
            todo = [(r, st) for r, st in todo
                    if not all(model_file(n, pid, view, st).exists() for n in names)
                    or (need_blobs and not aux_file("blobs", pid, view, st).exists())]
            todo = todo[:args.max_clips]
            if not todo:
                continue
            if det is None and need_boxes and not ORT_GPU:      # keep torch's CUDA libraries out of ORT jobs
                det = tm.Detector()
            for n in names:
                if n not in nets:
                    nets[n] = FineTunedSapiens(n) if n.startswith("ft_sapiens2") else FineTuned(n) if n.startswith("ft_") else FACTORY[n]()
            cap = cv2.VideoCapture(str(ROOT / video))
            fps, nf = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ref = reference_box(det, cap, g, view) if det is not None else None
            print(f"[{pid} {view}] {video}: {len(todo)} clips, {fps:.2f} fps, reference box {ref}", flush=True)
            for r, stem in tqdm(todo, desc=f"{pid} {view}", mininterval=30):
                s0, n0 = int(r[f"{view}_start_frame"]), int(r[f"{view}_n_frames"])
                pad = int(round(PAD_S * fps))
                f0, f1 = max(s0 - pad, 0), min(s0 + n0 + pad, nf)
                frames, t = tm.read_range(cap, f0, f1)
                fidx = np.arange(f0, f0 + len(frames))
                bf, boxes = aux_file("boxes", pid, view, stem), None
                if need_boxes and bf.exists():
                    boxes = np.load(bf)["box"]
                elif need_boxes and ORT_GPU:
                    print(f"[{pid} {view}] {stem}: no boxes yet; skipped (run the boxes job first)", flush=True)
                    continue
                elif need_boxes:
                    boxes = tm.participant_boxes(det, frames, ref)
                    save(bf, frame=fidx, box=boxes, fps=fps)
                for n in names:
                    mf = model_file(n, pid, view, stem)
                    if mf.exists():
                        continue
                    prior = lambda m: dict(np.load(model_file(m, pid, view, stem)))
                    save(mf, frame=fidx, fps=fps, **run_model(n, nets[n], frames, t, boxes, prior))
                lf = aux_file("blobs", pid, view, stem)
                if need_blobs and not lf.exists():
                    sel = [i for i, f in enumerate(fidx) if s0 <= f < s0 + n0 and (f - s0) % BLOB_STEP == 0]
                    bl = [detect_candidates(frames[i])[:, :2] for i in sel]
                    save(lf, frame=fidx[sel], n=np.array([len(b) for b in bl]),
                         xy=np.concatenate(bl) if bl else np.zeros((0, 2), np.float32))
            cap.release()
    print(f"[{pid}] done", flush=True)


if __name__ == "__main__":
    main()
