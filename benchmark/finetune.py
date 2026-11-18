"""ViTMS fine-tuning of ViTPose+-B on the joint set: projected VICON marker labels and the trajectory label.

Trains on the label files of labels.py and labels_traj.py of folds 2-5 of splits.json (--n-train: a subset stratified
by TD / UCP) and keeps the epoch with the lowest median validation error on fold 1; the test participants are never
read. Writes model.pt, log.csv and config.json to outputs/benchmark/finetune/<run>/.

    python benchmark/finetune.py --run ft_b_full_pooled_traj --views top,side --mode full --traj-weight 1
    python benchmark/finetune.py --run ft_b_full_pooled [--mode head] [--n-train 8] [--label-shift-ms 67] [--mask has]
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from benchlib import BENCH, manifest, splits

LAB = BENCH / "labels"
OUT = BENCH / "finetune"
K = 10                                           # L shoulder, elbow, wrist, hand, fingertip, then R
FLIP = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]            # channel order after a horizontal flip
COCO_INIT = [5, 7, 9, 9, 9, 6, 8, 10, 10, 10]    # COCO-17 channel that initialises each output
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EPOCHS_HEAD, EPOCHS_FULL = 5, 10                 # 'full': decoder only, then the whole model; 'head': decoder only
HM_H, HM_W, SIGMA = 64, 48, 2.0                  # heatmap px = crop px / 4


def clip_files(pids, views):
    return [f for p in pids for v in views for f in sorted((LAB / p / v).glob("*.npz")) if ".tmp" not in f.name]


class Frames(Dataset):
    """Labelled frames -> crop, heatmaps (K, HM_H, HM_W), loss weights (K,), joints (K, 2) in crop px (-1 if missing).
    shift_key: the kp_shift<+ms> labels of labels_shift.py. traj = (weight, mode): the trajectory label (kp_traj of
    labels_traj.py) as the target of the active fingertip channel, where the marker label is masked out ('fill') or
    always ('replace')."""
    def __init__(self, files, augment, mask="vis", shift_key=None, traj=(0.0, "fill")):
        self.items, self.data = [], []
        self.tw, self.tmode = traj
        for f in files:
            z = np.load(f)
            kp = z[shift_key] if shift_key else z["kp"]
            m = z["vis"] if mask == "vis" else z["has"]
            if shift_key:
                m = m & ~np.isnan(kp).any(2)
            if self.tw > 0 and "kp_traj" in z:
                kt, side = z["kp_traj"], int(z["side_traj"])
            else:
                kt, side = np.full((len(kp), 2), np.nan, np.float32), 0
            self.data.append((z["jpg"], z["jpg_off"], kp, m, kt, side))
            self.items += [(len(self.data) - 1, i) for i in range(len(z["frame"])) if m[i].any() or (self.tw > 0 and not np.isnan(kt[i]).any())]
        self.augment = augment
        ys, xs = np.mgrid[0:HM_H, 0:HM_W]
        self.grid = (xs.astype(np.float32), ys.astype(np.float32))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        c, i = self.items[n]
        jpg, off, kp, m, kt, side = self.data[c]
        img = cv2.cvtColor(cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        kp, m = np.r_[kp[i], kt[i][None]].copy(), m[i].copy()      # row K: trajectory point of the active fingertip
        if self.augment:
            rng = np.random
            if rng.rand() < 0.5:
                img = img[:, ::-1].copy()
                kp[:, 0] = img.shape[1] - 1 - kp[:, 0]
                kp[:K], m, side = kp[FLIP], m[FLIP], 1 - side
            A = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(-30, 30), rng.uniform(0.75, 1.25))
            img = cv2.warpAffine(img, A, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
            kp = kp @ A[:, :2].T + A[:, 2]
            img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        hx, hy = kp[:, 0] / 4.0, kp[:, 1] / 4.0
        inside = (hx >= 0) & (hx < HM_W) & (hy >= 0) & (hy < HM_H) & ~np.isnan(hx)
        w = (m & inside[:K]).astype(np.float32)
        gx, gy = self.grid
        hm = np.zeros((K, HM_H, HM_W), np.float32)
        for j in np.flatnonzero(w):
            hm[j] = np.exp(-((gx - hx[j]) ** 2 + (gy - hy[j]) ** 2) / (2 * SIGMA ** 2))
        j = 4 + 5 * side                                                   # the active hand's fingertip channel
        if self.tw > 0 and inside[K] and (w[j] == 0 or self.tmode == "replace"):
            hm[j] = np.exp(-((gx - hx[K]) ** 2 + (gy - hy[K]) ** 2) / (2 * SIGMA ** 2))
            w[j] = self.tw if w[j] == 0 else max(w[j], self.tw)
            kp[j] = kp[K]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(hm), torch.from_numpy(w), \
            torch.from_numpy(np.nan_to_num(kp[:K].astype(np.float32), nan=-1))


BACKBONES = {"base": "usyd-community/vitpose-plus-base", "large": "usyd-community/vitpose-plus-large", "huge": "usyd-community/vitpose-plus-huge"}


def build_model(backbone="base"):
    """Pretrained ViTPose+ with the last decoder convolution replaced by K outputs, initialised from COCO_INIT."""
    from transformers import VitPoseForPoseEstimation
    m = VitPoseForPoseEstimation.from_pretrained(BACKBONES[backbone])
    old = m.head.conv
    new = nn.Conv2d(old.in_channels, K, 1)
    with torch.no_grad():
        new.weight.copy_(old.weight[COCO_INIT])
        new.bias.copy_(old.bias[COCO_INIT])
    m.head.conv = new
    return m


def forward(model, x):
    """(B, K, 64, 48) heatmaps; dataset_index 0 selects the COCO expert of ViTPose+."""
    return model(pixel_values=x, dataset_index=torch.zeros(len(x), dtype=torch.long, device=x.device)).heatmaps


def decode(hm, quarter=False):
    """(B, K, H, W) heatmaps -> crop px (B, K, 2) and peak score. Sub-pixel peak by DARK (Zhang et al.
    2020): the heatmap is smoothed with the training Gaussian, and a second-order Taylor step on its
    log is taken from the arg-max. quarter=True: the arg-max shifted by a quarter bin toward the higher
    neighbour (SimpleBaseline decoding; keypoints on a 1-bin = 4 crop px grid)."""
    B, Kk, H, W = hm.shape
    flat = hm.reshape(B, Kk, -1)
    score, idx = flat.max(-1)
    y, x = (idx // W).float(), (idx % W).float()
    xi, yi = x.long().clamp(1, W - 2), y.long().clamp(1, H - 2)
    b = torch.arange(B, device=hm.device)[:, None]
    k = torch.arange(Kk, device=hm.device)[None]
    if quarter:
        dx = torch.sign(hm[b, k, yi, xi + 1] - hm[b, k, yi, xi - 1]) * 0.25
        dy = torch.sign(hm[b, k, yi + 1, xi] - hm[b, k, yi - 1, xi]) * 0.25
        return torch.stack([(x + dx) * 4, (y + dy) * 4], -1), score    # targets: heatmap index = crop px / 4
    r = int(3 * SIGMA)
    g = torch.exp(-torch.arange(-r, r + 1, device=hm.device, dtype=hm.dtype) ** 2 / (2 * SIGMA ** 2))
    g = g / g.sum()
    s = hm.reshape(B * Kk, 1, H, W).clamp(min=0)
    s = torch.nn.functional.conv2d(torch.nn.functional.pad(s, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    s = torch.nn.functional.conv2d(torch.nn.functional.pad(s, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
    s = s.reshape(B, Kk, H, W)
    s = s * (flat.amax(-1) / s.reshape(B, Kk, -1).amax(-1).clamp(min=1e-12))[..., None, None]
    L = torch.log(s.clamp(min=1e-10))

    def at(dy_, dx_):
        return L[b, k, yi + dy_, xi + dx_]
    gx, gy = 0.5 * (at(0, 1) - at(0, -1)), 0.5 * (at(1, 0) - at(-1, 0))
    hxx, hyy = at(0, 1) - 2 * at(0, 0) + at(0, -1), at(1, 0) - 2 * at(0, 0) + at(-1, 0)
    hxy = 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))
    det = hxx * hyy - hxy ** 2
    ok = (det > 1e-12) & (hxx < 0) & (hyy < 0)
    det = torch.where(ok, det, torch.ones_like(det))
    dx = torch.where(ok, -(hyy * gx - hxy * gy) / det, torch.zeros_like(det)).clamp(-1, 1)
    dy = torch.where(ok, -(hxx * gy - hxy * gx) / det, torch.zeros_like(det)).clamp(-1, 1)
    px = torch.where(xi.float() == x, x + dx, x)                         # peak on the border: no step
    py = torch.where(yi.float() == y, y + dy, y)
    return torch.stack([px * 4, py * 4], -1), score


def evaluate(model, loader, dev):
    """Median and mean error (crop px) of the labelled joints of the loader, DARK decoding without flip."""
    model.eval()
    errs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, hm, m, kp in loader:
            p, _ = decode(forward(model, x.to(dev)).float())
            e = torch.linalg.norm(p.cpu() - kp, dim=-1)
            errs.append(e[m > 0])
    model.train()
    e = torch.cat(errs)
    return float(e.median()), float(e.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--views", default="top,side")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="base", help="ViTPose+ size")
    ap.add_argument("--mode", choices=["head", "full"], default="full")
    ap.add_argument("--mask", choices=["vis", "has"], default="vis")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--label-shift-ms", type=int, default=0)
    ap.add_argument("--traj-weight", type=float, default=0.0, help="weight of the trajectory label (labels_traj.py)")
    ap.add_argument("--traj-mode", choices=["fill", "replace"], default="fill")
    ap.add_argument("--label", default=None, help="model label in the result tables")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 1)
    ap.add_argument("--max-steps", type=int, default=None, help="testing")
    args = ap.parse_args()
    torch.manual_seed(2027)
    np.random.seed(2027)
    dev = "cuda"
    odir = OUT / args.run
    odir.mkdir(parents=True, exist_ok=True)
    sp = splits()
    val_p = sp["folds"][0]
    train_p = [p for f in sp["folds"][1:] for p in f]
    if args.n_train:
        g = manifest().drop_duplicates("pid").set_index("pid").group
        rng = np.random.default_rng(2027)
        td = rng.permutation([p for p in train_p if g[p] == "TD"]).tolist()
        ucp = rng.permutation([p for p in train_p if g[p] == "UCP"]).tolist()
        n_u = max(1, round(args.n_train * len(ucp) / len(train_p)))
        train_p = td[:args.n_train - n_u] + ucp[:n_u]
    views = args.views.split(",")
    shift = f"kp_shift{args.label_shift_ms:+d}" if args.label_shift_ms else None
    tr = Frames(clip_files(train_p, views), True, args.mask, shift, (args.traj_weight, args.traj_mode))
    va = Frames(clip_files(val_p, views), False, "vis")
    va.items = va.items[::max(1, len(va.items) // 4000)]
    cfg = dict(vars(args), train_participants=train_p, val_participants=val_p, n_train_frames=len(tr), n_val_frames=len(va))
    (odir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: v for k, v in cfg.items() if k != "train_participants"}), flush=True)
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, 128, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = build_model(args.backbone).to(dev)
    head = list(model.head.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    scaler = torch.cuda.amp.GradScaler()
    total = EPOCHS_HEAD + EPOCHS_FULL
    steps_per = len(tl)
    log, best, step = [], np.inf, 0
    for ep in range(total):
        full = args.mode == "full" and ep >= EPOCHS_HEAD
        for p in body:
            p.requires_grad_(full)
        if ep == 0 or (full and ep == EPOCHS_HEAD):
            groups = [dict(params=head, lr=5e-4)] + ([dict(params=body, lr=1e-5)] if full else [])
            opt = torch.optim.AdamW(groups, weight_decay=1e-4)
            n_ep = EPOCHS_HEAD if (args.mode == "full" and not full) else (total - ep)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / (n_ep * steps_per), 1))))
        t0, losses = time.time(), []
        for x, hm, m, _ in tl:
            x, hm, m = x.to(dev, non_blocking=True), hm.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = forward(model, x)
            loss = (((pred.float() - hm) ** 2).mean((2, 3)) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        med, mean = evaluate(model, vl, dev)
        log.append(dict(epoch=ep, stage="full" if full else "head", train_loss=np.mean(losses), val_median_px=med,
                        val_mean_px=mean, minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
        pd.DataFrame(log).to_csv(odir / "log.csv", index=False)
        if med < best:
            best = med
            torch.save(model.state_dict(), odir / "model.pt")
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[{args.run}] best validation median {best:.2f} crop px", flush=True)


if __name__ == "__main__":
    main()
