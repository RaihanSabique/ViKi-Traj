"""ViTMS for Sapiens2 (0.4B, 1B): the pose head is trained on the joint set, the backbone stays frozen.

Reads the 768 x 1024 crops of every 9th frame in outputs/benchmark/labels_hr (labels.py --size 768x1024 --every 3
--out labels_hr, then labels_traj.py --labels labels_hr). Splits, loss weights and augmentation as in finetune.py,
heatmap targets from the model's UDP codec. Writes head.pt (decode head of the best validation epoch), log.csv and
config.json to outputs/benchmark/finetune/<run>/.

    python benchmark/finetune_sapiens.py --run ft_sapiens2_0.4b_head --size 0.4b --batch 8 --epochs 5
    python benchmark/finetune_sapiens.py --run ft_sapiens2_1b_head --size 1b --batch 8 --epochs 4 --train-every 2
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from benchlib import BENCH, KP, ROOT, splits
from finetune import FLIP, K, OUT

W, H = 768, 1024
HM_W, HM_H, SIGMA = 192, 256, 6.0
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
KEEP = KP["sapiens2_1b"]["L"] + KP["sapiens2_1b"]["R"]              # Sapiens2 channels of the joint set, in its order
POSE = ROOT / "third_party" / "sapiens2" / "sapiens" / "pose"


def sapiens_model(size, device="cuda"):
    """Pretrained Sapiens2 pose model of this size and the UDP codec of its heatmaps."""
    sys.path.insert(0, str(ROOT / "third_party" / "sapiens2"))
    from sapiens.pose.datasets import UDPHeatmap
    from sapiens.pose.models import init_model
    cfg = POSE / f"configs/keypoints308/shutterstock_goliath_3po/sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py"
    m = init_model(str(cfg), str(ROOT / "models" / "sapiens2" / "pose" / f"sapiens2_{size}_pose.safetensors"), device=device)
    return m, UDPHeatmap(input_size=(W, H), heatmap_size=(HM_W, HM_H), sigma=SIGMA)


class SapFrames(Dataset):
    """As finetune.Frames, with the targets encoded by the Sapiens2 codec; every: use every N-th frame of a clip."""
    def __init__(self, files, augment, codec, traj=(0.0, "fill"), every=1):
        self.items, self.data = [], []
        self.tw, self.tmode, self.codec = traj[0], traj[1], codec
        for f in files:
            z = np.load(f)
            kp, m = z["kp"], z["vis"]
            if self.tw > 0 and "kp_traj" in z:
                kt, side = z["kp_traj"], int(z["side_traj"])
            else:
                kt, side = np.full((len(kp), 2), np.nan, np.float32), 0
            self.data.append((z["jpg"], z["jpg_off"], kp, m, kt, side))
            self.items += [(len(self.data) - 1, i) for i in range(0, len(z["frame"]), every)
                           if m[i].any() or (self.tw > 0 and not np.isnan(kt[i]).any())]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        c, i = self.items[n]
        jpg, off, kp, m, kt, side = self.data[c]
        img = cv2.cvtColor(cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        kp, m = np.r_[kp[i], kt[i][None]].copy(), m[i].copy()
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
        x = ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
        inside = (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H) & ~np.isnan(kp).any(1)
        w = (m & inside[:K]).astype(np.float32)
        j = 4 + 5 * side                                                   # the active hand's fingertip channel
        if self.tw > 0 and inside[K] and (w[j] == 0 or self.tmode == "replace"):
            kp[j] = kp[K]
            w[j] = self.tw if w[j] == 0 else max(w[j], self.tw)
        enc = self.codec.encode(np.nan_to_num(kp[:K])[None].astype(np.float32), (w > 0).astype(np.float32)[None])
        hm = enc["heatmaps"].astype(np.float32)                                   # (K, HM_H, HM_W)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(hm), torch.from_numpy(w), \
            torch.from_numpy(np.nan_to_num(kp[:K].astype(np.float32), nan=-1))


def clip_files(lab, pids, views):
    return [f for p in pids for v in views for f in sorted((lab / p / v).glob("*.npz")) if ".tmp" not in f.name]


def heatmaps(model, x, train_head):
    """Heatmaps of the ten joint-set channels; the backbone runs without gradients, the head with them if train_head."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        feat = model.backbone(x)[0]
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.set_grad_enabled(train_head):
            return model.decode_head(feat)[:, KEEP]


def evaluate(model, codec, loader):
    """Median and mean error (crop px) of the labelled joints of the loader, UDP decoding."""
    model.decode_head.eval()
    errs = []
    for x, hm, w, kp in loader:
        pred = heatmaps(model, x.to("cuda", non_blocking=True), False).float().cpu().numpy()
        for n in range(len(pred)):
            p, _ = codec.decode(pred[n])
            e = np.linalg.norm(p[0] - kp[n].numpy(), axis=1)
            errs.append(e[w[n].numpy() > 0])
    model.decode_head.train()
    e = np.concatenate(errs)
    return float(np.median(e)), float(e.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--size", choices=["0.4b", "1b"], default="1b")
    ap.add_argument("--labels", default="labels_hr")
    ap.add_argument("--traj-weight", type=float, default=1.0)
    ap.add_argument("--traj-mode", choices=["fill", "replace"], default="fill")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 1)
    ap.add_argument("--train-every", type=int, default=1, help="use every N-th training frame")
    ap.add_argument("--max-steps", type=int, default=None, help="testing")
    args = ap.parse_args()
    torch.manual_seed(2027)
    np.random.seed(2027)
    lab = BENCH / args.labels
    odir = OUT / args.run
    odir.mkdir(parents=True, exist_ok=True)
    sp = splits()
    val_p, train_p = sp["folds"][0], [p for f in sp["folds"][1:] for p in f]
    model, codec = sapiens_model(args.size)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.eval()
    model.decode_head.train()
    views = ["top", "side"]
    tr = SapFrames(clip_files(lab, train_p, views), True, codec, (args.traj_weight, args.traj_mode), every=args.train_every)
    va = SapFrames(clip_files(lab, val_p, views), False, codec, every=10)          # every 10th validation frame
    cfg = dict(vars(args), backbone=f"sapiens2_{args.size}", mode="head", views="top,side", mask="vis", kp_layout="sapiens2",
               label=f"Sapiens2-{args.size.upper()} FT head" + (", trajectory" if args.traj_weight else ""),
               train_participants=train_p, val_participants=val_p, n_train_frames=len(tr), n_val_frames=len(va))
    (odir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: v for k, v in cfg.items() if k != "train_participants"}), flush=True)
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, args.batch * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    head = [p for p in model.decode_head.parameters()]
    opt = torch.optim.AdamW(head, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / max(total_steps, 1), 1))))
    scaler = torch.cuda.amp.GradScaler()
    log, best, step = [], np.inf, 0
    for ep in range(args.epochs):
        t0, losses = time.time(), []
        for x, hm, w, _ in tl:
            x, hm, w = x.to("cuda", non_blocking=True), hm.to("cuda", non_blocking=True), w.to("cuda", non_blocking=True)
            pred = heatmaps(model, x, True)
            loss = (((pred.float() - hm) ** 2).mean((2, 3)) * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                print(f"epoch {ep} step {step} loss {np.mean(losses[-200:]):.5f} {(time.time() - t0) / 60:.1f} min", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        med, mean = evaluate(model, codec, vl)
        log.append(dict(epoch=ep, train_loss=np.mean(losses), val_median_px=med, val_mean_px=mean, minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
        pd.DataFrame(log).to_csv(odir / "log.csv", index=False)
        if med < best:
            best = med
            torch.save(model.decode_head.state_dict(), odir / "head.pt")
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[{args.run}] best validation median {best:.2f} crop px (768 x 1024 crops)", flush=True)


if __name__ == "__main__":
    main()
