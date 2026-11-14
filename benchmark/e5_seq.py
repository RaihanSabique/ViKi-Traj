"""ViconTraj-MS temporal head (seq2d) and two-view lifting network (lift3d) on the keypoints of a pose model.

Both train a residual TCN that adds a correction to its input (folds 2-5 train, fold 1 selects the epoch). seq2d
corrects the 2D keypoints of each view and writes the test clips as the model seq_<base>; lift3d corrects the
triangulated joints of e4_kinematics.py and writes results/E5_lift3d_<base>.csv and E5_lift3d_<base>_participants.csv.
Weights and logs: outputs/benchmark/finetune/seq_<base>_<view>/ and lift_<mlp|tcn>_<base>/.

    python benchmark/e5_seq.py --part seq2d --base sapiens2_1b
    python benchmark/e5_seq.py --part lift3d --base sapiens2_1b
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from benchlib import BENCH, JOINT_NAMES, aux_file, clip_stem, frame_clock, markers, model_file, \
    model_joints, set_lr_swap, splits, valid_rows
from calibrate import camera_id
from evaluate import RES, at, boot_ci, cameras, ref_track
from vicon_eval import project

OUT = BENCH / "finetune"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 64                                                    # frames per training window


class TCN(nn.Module):
    """Residual dilated 1D convolutions over (B, C, T); temporal=False sets every kernel to 1 (per frame)."""
    def __init__(self, cin, cout, ch=128, k=5, dil=(1, 2, 4), temporal=True):
        super().__init__()
        kk = k if temporal else 1
        self.inp = nn.Conv1d(cin, ch, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(ch, ch, kk, padding=(kk // 2) * d, dilation=d), nn.GELU(),
                                                   nn.Conv1d(ch, ch, 1)) for d in dil])
        self.out = nn.Conv1d(ch, cout, 1)

    def forward(self, x):                                   # (B, C, T)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(torch.nn.functional.gelu(h))


def seq_loss(pred, tgt, m):
    """pred, tgt (B, D, T); m (B, D, T) mask. L1 position + velocity + 0.1 jerk."""
    l = (torch.abs(pred - tgt) * m).sum() / m.sum().clamp(min=1)
    dv = lambda x: x[..., 1:] - x[..., :-1]
    mv = m[..., 1:] * m[..., :-1]
    l = l + (torch.abs(dv(pred) - dv(tgt)) * mv).sum() / mv.sum().clamp(min=1)
    mj = mv[..., 2:] * mv[..., 1:-1] * mv[..., :-2]
    l = l + 0.1 * (torch.abs(dv(dv(dv(pred))) - dv(dv(dv(tgt)))) * mj).sum() / mj.sum().clamp(min=1)
    return l


def windows(seqs, n, rng):
    """Random WIN-long windows from a list of (x, y, m) arrays (T, D)."""
    lens = np.array([len(s[0]) for s in seqs])
    p = np.maximum(lens - WIN + 1, 0).astype(float)
    p /= p.sum()
    out = [[], [], []]
    for i in rng.choice(len(seqs), n, p=p):
        s0 = rng.integers(0, lens[i] - WIN + 1)
        for a in range(3):
            out[a].append(seqs[i][a][s0:s0 + WIN])
    return [torch.tensor(np.stack(o), dtype=torch.float32).permute(0, 2, 1) for o in out]


def train(seqs_tr, seqs_va, cin, cout, temporal, name, epochs=40, steps=200, batch=64):
    """Train a TCN to correct x[:, :cout] toward y; keeps the weights of the best validation L1 (finetune/<name>/)."""
    torch.manual_seed(2027)
    rng = np.random.default_rng(2027)
    net = TCN(cin, cout, temporal=temporal).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    va = windows(seqs_va, 512, np.random.default_rng(1))
    best, log = np.inf, []
    odir = OUT / name
    odir.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        net.train()
        for _ in range(steps):
            x, y, m = [a.to(DEV) for a in windows(seqs_tr, batch, rng)]
            loss = seq_loss(x[:, :cout] + net(x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            x, y, m = [a.to(DEV) for a in va]
            vl = float((torch.abs(x[:, :cout] + net(x) - y) * m).sum() / m.sum())
        log.append(dict(epoch=ep, train_loss=float(loss), val_l1=vl))
        if vl < best:
            best = vl
            torch.save(net.state_dict(), odir / "model.pt")
    pd.DataFrame(log).to_csv(odir / "log.csv", index=False)
    net.load_state_dict(torch.load(odir / "model.pt"))
    print(f"[{name}] best validation L1 {best:.4f}", flush=True)
    return net.eval()


# ----------------------------------------------------------------------------- seq2d
def clip_seq(r, view, base, cams):
    """(x (T, 30), y (T, 20), m (T, 20), box centre (T, 2), box size (T,), frame, fps) of one clip, None without
    base model output. Coordinates are (px - box centre) / box size, box size = max(width, height)."""
    pid, row = r.pid, int(r.aligned_mat_row)
    stem = clip_stem(r, view)
    f = model_file(base, pid, view, stem)
    if not f.exists():
        return None
    z = dict(np.load(f))
    set_lr_swap({base: z}, pid, view, r[f"{view}_source_video"])
    bz = np.load(aux_file("boxes", pid, view, stem))
    c = (bz["box"][:, :2] + bz["box"][:, 2:]) / 2
    sz = np.maximum(bz["box"][:, 2] - bz["box"][:, 0], bz["box"][:, 3] - bz["box"][:, 1])
    kp = np.concatenate([model_joints(z, base, s, view) for s in "LR"], 1)                  # (T, 10, 2)
    xn = (kp - c[:, None]) / sz[:, None, None]
    pres = ~np.isnan(xn).any(2)
    x = np.concatenate([np.nan_to_num(xn).reshape(len(kp), -1), pres.astype(np.float32)], 1)
    y = np.zeros((len(kp), 20), np.float32)
    m = np.zeros((len(kp), 20), np.float32)
    cid = camera_id(pid, int(r.visit), view, r[f"{view}_source_video"])
    if cid in cams:
        v = markers(pid)[0][row]
        t = frame_clock(pid, row, view, z["frame"])
        for a, (s, j) in enumerate([(s, j) for s in "LR" for j in JOINT_NAMES]):
            ref = at(v["t"], ref_track(v, s, j), t)
            ok = ~np.isnan(ref).any(1)
            p = np.full((len(t), 2), np.nan)
            p[ok] = project(cams[cid], ref[ok])
            pn = (p - c) / sz[:, None]
            y[:, 2 * a:2 * a + 2] = np.nan_to_num(pn)
            m[:, 2 * a:2 * a + 2] = ok[:, None]
    return x.astype(np.float32), y, m, c, sz, z["frame"], z["fps"]


def part_seq2d(base):
    cams = cameras()
    sp = splits()
    val_p, train_p, test_p = sp["folds"][0], [p for f in sp["folds"][1:] for p in f], sp["test"]
    name = f"seq_{base}"
    for view in ("top", "side"):
        rows = valid_rows(view)
        data = {}
        for _, r in rows.iterrows():
            s = clip_seq(r, view, base, cams)
            if s is not None:
                data[(r.pid, int(r.aligned_mat_row))] = (r, s)
        tr = [s[:3] for (p, _), (r, s) in data.items() if p in train_p and len(s[0]) > WIN]
        va = [s[:3] for (p, _), (r, s) in data.items() if p in val_p and len(s[0]) > WIN]
        net = train(tr, va, 30, 20, True, f"{name}_{view}")
        with torch.no_grad():
            for (p, row), (r, s) in data.items():
                if p not in test_p:
                    continue
                x, _, _, c, sz, fr, fps = s
                pred = (torch.tensor(x).T[None].to(DEV)[:, :20] + net(torch.tensor(x).T[None].to(DEV)))[0].T.cpu().numpy()
                k = pred.reshape(-1, 10, 2) * sz[:, None, None] + c[:, None]
                out = model_file(name, p, view, clip_stem(r, view))
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, frame=fr, fps=fps, k=k.astype(np.float32), s=np.ones(k.shape[:2], np.float32))
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "config.json").write_text(json.dumps(dict(run=name, base=base, mode="sequential", views="top,side",
                                                            label=f"{KP_LABEL(base)} + sequential head (E5.4)")))


def KP_LABEL(base):
    """Display name of a model (benchlib.MODEL_LABEL)."""
    from benchlib import MODEL_LABEL
    return MODEL_LABEL.get(base, base)


# ----------------------------------------------------------------------------- lift3d
def path_seq(z, base):
    """One trial (path) of the track cache: x (T, 16), the base model's 3D joints in metres relative to its mean
    shoulder, then presence flags; y, m (T, 12): VICON target and mask; the shoulder anchor (mm). None when the base
    has no shoulder in the trial."""
    X, V = z[base], z["vicon"]                                    # (4, T, 3)
    anchor = np.nanmean(X[0], 0)
    if np.isnan(anchor).any():
        return None
    xr = (X - anchor).transpose(1, 0, 2) / 1000.0                  # metres
    yr = (V - anchor).transpose(1, 0, 2) / 1000.0
    pres = ~np.isnan(xr).any(2)
    x = np.concatenate([np.nan_to_num(xr).reshape(len(xr), -1), pres], 1).astype(np.float32)
    m = np.repeat(~np.isnan(yr).any(2), 3, 1).astype(np.float32)
    return x, np.nan_to_num(yr).reshape(len(yr), -1).astype(np.float32), m, anchor


def part_lift3d(base):
    from e4_kinematics import TRK, J4
    sp = splits()
    val_p, train_p, test_p = sp["folds"][0], [p for f in sp["folds"][1:] for p in f], set(sp["test"])
    seqs = {}
    for f in sorted(TRK.glob("*.npz")):
        z = np.load(f)
        if base not in z.files:
            continue
        pid = f.stem.split("_r")[0]
        s = path_seq(z, base)
        if s is not None:
            seqs[f.stem] = (pid, s, z[base], z["vicon"])
    tr = [s[:3] for p, s, *_ in seqs.values() if p in train_p and len(s[0]) > WIN]
    va = [s[:3] for p, s, *_ in seqs.values() if p in val_p and len(s[0]) > WIN]
    rec = []
    nets = {kind: train(tr, va, 16, 12, kind == "tcn", f"lift_{kind}_{base}") for kind in ("mlp", "tcn")}
    with torch.no_grad():
        for key, (pid, (x, y, m, anchor), Xb, V) in seqs.items():
            if pid not in test_p:
                continue
            xt = torch.tensor(x).T[None].to(DEV)
            ests = {"triangulation": Xb}
            for kind, net in nets.items():
                pr = (xt[:, :12] + net(xt))[0].T.cpu().numpy().reshape(-1, 4, 3) * 1000.0 + anchor
                pr[~(x[:, 12:] > 0)] = np.nan                       # no estimate where the base had none
                ests[kind] = pr.transpose(1, 0, 2)
            for meth, E in ests.items():
                for a, j in enumerate(J4):
                    e = np.linalg.norm(E[a] - V[a], axis=1)
                    ok = ~np.isnan(V[a]).any(1)
                    rec.append(dict(pid=pid, path=key, method=meth, joint=j, err=e[ok]))
    # per participant: pooled samples. pid: 7-character prefix + study number; pid[7] is its first digit
    # (1xx = TD, 2xx = UCP)
    rows = []
    for (pid, meth, j), g in pd.DataFrame(rec).groupby(["pid", "method", "joint"]):
        e = np.concatenate(g.err.values)
        got = ~np.isnan(e)
        rows.append(dict(pid=pid, group="UCP" if pid[7] == "2" else "TD", method=meth, joint=j, n=len(e), coverage=got.mean(),
                         median_mm=np.median(e[got]) if got.any() else np.nan,
                         p90_mm=np.percentile(e[got], 90) if got.any() else np.nan))
    p = pd.DataFrame(rows)
    out = []
    for group in ("all", "TD", "UCP"):
        q = p if group == "all" else p[p.group == group]
        for (meth, j), g in q.groupby(["method", "joint"]):
            lo, hi = boot_ci(g.median_mm)
            out.append(dict(base=base, group=group, method=meth, joint=j, n_participants=len(g), median_mm=g.median_mm.mean(),
                            ci_lo=lo, ci_hi=hi, p90_mm=g.p90_mm.mean(), coverage=g.coverage.mean()))
    pd.DataFrame(out).to_csv(RES / f"E5_lift3d_{base}.csv", index=False, float_format="%.4f")
    p.to_csv(RES / f"E5_lift3d_{base}_participants.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(out).query("group == 'all'").pivot_table(index="method", columns="joint", values="median_mm").round(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["seq2d", "lift3d"], required=True)
    ap.add_argument("--base", default="sapiens2_1b")
    args = ap.parse_args()
    (part_seq2d if args.part == "seq2d" else part_lift3d)(args.base)


if __name__ == "__main__":
    main()
