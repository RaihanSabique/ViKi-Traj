"""Marker labels and crops for ViconTraj-MS fine-tuning, one file per valid clip with a calibrated camera.

Every blob frame (every third frame of the movement window) is cropped around the participant box; joints are the 6 Hz
low-passed VICON proxies projected with the video's camera. Writes outputs/benchmark/labels/<pid>/<view>/<clip>.npz:
    frame (N,)                  source-video frame index
    jpg, jpg_off                JPEG crops, concatenated: crop n = jpg[jpg_off[n]:jpg_off[n + 1]] (W x H = 192 x 256)
    M (N, 2, 3)                 affine image -> crop (RT-DETR participant box padded by 1.25, as for RTMPose)
    kp (N, 10, 2)               joint set in crop px: L shoulder, elbow, wrist, hand, fingertip, then R
    vis (N, 10)                 1 = every marker of the joint has a white blob within VIS_PX of its projection
    has (N, 10)                 1 = the reference exists (VICON markers present)
    mk (N, 24, 2), mvis, mhas   the 24 markers in MARKERS order: crop px, blob within VIS_PX, position present
    kp_img (N, 10, 2)           joint set in image px

    python benchmark/labels.py --participant PID
    python benchmark/labels.py --participant PID --size 768x1024 --every 3 --out labels_hr      (Sapiens2 crops)
"""
import argparse

import cv2
import numpy as np

from benchlib import BENCH, JOINT_NAMES, JOINTS, ROOT, VIEWS, aux_file, clip_stem, frame_clock, markers, valid_rows
from calibrate import camera_id, interp_markers
from evaluate import at, cameras, ref_track
from vicon_eval import project

import trial_models as tm
from common import box_to_affine

VIS_PX = 8.0                  # px
# canonical order of the 24 markers (a recording can miss a marker: NaN there)
MARKERS = ['LFHD', 'RFHD', 'LBHD', 'C7', 'T10', 'CLAV', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB',
           'LHND', 'LFIN', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'RHND', 'RFIN', 'RBHD', 'RFRM', 'RUPA']
W, H = 192, 256               # crop size (--size)
LAB = BENCH / "labels"        # output directory (--out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--size", default=None, help="crop WxH, default 192x256")
    ap.add_argument("--out", default=None, help="output directory name under outputs/benchmark, default labels")
    ap.add_argument("--every", type=int, default=1, help="keep every N-th label frame")
    args = ap.parse_args()
    global W, H, LAB
    if args.size:
        W, H = (int(x) for x in args.size.lower().split("x"))
    if args.out:
        LAB = BENCH / args.out
    pid = args.participant
    cams = cameras()
    vic = markers(pid)[0]
    n_clips = n_frames = 0
    for view in VIEWS:
        rows = valid_rows(view, [pid])
        for video, g in rows.groupby(f"{view}_source_video"):
            cid = camera_id(pid, int(g.visit.iloc[0]), view, video)
            if cid not in cams:
                continue
            P = cams[cid]
            cap = cv2.VideoCapture(str(ROOT / video))
            for _, r in g.iterrows():
                stem = clip_stem(r, view)
                out = LAB / pid / view / f"{stem}.npz"
                if out.exists():
                    continue
                bz, lz = np.load(aux_file("boxes", pid, view, stem)), np.load(aux_file("blobs", pid, view, stem))
                bfr, boxes = bz["frame"], bz["box"]
                sel = lz["frame"][::args.every]                                # blob frames: every third of the window
                bxy = np.split(lz["xy"], np.cumsum(lz["n"])[:-1])[::args.every]
                v = vic[int(r.aligned_mat_row)]
                t = frame_clock(pid, int(r.aligned_mat_row), view, sel)
                ref = np.stack([at(v["t"], ref_track(v, s, j), t) for s in "LR" for j in JOINT_NAMES], 1)  # (N,10,3)
                Xr = interp_markers(v, t)                                      # (N, M, 3) raw markers
                Xm = np.full((len(t), len(MARKERS), 3), np.nan)
                for c, name in enumerate(MARKERS):
                    if name in v["labels"]:
                        Xm[:, c] = Xr[:, v["labels"].index(name)]
                frames, _ = tm.read_range(cap, int(sel[0]), int(sel[-1]) + 1)
                rec = dict(frame=[], jpg=[], M=[], kp=[], vis=[], has=[], mk=[], mvis=[], mhas=[], kp_img=[])
                for i, f in enumerate(sel):
                    k = int(f - sel[0])
                    b = boxes[np.searchsorted(bfr, f)] if f <= bfr[-1] else boxes[-1]
                    if k >= len(frames) or np.isnan(b).any():
                        continue
                    M = box_to_affine(b, W, H, pad=1.25)
                    crop = cv2.warpAffine(frames[k], M, (W, H), flags=cv2.INTER_LINEAR)
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    has = ~np.isnan(ref[i]).any(1)
                    kp_img = np.full((10, 2), np.nan)
                    kp_img[has] = project(P, ref[i][has])
                    mh = ~np.isnan(Xm[i]).any(1)
                    mk_img = np.full((24, 2), np.nan)
                    mk_img[mh] = project(P, Xm[i][mh])
                    near = np.zeros(24, bool)
                    if len(bxy[i]) and mh.any():
                        near[mh] = np.linalg.norm(mk_img[mh][:, None] - bxy[i][None], axis=2).min(1) <= VIS_PX
                    vis = np.array([all(near[MARKERS.index(s + n)] for n in JOINTS[j]) for s in "LR" for j in JOINT_NAMES])
                    to_crop = lambda p: p @ M[:, :2].T + M[:, 2]
                    rec["frame"].append(f); rec["jpg"].append(buf.tobytes()); rec["M"].append(M)
                    rec["kp"].append(to_crop(kp_img)); rec["vis"].append(vis & has); rec["has"].append(has)
                    rec["mk"].append(to_crop(mk_img)); rec["mvis"].append(near); rec["mhas"].append(mh)
                    rec["kp_img"].append(kp_img)
                if not rec["frame"]:
                    continue
                off = np.cumsum([0] + [len(j) for j in rec["jpg"]])
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez(tmp, frame=np.array(rec["frame"]), jpg=np.frombuffer(b"".join(rec["jpg"]), np.uint8),
                         jpg_off=off, M=np.array(rec["M"], np.float32), kp=np.array(rec["kp"], np.float32),
                         vis=np.array(rec["vis"]), has=np.array(rec["has"]), mk=np.array(rec["mk"], np.float32),
                         mvis=np.array(rec["mvis"]), mhas=np.array(rec["mhas"]), kp_img=np.array(rec["kp_img"], np.float32))
                tmp.replace(out)
                n_clips += 1
                n_frames += len(rec["frame"])
            cap.release()
    print(f"[{pid}] {n_clips} clips, {n_frames} labelled frames", flush=True)


if __name__ == "__main__":
    main()
