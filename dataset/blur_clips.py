"""Per-trial video clips of the release: side view with heads and laptop screens blurred, top view unblurred.

For every run and view with an aligned clip (benchmark manifest <view>_clip_path) it writes
<out>/sourcedata/videos/<sub>/<ses>/<sub>_<ses>_task-reaching_acq-<view>_run-NN_video.mp4, and per participant a QA
table and JPEGs of the strongest faces found after blurring under <qa>/ (release/private/blur_qa for the release root,
else <out>/private/blur_qa; never released). The side view needs a GPU for the person detector (RT-DETR).

    python dataset/blur_clips.py --participant XXXXXXX110 --out release/test/blur [--views top,side] [--runs 3 7]
    python dataset/blur_clips.py --all [--views top] [--skip-existing]
"""
import importlib
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as rc                                                     # noqa: E402

POSE_HZ = 10                  # detector + pose rate: every 3rd frame at 30 fps
DET_BATCH = 16
HEAD_KP = list(range(5)) + list(range(23, 91))     # COCO-WholeBody nose, eyes, ears, 68 face points
KP_THR = np.r_[np.full(5, 0.3), np.full(68, 0.5)]  # face points of a head seen from behind score 0.3-0.5 all over it
HAND_KP, HAND_THR = [9, 10] + list(range(91, 133)), 0.3   # wrists and the 42 hand points
BOX_SURE = 0.7                # RT-DETR scores its "person" boxes on the game screen and on a hand 0.3-0.5
HEAD_SURE_N, HEAD_SURE = 3, 0.5   # strong head: at least 3 of nose, eyes, ears at >= 0.5
YUNET_LOW, YUNET_SURE, QA_THR = 0.3, 0.9, 0.5    # game-screen false positives score up to 0.90
QA_STEP = 5                   # QA: every 5th frame of the written video
HOLD_FRAMES, HOLD_S = 3, 0.1  # a region is kept over +-max(3 frames, 0.1 s)
R_SCALE, R_MIN, R_MAX, R_ASPECT = 1.4, 0.08, 0.2, 1.15   # blur ellipse size (ellipse())
CHIN_DROP_MM, HEAD_SPAN_MM, VICON_TOL_S = 100.0, 200.0, 0.5
MIN_IN_FRONT = 0.99           # usable camera (as build_frame_tables): share of the VICON samples in front of it
ARM_MARKERS = [s + m for s in "LR" for m in ("SHO", "UPA", "ELB", "FRM", "WRA", "WRB", "HND", "FIN")]
OFF_HEAD = 1.5                # a hand point further than 1.5 VICON head radii from the head is off the child's head
BLOCKS_PER_WIDTH, FEATHER_PX = 40, 5   # pixel block = 1/40 of the image width; mask edge feather sigma (px)
CRF = 20                      # libx264 quality


# ----------------------------------------------------------------------------- benchmark and src modules
@contextmanager
def benchmark_namespace():
    """Import context for benchmark/ and src/ modules: both ahead of dataset/ on sys.path (benchmark/ first, as
    benchlib expects) and dataset/common.py out of sys.modules, so that trial_overlays and trial_models (imported by
    zero_shot) import src/common.py as `common`."""
    saved_path, saved_common = sys.path[:], sys.modules.pop("common")
    sys.path[:0] = [str(rc.ROOT / "benchmark"), str(rc.ROOT / "src")]
    try:
        yield
    finally:
        sys.path[:] = saved_path
        sys.modules["common"] = saved_common


with benchmark_namespace():
    evaluate = importlib.import_module("evaluate")            # benchmark/evaluate.py: cameras()
    calibrate = importlib.import_module("calibrate")          # camera_id()
    vicon_eval = importlib.import_module("vicon_eval")        # project()
    trial_overlays = importlib.import_module("trial_overlays")  # FaceBlur (YuNet)
    zero_shot = importlib.import_module("zero_shot")          # RTMPoseCPU, trial_models.Detector


BLUR_VIEWS = ("side",)       # views whose heads are blurred; the top camera films the participant from behind


class Models:
    """Person detector (GPU), whole-body pose (ONNX, CPU), YuNet at the detection and the QA threshold, cameras.
    detect=False (only unblurred views): YuNet for the QA only."""
    def __init__(self, detect=True):
        self.det = zero_shot.tm.Detector() if detect else None
        self.pose = zero_shot.RTMPoseCPU() if detect else None
        self.yunet = trial_overlays.FaceBlur().det
        self.yunet.setScoreThreshold(YUNET_LOW)
        self.yunet_qa = trial_overlays.FaceBlur().det
        self.yunet_qa.setScoreThreshold(QA_THR)
        self.cams = evaluate.cameras()
        self.usable = {}

    def camera(self, row, view):
        """Signed projection matrix of the run's video of this view when the camera is usable, else None."""
        if row.pid not in self.usable:
            self.usable[row.pid] = usable_cameras(self.cams, row.pid)
        return self.usable[row.pid].get(video_camera(row, view))


# ----------------------------------------------------------------------------- cameras
def video_camera(row, view):
    """Camera id of the run's video of this view (None without a video)."""
    video = row[f"{view}_source_video"]
    return calibrate.camera_id(row.pid, int(row.visit), view, video) if isinstance(video, str) else None


def depth(P, X):
    """Projective depth of points (N, 3) under P (sign = side of the focal plane)."""
    return X @ P[2, :3] + P[2, 3]


def usable_cameras(cams, pid):
    """{camera id: P} of the participant's calibrated top / side videos that build_frame_tables also uses: P signed so
    that depth > 0 on the side of most of the VICON marker samples of the video's trials, and at least MIN_IN_FRONT of
    them on that side. A calibration fitted to another person's blobs (an adult) can pass the reprojection test with
    the child behind the focal plane; its head projection would land on that person."""
    vic, samples = rc.benchlib.markers(pid)[0], {}
    for _, r in manifest().query("pid == @pid").iterrows():
        for view in ("top", "side"):
            cid, v = video_camera(r, view), vic.get(int(r.aligned_mat_row))
            if cid in cams and v is not None:
                X = v["X"].reshape(-1, 3)
                samples.setdefault(cid, []).append(X[np.isfinite(X).all(1)])
    usable = {}
    for cid, xs in samples.items():
        d = depth(cams[cid], np.concatenate(xs))
        front = np.sign(np.median(d))
        if np.mean(d * front > 0) >= MIN_IN_FRONT:
            usable[cid] = cams[cid] * front
    return usable


def project(P, X):
    """(..., 3) mm -> (..., 2) px; NaN for missing points and points behind the camera (P signed as usable_cameras)."""
    flat = X.reshape(-1, 3)
    xy = vicon_eval.project(P, flat)
    xy[~(depth(P, flat) > 0)] = np.nan
    return xy.reshape(X.shape[:-1] + (2,))


# ----------------------------------------------------------------------------- clips
@dataclass
class Clip:
    row: pd.Series            # manifest row
    view: str
    src: Path                 # aligned clip
    dst: Path                 # released video
    n_frames: int             # <view>_n_frames: frames of the movement window
    start_frame: int          # source-video frame of output frame 0 (negative: black lead)
    fmap: np.ndarray          # output frame -> clip frame (-1: black)
    fps: Fraction
    size: tuple               # (W, H)

    @property
    def name(self):
        return self.dst.name


def manifest():
    """common.manifest() with trial_in_visit (game round) filled where it is missing (four trials of one visit without
    VICON data) by the dense rank of group_no within the visit, which equals trial_in_visit on every other row."""
    m = rc.manifest()
    rank = m.groupby(["pid", "visit"]).group_no.rank(method="dense")
    m["trial_in_visit"] = m.trial_in_visit.fillna(rank)
    return m


def clips(pid, out, views, runs=None):
    """Every (run, view) of the participant whose aligned clip exists, in manifest order. Output frame k shows source
    frame <view>_start_frame + k (frame 0 = movement start); frames before the source video starts (negative start
    frame) or after it ends are black."""
    m = manifest()
    res = []
    for _, r in m[m.pid == pid].iterrows():
        sub, ses, run = rc.row_ids(r)
        if runs and rc.run_index(r.trial_in_visit, r.path) not in runs:
            continue
        for view in views:
            src = r[f"{view}_clip_path"]
            if not isinstance(src, str) or not (rc.ROOT / src).exists():
                continue
            cap = cv2.VideoCapture(str(rc.ROOT / src))
            n_clip, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            n, s0 = int(r[f"{view}_n_frames"]), int(r[f"{view}_start_frame"])
            idx = np.arange(n) + s0 - max(s0, 0)              # the clip starts at source frame max(s0, 0)
            dst = rc.subses_dir(out / rc.VIDEOS, sub, ses) / rc.bids_name(sub, ses, "video", ".mp4", acq=view, run=run)
            res.append(Clip(r, view, rc.ROOT / src, dst, n, s0, np.where((idx >= 0) & (idx < n_clip), idx, -1),
                            Fraction(fps).limit_denominator(1001), size))
    return res


def read_frames(clip):
    """(output frame, image) for every output frame that has a clip frame, in order."""
    cap = cv2.VideoCapture(str(clip.src))
    for k in np.flatnonzero(clip.fmap >= 0):
        ok, img = cap.read()
        if not ok:
            break
        yield int(k), img
    cap.release()


# ----------------------------------------------------------------------------- regions
def ellipse(points, W, min_r=0.0):
    """Blur ellipse (cx, cy, rx, ry) around image points, or None without a finite point: rx = R_SCALE x the largest
    point distance from the centre (at least min_r) within R_MIN-R_MAX x the image width W, ry = R_ASPECT x rx."""
    p = points[np.isfinite(points).all(1)]
    if not len(p):
        return None
    c = (p.min(0) + p.max(0)) / 2              # not the mean: the 68 face points would pull it off the head centre
    r = float(np.clip(max(R_SCALE * np.linalg.norm(p - c, axis=1).max(), min_r), R_MIN * W, R_MAX * W))
    return float(c[0]), float(c[1]), r, R_ASPECT * r


def diameter(points):
    """Largest distance between two of the points (0 for fewer than two)."""
    return float(np.max(np.linalg.norm(points[:, None] - points[None], axis=2))) if len(points) > 1 else 0.0


def inside(pt, e):
    return ((pt[0] - e[0]) / e[2]) ** 2 + ((pt[1] - e[1]) / e[3]) ** 2 <= 1


def grown(e, f):
    """The ellipse e with both radii scaled by f."""
    return e[0], e[1], f * e[2], f * e[3]


def markers_at(v, names, t):
    """(T, len(names), 3) VICON markers at game times t: linear over gaps, held up to VICON_TOL_S past the ends of
    the crop, NaN beyond and for absent markers."""
    out = np.full((len(t), len(names), 3), np.nan)
    for j, name in enumerate(names):
        x = v["X"][:, v["labels"].index(name)] if name in v["labels"] else np.zeros((0, 3))
        ok = np.isfinite(x).all(1)
        if ok.sum() < 2:
            continue
        tv = v["t"][ok]
        near = (t >= tv[0] - VICON_TOL_S) & (t <= tv[-1] + VICON_TOL_S)
        out[near, j] = np.stack([np.interp(t[near], tv, x[ok, d]) for d in range(3)], 1)
    return out


def head_points(head):
    """(T, 5, 3): the head markers without those further than HEAD_SPAN_MM from the head's median (mislabels),
    plus a virtual chin CHIN_DROP_MM below the front markers (VICON z is up)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                 # frames without any head marker
        med = np.nanmedian(head, 1, keepdims=True)
        head = np.where((np.linalg.norm(head - med, axis=2) > HEAD_SPAN_MM)[..., None], np.nan, head)
        front = np.nanmean(head[:, :2], 1)
        front = np.where(np.isfinite(front), front, np.nanmean(head, 1))
    return np.concatenate([head, (front - [0, 0, CHIN_DROP_MM])[:, None]], 1)


def vicon_view(clip, models):
    """Per output frame the ellipse of the projected head points (or None), and the projected arm markers
    (T, 16, 2) for the QA; ([None] * T, None) for a video without a usable camera or a trial without VICON markers.
    Markers are taken at the frame's game time (benchlib.frame_clock, the drift-corrected video clock)."""
    r, view = clip.row, clip.view
    none = [None] * clip.n_frames, None
    P = models.camera(r, view)
    v = rc.benchlib.markers(r.pid)[0].get(int(r.aligned_mat_row))
    if P is None or v is None:
        return none
    t = rc.benchlib.frame_clock(r.pid, int(r.aligned_mat_row), view, clip.start_frame + np.arange(clip.n_frames))
    heads = project(P, head_points(markers_at(v, rc.HEAD_MARKERS, t)))
    return [ellipse(p, clip.size[0]) for p in heads], project(P, markers_at(v, ARM_MARKERS, t))


@dataclass
class Person:
    box: np.ndarray           # x0, y0, x1, y1
    score: float              # RT-DETR
    head: tuple               # head ellipse or None
    strong_head: bool         # at least HEAD_SURE_N of nose, eyes, ears at >= HEAD_SURE
    hands: np.ndarray         # (N, 2) hand keypoints at >= HAND_THR

    @property
    def sure(self):
        return self.score >= BOX_SURE

    def contains(self, pt):
        return self.box[0] <= pt[0] <= self.box[2] and self.box[1] <= pt[1] <= self.box[3]


def find_persons(batch, models, W):
    """{output frame: [Person]} for every person in a batch of (output frame, image). The head radius is at least the
    largest nose / eye / ear distance (in profile the back of the head is that far away). Box size is not used:
    persons on a background monitor (0.10-0.19 of the image height) and game-screen boxes (up to 0.32) overlap."""
    found = models.det.people([img for _, img in batch])
    crops, owners = [], []
    for (k, img), (boxes, scores) in zip(batch, found):
        crops += [(img, b, s) for b, s in zip(boxes, scores)]
        owners += [k] * len(boxes)
    res = {k: [] for k, _ in batch}
    if not crops:
        return res
    kp, sc = models.pose([c[0] for c in crops], np.array([c[1] for c in crops], np.float32))
    for k, (_, box, score), p, s in zip(owners, crops, kp, sc):
        ok = s[HEAD_KP] >= KP_THR
        head = ellipse(p[HEAD_KP][ok], W, min_r=diameter(p[:5][ok[:5]]))
        res[k].append(Person(box, float(score), head, int((s[:5] >= HEAD_SURE).sum()) >= HEAD_SURE_N,
                             p[HAND_KP][s[HAND_KP] >= HAND_THR]))
    return res


def vicon_hands(arm, vic):
    """Per output frame the projected FIN and HND of both hands that lie off the child's head (further than OFF_HEAD x
    the VICON head ellipse; none in a frame without one); None without a VICON projection. Both hands: the manifest's
    hand is not the hand on the screen in some invalid trials."""
    if arm is None:
        return None
    idx = [ARM_MARKERS.index(s + m) for s in "LR" for m in ("FIN", "HND")]
    out = []
    for pts, e in zip(arm[:, idx], vic):
        pts = pts[np.isfinite(pts).all(1)] if e else np.zeros((0, 2))
        out.append(np.array([p for p in pts if not inside(p, grown(e, OFF_HEAD))]).reshape(-1, 2))
    return out


def hand_points(persons, vicon_hand):
    """Points of a hand that may be on the game screen: the VICON hands where the clip has a projection, else the hand
    keypoints of the confident persons."""
    if vicon_hand is not None:
        return vicon_hand
    pts = [p.hands for p in persons if p.sure]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def on_hand(e, hand):
    return any(inside(h, e) for h in hand)


def pose_heads(persons, hand):
    """Head ellipses of the persons; the head of a weak box (score < BOX_SURE) with weak head evidence is dropped when
    a hand point lies in it: RT-DETR boxes the game screen under the finger and the hand, and the pose model puts a
    head there."""
    return [p.head for p in persons if p.head and (p.sure or p.strong_head or not on_hand(p.head, hand))]


def face_regions(faces, persons, heads, hand, W):
    """Ellipses of the YuNet faces (x, y, w, h, score) that are kept: at >= YUNET_SURE anywhere, below that only inside
    a confident person whose head the pose model did not locate (then it is the only evidence of that head); and not
    when the face lies in no pose / VICON head region and a hand point lies in its ellipse (the game screen under the
    finger scores up to 0.90, the hand up to 0.8)."""
    res = []
    for d in faces:
        c = (d[0] + d[2] / 2, d[1] + d[3] / 2)
        if not (d[4] >= YUNET_SURE or any(p.sure and p.head is None and p.contains(c) for p in persons)):
            continue
        e = face_ellipse(d, W)
        if any(inside(c, h) for h in heads) or not on_hand(e, hand):
            res.append(e)
    return res


def face_ellipse(face, W):
    x, y, w, h = face[:4]
    return ellipse(np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]), W)


def scan(clip, models):
    """Pass 1: {source: [ellipses per output frame]} before the temporal hold, the most persons in a frame and the
    projected arm markers (or None). Sources: vicon (projected head markers and chin), pose (heads of the RT-DETR
    persons, POSE_HZ times a second) and face (YuNet on every frame)."""
    W, H = clip.size
    step = max(1, round(float(clip.fps) / POSE_HZ))
    vic, arm = vicon_view(clip, models)
    persons, faces, batch = {}, {}, []
    models.yunet.setInputSize((W, H))
    for j, (k, img) in enumerate(read_frames(clip)):
        f = models.yunet.detect(img)[1]
        faces[k] = f[:, [0, 1, 2, 3, 14]] if f is not None else np.zeros((0, 5))
        if j % step == 0:
            batch.append((k, img))
        if len(batch) == DET_BATCH:
            persons.update(find_persons(batch, models, W))
            batch = []
    if batch:
        persons.update(find_persons(batch, models, W))
    pose_frames = np.array(sorted(persons))
    vicon_hand = vicon_hands(arm, vic)
    hand = lambda k, ps: hand_points(ps, None if vicon_hand is None else vicon_hand[k])
    found = {"vicon": [[e] if e else [] for e in vic], "pose": [[] for _ in vic], "face": [[] for _ in vic]}
    for k, ps in persons.items():
        found["pose"][k] = pose_heads(ps, hand(k, ps))
    for k, f in faces.items():
        near = int(pose_frames[np.abs(pose_frames - k).argmin()])
        ps = persons[near]
        found["face"][k] = face_regions(f, ps, found["pose"][near] + found["vicon"][k], hand(k, ps), W)
    return found, max((len(p) for p in persons.values()), default=0), arm


def hold(per_frame, h):
    """Union of each frame's ellipses with those of the h frames on either side."""
    n = len(per_frame)
    return [[e for i in range(max(0, k - h), min(n, k + h + 1)) for e in per_frame[i]] for k in range(n)]


# ----------------------------------------------------------------------------- blur, encode, QA
STATIC_MASKS = Path(__file__).resolve().with_name("static_masks.tsv")


@lru_cache(None)
def _static_masks():
    return pd.read_csv(STATIC_MASKS, sep="\t", comment="#")


def static_rects(clip):
    """Pixel rectangles (x0, y0, x1, y1) of the static masks of the clip's session and view: static_masks.tsv lists,
    as fractions of the frame, the laptop screens of the side view that show the live top-view video or other content;
    they are blurred in every frame."""
    W, H = clip.size
    t = _static_masks()
    t = t[(t.participant_id == rc.sub_id(clip.row.pid)) & (t.session == rc.ses_id(clip.row.visit)) & (t.view == clip.view)]
    return [(int(r.x0 * W), int(r.y0 * H), int(np.ceil(r.x1 * W)), int(np.ceil(r.y1 * H))) for r in t.itertuples()]


def blur(img, ellipses, rects=()):
    """Pixelate (blocks of 1/BLOCKS_PER_WIDTH of the width) and Gaussian-blur (sigma half a block) the union of the
    ellipses, with a feathered edge, in place. Works on the bounding box of the regions; the Gaussian runs on the
    pixelated image at a quarter of the resolution and is upsampled bilinearly."""
    if not ellipses and not rects:
        return img
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for cx, cy, rx, ry in ellipses:
        centre = (int(np.clip(cx, -2 * W, 3 * W)), int(np.clip(cy, -2 * H, 3 * H)))
        cv2.ellipse(mask, centre, (int(rx), int(ry)), 0, 0, 360, 255, -1)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0:
        return img
    block = max(4, W // BLOCKS_PER_WIDTH)
    pad = 2 * block
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    w, h = x1 - x0, y1 - y0
    roi = img[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_NEAREST)
    pix = cv2.resize(cv2.GaussianBlur(pix, (0, 0), block / 8), (w, h), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(mask[y0:y1, x0:x1], (0, 0), FEATHER_PX).astype(np.float32) / 255.0
    roi[:] = cv2.blendLinear(pix, roi, m, 1.0 - m)
    return img


def ffmpeg_writer(clip, path):
    """ffmpeg reading raw BGR frames from stdin: H.264 (libx264, yuv420p), no audio, no metadata or encoder tags."""
    import imageio_ffmpeg
    W, H = clip.size
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(clip.fps), "-i", "-",
           "-an", "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
           "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def encode(clip, held, rects=()):
    """Pass 2: blurred frames (black where the clip has none) piped to ffmpeg; written atomically.
    Returns the mask of output frames that carry video."""
    tmp = clip.dst.with_name(clip.dst.stem + ".tmp.mp4")
    proc = ffmpeg_writer(clip, tmp)
    black = np.zeros((clip.size[1], clip.size[0], 3), np.uint8)
    content = np.zeros(clip.n_frames, bool)
    frames = read_frames(clip)
    nxt = next(frames, None)
    for k in range(clip.n_frames):
        img = black
        if nxt is not None and nxt[0] == k:
            img, content[k] = blur(nxt[1], held[k], rects), True
            nxt = next(frames, None)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed on {clip.name}: {proc.stderr.read().decode()}")
    tmp.replace(clip.dst)
    return content


def check(clip, held, models, preview_dir, rects=()):
    """Pass 3: YuNet at QA_THR on every QA_STEP-th frame of the written video: frames with a detection centred inside /
    outside a blur region and the highest score of each kind; the frame of each highest score is saved as a preview."""
    cap = cv2.VideoCapture(str(clip.dst))
    models.yunet_qa.setInputSize(clip.size)
    n, checked = 0, 0
    frames, best = {"in": 0, "out": 0}, {"in": (0.0, None), "out": (0.0, None)}
    while cap.grab():
        if n % QA_STEP == 0:
            img = cap.retrieve()[1]
            f = models.yunet_qa.detect(img)[1]
            checked += 1
            for kind, ds in split_faces(f, held[n] if n < len(held) else [], rects).items():
                if len(ds):
                    frames[kind] += 1
                    if ds[:, 14].max() > best[kind][0]:
                        best[kind] = (float(ds[:, 14].max()), (img, ds, n))
        n += 1
    cap.release()
    for kind, (_, shot) in best.items():
        if shot is not None:
            save_preview(shot[0], shot[1], preview_dir / f"{clip.dst.stem}_{kind}_frame{shot[2]:05d}.jpg")
    return dict(n_frames_written=n, qa_frames=checked, qa_frames_face_in_blur=frames["in"],
                qa_frames_face_outside_blur=frames["out"], qa_max_score_in_blur=round(best["in"][0], 3),
                qa_max_score_outside_blur=round(best["out"][0], 3))


def split_faces(faces, ellipses, rects=()):
    """YuNet detections split by whether their centre lies in one of the ellipses or rectangles."""
    if faces is None:
        return {"in": [], "out": []}
    in_rect = lambda x, y: any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in rects)
    hit = np.array([any(inside((d[0] + d[2] / 2, d[1] + d[3] / 2), e) for e in ellipses)
                    or in_rect(d[0] + d[2] / 2, d[1] + d[3] / 2) for d in faces], bool)
    return {"in": faces[hit], "out": faces[~hit]}


def save_preview(img, faces, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.copy()
    for d in faces:
        x, y, w, h = d[:4].astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img, f"{d[14]:.2f}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    s = 640 / img.shape[1]
    cv2.imwrite(str(path), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])


# ----------------------------------------------------------------------------- driver
def process(clip, models, preview_dir):
    """Blur, encode and check one clip; returns its QA row (regions per source, arm markers inside a region, YuNet
    detections on the written video)."""
    if clip.view not in BLUR_VIEWS:
        return process_unblurred(clip, models, preview_dir)
    t0 = time.time()
    found, n_persons, arm = scan(clip, models)
    h = max(HOLD_FRAMES, round(HOLD_S * float(clip.fps)))
    held_by = {s: hold(f, h) for s, f in found.items()}
    held = [sum((held_by[s][k] for s in held_by), []) for k in range(clip.n_frames)]
    rects = static_rects(clip)
    content = np.flatnonzero(encode(clip, held, rects))
    qa = check(clip, held, models, preview_dir, rects)
    mean = lambda x: round(float(np.mean(x)), 3) if len(x) else 0.0
    return dict(clip=clip.name, view=clip.view, n_frames=clip.n_frames, n_black=clip.n_frames - len(content),
                regions_per_frame=mean([sum(len(found[s][k]) for s in found) for k in content]),
                frames_zero_regions=int(sum(not held[k] for k in content)),
                frac_vicon=mean([bool(held_by["vicon"][k]) for k in content]),
                frac_pose=mean([bool(held_by["pose"][k]) for k in content]),
                frac_face=mean([bool(held_by["face"][k]) for k in content]),
                max_persons=n_persons, **markers_in_blur(clip, arm, found["vicon"], held, content), **qa,
                n_static_masks=len(rects),
                seconds=round(time.time() - t0, 1))


def process_unblurred(clip, models, preview_dir):
    """A view released without blur: the clip re-encoded (black frames where the source video has none, metadata
    stripped) and the QA face count of the written video."""
    t0 = time.time()
    held = [[] for _ in range(clip.n_frames)]
    content = np.flatnonzero(encode(clip, held))
    qa = check(clip, held, models, preview_dir)
    return dict(clip=clip.name, view=clip.view, n_frames=clip.n_frames, n_black=clip.n_frames - len(content),
                regions_per_frame=0.0, frames_zero_regions=len(content), frac_vicon=0.0, frac_pose=0.0, frac_face=0.0,
                max_persons=None, **qa, n_static_masks=0, blurred=False, seconds=round(time.time() - t0, 1))


def markers_in_blur(clip, arm, vicon, held, frames):
    """Shares of the projected arm markers inside the image that lie in a blur region (NaN without a VICON projection):
    shoulders (LSHO, RSHO), all 16 arm markers and the active fingertip, and the share of fingertip frames blurred
    outside OFF_HEAD x the VICON head ellipse (by a region that is not the child's head)."""
    if arm is None:
        return dict(shoulders_in_blur=np.nan, arm_markers_in_blur=np.nan, fingertip_in_blur=np.nan,
                    fingertip_off_head_in_blur=np.nan)
    W, H = clip.size
    hit = np.full(arm.shape[:2], np.nan)
    fin = ARM_MARKERS.index(str(clip.row.hand)[0].upper() + "FIN")
    off_head = np.full(len(arm), np.nan)
    for k in frames:
        for j, p in enumerate(arm[k]):
            if np.isfinite(p).all() and 0 <= p[0] < W and 0 <= p[1] < H:
                hit[k, j] = any(inside(p, e) for e in held[k])
        if np.isfinite(hit[k, fin]):
            near_head = any(inside(arm[k, fin], grown(e, OFF_HEAD)) for e in vicon[k])
            off_head[k] = bool(hit[k, fin]) and bool(vicon[k]) and not near_head
    sho = [ARM_MARKERS.index("LSHO"), ARM_MARKERS.index("RSHO")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frac = lambda x: round(float(np.nanmean(x)), 3)
        return dict(shoulders_in_blur=frac(hit[:, sho]), arm_markers_in_blur=frac(hit), fingertip_in_blur=frac(hit[:, fin]),
                    fingertip_off_head_in_blur=frac(off_head))


def qa_root(out):
    return rc.PRIVATE / "blur_qa" if Path(out).resolve() == rc.RELEASE.resolve() else Path(out) / "private" / "blur_qa"


def run_participant(pid, args, models):
    sub = rc.sub_id(pid)
    todo = clips(pid, args.out, args.views, args.runs)
    if args.skip_existing:
        todo = [c for c in todo if not c.dst.exists()]
    print(f"[{sub}] {len(todo)} clips", flush=True)
    rows = []
    for c in todo:
        q = process(c, models, qa_root(args.out) / sub)
        rows.append(q)
        print(f"[{sub}] {q['clip']}: {q['n_frames']} frames ({q['n_black']} black), {q['regions_per_frame']:.2f} regions/frame, "
              f"{q['frames_zero_regions']} unblurred, QA faces in/outside blur {q['qa_frames_face_in_blur']}/"
              f"{q['qa_frames_face_outside_blur']}, {q['n_frames'] / max(q['seconds'], 0.1):.1f} fps", flush=True)
    qa = pd.DataFrame(rows)
    f = qa_root(args.out) / f"{sub}.tsv"
    if f.exists() and len(qa):                  # keep the rows of the clips not processed in this run
        old = pd.read_csv(f, sep="\t", na_values="n/a")
        qa = pd.concat([old[~old["clip"].isin(qa["clip"])], qa], ignore_index=True)
    if len(qa):
        rc.write_tsv(qa, f)
    return qa


def summary(qa):
    if qa.empty:
        print("no clips")
        return
    g = qa.groupby("view").agg(clips=("clip", "size"), frames=("n_frames", "sum"), black=("n_black", "sum"),
                               seconds=("seconds", "sum"), unblurred=("frames_zero_regions", "sum"),
                               regions=("regions_per_frame", "mean"), qa_in=("qa_frames_face_in_blur", "sum"),
                               qa_out=("qa_frames_face_outside_blur", "sum"))
    g["fps"] = (g.frames / g.seconds).round(1)
    print(g.to_string())
    flagged = qa[(qa[["qa_max_score_in_blur", "qa_max_score_outside_blur"]].max(1) >= YUNET_SURE) | (qa.frames_zero_regions > 0)]
    print(f"{len(flagged)} of {len(qa)} clips to review (a face at >= {YUNET_SURE} after blurring, or frames without any region)")
    for c in flagged["clip"]:
        print("  review", c)


def main():
    ap = rc.cli(__doc__.splitlines()[0])
    ap.add_argument("--views", default=",".join(rc.VIEWS), help="comma-separated subset of top,side")
    ap.add_argument("--runs", type=int, nargs="*", help="testing: only these run numbers (1-12)")
    ap.add_argument("--skip-existing", action="store_true", help="only clips whose video is not written yet")
    args = ap.parse_args()
    args.views = args.views.split(",")
    models = Models(detect=any(v in BLUR_VIEWS for v in args.views))
    summary(pd.concat([run_participant(pid, args, models) for pid in rc.selected(args)], ignore_index=True))


if __name__ == "__main__":
    main()
