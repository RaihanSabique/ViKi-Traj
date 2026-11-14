"""Duration of every clip next to the duration of its trial's movement window and VICON crop.

Reads the clips listed in video/video_segments_all.csv and writes video/clip_durations.csv (both in
outputs/game_vicon_aligned/reaching/). Durations are in s; the diff_* columns are clip - game, clip - VICON and
VICON - game (NaN without VICON data).

    python alignment/video/clip_durations.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

ALIGNED = Path(__file__).resolve().parents[2] / "outputs" / "game_vicon_aligned" / "reaching"
VIDEO = ALIGNED / "video"


def vicon_frames(pid):
    """Frames and frame rate of the VICON crop of every trial (row = outputTable row)."""
    m = sio.loadmat(ALIGNED / f"{pid}_reaching_paths_plain.mat", squeeze_me=True, struct_as_record=False)
    return pd.DataFrame([dict(pid=pid, row=int(r.row), vicon_frames=int(r.total_frames),
                              vicon_rate=float(r.framerate))
                         for r in np.atleast_1d(m["paths"])])


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    seg = pd.read_csv(VIDEO / "video_segments_all.csv", parse_dates=["start", "end", "move_start", "move_end"])
    vic = pd.concat([vicon_frames(p) for p in seg.pid.unique()])
    seg = seg.merge(vic, on=["pid", "row"], how="left")

    frames, fps = [], []
    for pid, clip in zip(seg["pid"], seg["clip"]):
        cap = cv2.VideoCapture(str(VIDEO / pid / clip))
        frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps.append(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    seg["clip_frames"] = frames
    seg["clip_fps"] = fps

    out = pd.DataFrame({
        "pid": seg.pid, "visit": seg.visit, "trial_in_visit": seg.trial_in_visit, "path": seg.path,
        "hand": seg.hand, "view": seg.view, "clip": seg["clip"],
        "clip_frames": seg.clip_frames, "clip_fps": seg.clip_fps,
        "clip_duration_s": seg.clip_frames / seg.clip_fps,
        "game_duration_s": (seg.move_end - seg.move_start).dt.total_seconds(),
        "vicon_frames": seg.vicon_frames,
        "vicon_duration_s": np.where(seg.vicon_frames > 0, seg.vicon_frames / seg.vicon_rate, np.nan),
        "video_status": seg.video_status, "review_status": seg.review_status})
    out["diff_with_game_duration"] = out.clip_duration_s - out.game_duration_s
    out["diff_with_vicon_duration"] = out.clip_duration_s - out.vicon_duration_s
    out["diff_vicon_with_game_duration"] = out.vicon_duration_s - out.game_duration_s
    out = out.sort_values(["pid", "visit", "trial_in_visit", "path", "view"])
    out.to_csv(VIDEO / "clip_durations.csv", index=False, float_format="%.4f")
    print(f"wrote {VIDEO / 'clip_durations.csv'} ({len(out)} clips)")
    print(out.groupby("view")[["diff_with_game_duration", "diff_with_vicon_duration",
                               "diff_vicon_with_game_duration"]]
          .describe(percentiles=[0.5]).round(3).T)


if __name__ == "__main__":
    main()
