"""Participant list of the video alignment, and its per-participant tables combined.

In outputs/game_vicon_aligned/reaching/: 'participants' writes video_participants.txt, the study ID of every
<pid>_reaching_paths_plain.mat, one per line (line k is the participant of array task k of the slurm jobs).
'tables' concatenates video/<pid>/video_segments.csv and video/<pid>/video_alignment_summary.csv of all participants
into video/video_segments_all.csv and video/video_alignment_summary_all.csv.

    python alignment/video/collect_outputs.py participants     # before slurm/video_align.sbatch
    python alignment/video/collect_outputs.py tables           # after it, before clip_durations.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ALIGNED = Path(__file__).resolve().parents[2] / "outputs" / "game_vicon_aligned" / "reaching"
VIDEO = ALIGNED / "video"
PLAIN = "_reaching_paths_plain.mat"


def participants():
    pids = sorted(f.name[:-len(PLAIN)] for f in ALIGNED.glob(f"*{PLAIN}"))
    (ALIGNED / "video_participants.txt").write_text("".join(f"{p}\n" for p in pids))
    print(f"video_participants.txt: {len(pids)} participants")


def tables():
    for name in ("video_segments", "video_alignment_summary"):
        files = sorted(VIDEO.glob(f"*/{name}.csv"))
        if not files:
            sys.exit(f"no {name}.csv in {VIDEO}/<pid>/")
        t = pd.concat([pd.read_csv(f) for f in files])
        t.to_csv(VIDEO / f"{name}_all.csv", index=False, float_format="%.4f")     # the format of align_videos.py
        print(f"video/{name}_all.csv: {len(t)} rows from {len(files)} participants")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("participants", "tables"),
                    help="participants: video_participants.txt; tables: video/*_all.csv")
    args = ap.parse_args()
    if args.step == "participants":
        participants()
    else:
        tables()


if __name__ == "__main__":
    main()
