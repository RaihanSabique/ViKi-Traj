# Sample data

One trial of the ViKi-Traj release in the release's layout, copied by `scripts/make_sample.py`: sub-03 ses-1 run-06,
a typically developing test participant playing with the left arm (21.7 s; VICON, top and side views valid). It is the
trial of the demo videos in `docs/media/`, the test trial with the lowest median 3D error of Sapiens2-1B +
ViconTraj-MS.

| Path in `ViKi-Traj/` | Content |
|---|---|
| `participants.tsv`, `phenotype/` | the participant's rows |
| `sub-03/ses-1/beh/` | one row per run of the visit |
| `sub-03/ses-1/motion/` | run-06: VICON markers, end-effector trajectory on the screen (iPad), events |
| `sourcedata/trajectory/` | end-effector trajectory file of the visit (all 12 runs) |
| `sourcedata/vicon/` | the C3D recording that contains run-06 (runs 5-8), with its row of `recordings.tsv` |
| `sourcedata/videos/` | run-06 clips: top view (unblurred, filmed from behind) and side view (heads blurred) |
| `derivatives/vikitraj/` | manifest row, cameras; run-06 labels2d, keypoints2d (16 models), boxes, tracks3d |

Every table has a JSON sidecar describing its columns. The ViconTraj-MS checkpoints, the result tables and the dataset
card, license, changelog and citation files are only in the full dataset. The beh file is that of the whole visit,
because session time starts at the visit's first trial, so `dataset/validate_release.py --partial` lists its other 11
runs as not in the manifest. The C3D file reads with ezc3d or py-c3d 0.6 or later. The trial is covered by the terms of
the dataset (its LICENSE file): use for the review of the submission only, and no attempt to identify the participant.

```python
import pandas as pd

root = "sample_data/ViKi-Traj"
t = pd.read_csv(f"{root}/derivatives/vikitraj/manifest.tsv", sep="\t", na_values="n/a").iloc[0]

def motion(path):  # motion.tsv has no header; the column names are in its channels.tsv
    names = pd.read_csv(f"{root}/{path.replace('_motion.tsv', '_channels.tsv')}", sep="\t").name
    return pd.read_csv(f"{root}/{path}", sep="\t", header=None, names=names, na_values="n/a")

vicon = motion(t.vicon_motion)  # 24 markers, mm, 100 Hz
ipad = motion(t.ipad_motion)    # latency (s) and finger position on the screen (mm)
labels = pd.read_csv(f"{root}/{t.top_labels}", sep="\t", na_values="n/a")  # VICON joints projected, px
video = f"{root}/{t.top_video}"  # 652 frames at 30 Hz; frame k is row k of labels and keypoints

model = "ft_sapiens2_1b_head"   # Sapiens2-1B + ViconTraj-MS
kp_file = t.top_labels.replace("labels2d/", f"keypoints2d/{model}/").replace("_labels", "_keypoints")
kp = pd.read_csv(f"{root}/{kp_file}", sep="\t", na_values="n/a")  # same frames as labels

tracks_file = "derivatives/vikitraj/tracks3d/sub-03/ses-1/sub-03_ses-1_task-reaching_run-06_tracks3d.tsv"
tracks = pd.read_csv(f"{root}/{tracks_file}", sep="\t", na_values="n/a").set_index(["source", "time", "joint"])
err = ((tracks.loc[model] - tracks.loc["vicon"]) ** 2).sum(axis=1, min_count=3) ** 0.5
print(err.groupby(level="joint").median())  # mm: elbow 2.5, fingertip 4.3, shoulder 4.3, wrist 3.8
```

The demo videos can be drawn from the sample alone: `python scripts/render_demo.py --root sample_data/ViKi-Traj
--out <dir>` (about 1.5 min on a CPU) gives the files in `docs/media/`.

Another trial: `python scripts/make_sample.py --root path/to/ViKi-Traj --trial sub-XX ses-N run-NN --out <dir>`.
