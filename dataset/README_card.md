---
pretty_name: ViKi-Traj
license: other
license_name: vikitraj-review-terms
license_link: LICENSE
language:
  - en
size_categories:
  - n<1K
task_categories:
  - keypoint-detection
  - other
tags:
  - pose-estimation
  - human-pose
  - 3d-pose-estimation
  - motion-capture
  - vicon
  - upper-limb
  - kinematics
  - children
  - cerebral-palsy
  - rehabilitation
  - bids
  - motion-bids
  - video
gated: true
extra_gated_heading: "Access to ViKi-Traj (review copy)"
extra_gated_prompt: >-
  ViKi-Traj contains movement data and video of children, some with cerebral palsy.
  By requesting access you agree to use the data only for research, not to attempt to identify
  any participant, and to delete it on request.
extra_gated_fields:
  I agree to use the data only for research and not to attempt to re-identify participants: checkbox
configs:
  - config_name: trials
    data_files: derivatives/vikitraj/manifest.tsv
    sep: "\t"
    default: true
  - config_name: participants
    data_files: participants.tsv
    sep: "\t"
---

# ViKi-Traj: markerless video to upper-limb kinematics

ViKi-Traj pairs video of participant's reaching movements with two independent references: 24-marker VICON motion
capture and the end-effector trajectory logged by a touchscreen game. Children aged 5-12 years, typically developing
(TD) or with unilateral cerebral palsy (UCP), drag a game object with one finger along a curved path on an upright
iPad (the reaching task) while VICON, a top camera (behind and above the participant) and a side camera record the
arm.

This is the anonymous review copy (v1.0) of the dataset of a CVPR 2027 submission. Author names, institution,
license and DOI are withheld for double-blind review. The datasheet is Section A of the paper's supplementary material.

## Summary

| | |
|---|---|
| Participants | 29 (18 TD, 11 UCP), aged 5-12 years, two visits each |
| Trials | 696 (12 per visit; in most visits the hand changes every two trials), 128 min of movement |
| VICON | 24 markers, 100 Hz, mm; per trial: gap-filled and smoothed by the VICON software; full recordings: raw positions (C3D, gaps not filled) |
| End-effector trajectory | finger position while it drags the game object (which follows it), on the screen in mm, about 120 Hz while dragging, with touch events |
| Video clips | 617 top and 650 side clips, 1920x1080 at 30 Hz; side clips head-blurred, top clips unblurred (filmed from behind) |
| Labels | per frame of the 539 top and 535 side clips with VICON data and a usable camera: 10 joints and 24 markers projected into the image; visibility on every third frame of clips valid in that view |
| Model outputs | 2D keypoints of 8 zero-shot models and 8 ViconTraj-MS variants, person boxes, two-view 3D tracks of the active arm |
| Valid trials | VICON 556, top video 522, side video 488, all three streams 416 (72 min, 24 participants) |
| Splits | 8 test participants (5 TD, 3 UCP), 21 development participants in 5 folds |

## Layout

The dataset follows [BIDS](https://bids.neuroimaging.io) with the
[Motion-BIDS](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html) extension.
One trial is one run; `ses-1` and `ses-2` are the two visits. `<p>` below stands for `sub-01_ses-1_task-reaching`.

```
ViKi-Traj/
  README.md  LICENSE  CHANGES  CITATION.cff  dataset_description.json
  participants.tsv/.json   phenotype/anthropometrics.tsv/.json
  task-reaching_beh.json  task-reaching_events.json            column descriptions
  sub-01/ses-1/                                   (sub-01 ... sub-29, ses-1 and ses-2)
    beh/     <p>_beh.tsv                                       one row per run
    motion/  <p>_tracksys-vicon_run-01_motion.tsv              (+ _channels.tsv, _motion.json)
             <p>_tracksys-ipad_run-01_motion.tsv               (+ _channels.tsv, _motion.json)
             <p>_run-01_events.tsv                             (run-01 ... run-12)
  sourcedata/
    trajectory/sub-01/ses-1/<p>_trajectory.json                end-effector trajectory file of the 12 runs
    vicon/sub-01/ses-1/<p>_acq-rec1_vicon.c3d                  full recordings (+ recordings.tsv/.json)
    videos/sub-01/ses-1/<p>_acq-top_run-01_video.mp4           (acq-top, acq-side)
  derivatives/vikitraj/                           dataset_description.json, .bidsignore, README.md
    manifest.tsv/.json  splits.json  cameras/  results/  checkpoints/
    labels2d/sub-01/ses-1/<p>_acq-top_run-01_labels.tsv
    keypoints2d/<model>/sub-01/ses-1/<p>_acq-top_run-01_keypoints.tsv
    boxes/sub-01/ses-1/<p>_acq-top_run-01_boxes.tsv
    tracks3d/sub-01/ses-1/<p>_run-01_tracks3d.tsv
```


## Quick start

```python
import pandas as pd

root = "ViKi-Traj"
trials = pd.read_csv(f"{root}/derivatives/vikitraj/manifest.tsv", sep="\t", na_values="n/a")
t = trials[(trials.vicon_valid == True) & (trials.top_valid == True)].iloc[0]

# VICON markers of the trial (no header; column names in the channels file)
names = pd.read_csv(f"{root}/{t.vicon_motion.replace('_motion.tsv', '_channels.tsv')}", sep="\t").name
vicon = pd.read_csv(f"{root}/{t.vicon_motion}", sep="\t", header=None, names=names, na_values="n/a")

# end-effector trajectory on the screen and the projected joints of the top clip
ipad = pd.read_csv(f"{root}/{t.ipad_motion}", sep="\t", header=None, names=["latency", "x", "y"], na_values="n/a")
labels = pd.read_csv(f"{root}/{t.top_labels}", sep="\t", na_values="n/a")
video = f"{root}/{t.top_video}"
```

## De-identification

Study codes are replaced by `sub-01` to `sub-29`; names, dates of birth, calendar dates and clock times are removed;
ages converted to years; sex and handedness are not released
(handedness follows from `affected_arm`). In the side clips the heads of all persons are blurred. The top camera films the participant from behind and above, and its clips are not
blurred. The continuous visit videos are not public; they are available only under a data-use agreement. Do not
attempt to identify any participant.

## Versions

- 1.0.0 (review copy): initial release. Errata and later versions are listed in `CHANGES`.

## Citation

Withheld for double-blind review.
