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

# ViKi-Traj: markerless video to upper-limb kinematics in children

ViKi-Traj pairs video of children's reaching movements with two independent references: 24-marker VICON motion
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
| Video clips | 617 top and 650 side clips, 1920x1080 at 30 Hz (three source videos 1280x720); side clips head-blurred, top clips unblurred (filmed from behind) |
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

`derivatives/vikitraj` is a BIDS derivative by its `dataset_description.json` only; its custom layout is listed in its
`.bidsignore`. Every table has a JSON sidecar describing its columns: the `*_channels.tsv` and `*_motion.json` of each
motion file, the task-level `task-reaching_*.json` at the root and at the top of `labels2d/`, `keypoints2d/<model>/`,
`boxes/` and `tracks3d/`, and `manifest.json`, `recordings.json`, `results/README.md` and `checkpoints/README.md`.

## Conventions

- Time: every time is relative. Run time is seconds since the movement start of the trial: sample 0 of the
  motion files and frame 0 of the clips. Session time (`beh.tsv`, trajectory files) is seconds since the start of
  the first trial of the visit. No calendar date or clock time is included.
- VICON: positions in mm in the laboratory frame, z up. Sample k is the recorded frame nearest run time k/100 s
  (within 5 ms); a run's file has `floor(movement_time * 100) + 1` samples, `n/a` where a marker is missing. The
  files hold the VICON software's gap-filled, smoothed trajectories, not slices of the raw C3D: measured frames differ
  by a median of 0.24 mm, more at raw spikes. Sample k is C3D frame `StartFrameInRecording` + k (0-based); in 19 runs
  the recording started after the movement start, so `StartFrameInRecording` is negative and the file begins with
  n/a rows. `SpatialAxes` in each `motion.json` gives the lab axes relative to the participant: `ALS` in most
  sessions, `PRS` in sub-02 ses-1, sub-06 ses-2, sub-10 ses-2, sub-13 ses-2, sub-14 ses-2 and sub-15 ses-2.
- Screen: finger positions in mm on the iPad screen, origin at the screen centre, x to the participant's right,
  y up; 1 game point = 25.4/132 mm. Samples are logged only while the finger drags the object: they are 8-9 ms apart,
  but 197 runs contain pauses over 0.1 s (up to 24 s, 11 % of the movement time). Use `latency`, never the sample
  index, as time. `SamplingFrequencyEffective` is the rate while dragging. No screen-to-VICON calibration is
  released: the screen stood upright facing the participant, and the paper fits the map per session.
- Image: pixel coordinates of the full video frame, x right, y down, origin at the top-left corner.
- Frames: frame k of a clip is at nominal run time k / `<view>_fps` (manifest; the MP4 header may round 29.9968
  to 30); the `time` column of the frame tables gives the exact capture time. Where the source video began after the
  movement start, the clip starts with black frames (`<view>_lead_black_frames`), and the frame tables have no rows
  for them.
- Validity: no trial is removed. `derivatives/vikitraj/manifest.tsv` flags every stream of every trial as valid
  or not, with the reason. The paper's benchmark uses `vicon_valid & <view>_valid` (170 top and 160 side clips of
  the test participants). Six of these side clips (sub-01 ses-1) have no released side labels or 3D tracks (see
  Known limitations), so with the released labels the set is `vicon_valid & <view>_valid` with `<view>_labels`
  present.

## Model outputs

- `keypoints2d/<model>/`: zero-shot models (`vitpose_b`, `rtmpose_wb`, `rtmpose_hand`, `dwpose_l`, `rtmw_x`,
  `sapiens2_0.4b`, `sapiens2_1b`, `blazepose`) for 981 of the 1010 clips valid in their view (29 view-valid clips of
  VICON-invalid trials have none); ViconTraj-MS fine-tuned models (`ft_*`) for the clips with `vicon_valid` and
  `<view>_valid`; temporal heads (`seq_*`) for the test participants only. The `ft_*` outputs come from one
  checkpoint trained on folds 2-5 with early stopping on fold 1, so they are training data for the 17 participants
  of folds 2-5 and the early-stopping set for the 4 of fold 1; only the 8 test participants are held out. The paper
  uses a position only where its score is at least the threshold given in each sidecar.
- `boxes/`: the person box the benchmark used per frame (the crop of every model); rows align with keypoints2d.
- A clip's keypoints and boxes follow from `<view>_labels` by replacing `labels2d/` with `keypoints2d/<model>/` or
  `boxes/` and the suffix `_labels` with `_keypoints` or `_boxes`.
- `tracks3d/`: long format (`time`, `source` = vicon or a model, `joint`, `x`, `y`, `z` in mm, lab frame) at 50 Hz,
  for the shoulder, elbow, wrist and fingertip of the active arm.
- `checkpoints/`: ViconTraj-MS weights with their configs; `results/`: the paper's result tables.

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

## Known limitations

- 7 runs have no VICON recording (sub-28 ses-2 run-04 and run-08; sub-29 ses-2 run-04 and run-09 to run-12): they
  have iPad motion, events and beh rows but no `tracksys-vicon` files.
- One side camera (sub-01, ses-1) is excluded from the labels and 3D tracks (`CalibrationOK` false): 8 % of the
  marker samples project behind it. The paper's results include its 6 test trials.
- Sides of BlazePose keypoints are assigned per video (as in the paper), so a few clips have left and right crossed.
- In about half of the clips the first image repeats the second (a seek artefact of the clip cut); the two rows of the
  frame tables are then equal.
- iPad latency: in sub-14 ses-1 run-07 the logged clock steps back by 64 ms, and in 7 files two consecutive samples
  share a latency (sub-19 ses-2 run-10, sub-28 ses-2 run-05, sub-29 ses-1 run-01, run-09, run-10, sub-29 ses-2
  run-11, run-12). The samples are kept in logged order, and the `motion.json` has a `LatencyNote`.
- C3D readers: py-c3d 0.5 cannot iterate the frames of these files under numpy 2; py-c3d 0.6 or later and ezc3d
  read them. Gaps are invalid points (residual -1, coordinates 0); measured points have residual 1.0 in py-c3d and 0
  in ezc3d, so detect gaps with a negative residual.

## De-identification

Study codes are replaced by `sub-01` to `sub-29`; names, dates of birth, calendar dates and clock times are removed;
ages are whole years, heights are rounded to 10 mm and weights to 1 kg; sex and handedness are not released
(handedness follows from `affected_arm`). In the side clips the heads of all persons are blurred (projected VICON head
markers, a whole-body pose model on every person and a face detector, dilated in space and time), as are laptop
screens that show the live video. The top camera films the participant from behind and above, and its clips are not
blurred. The continuous visit videos are not public; they are available only under a data-use agreement. Do not
attempt to identify any participant.

## Versions

- 1.0.0 (review copy): initial release. Errata and later versions are listed in `CHANGES`.

## Citation

Withheld for double-blind review.
