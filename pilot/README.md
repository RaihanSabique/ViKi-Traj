# Pilot participant

The supplementary material reports a pilot participant whose marker data of the second visit were 0.8-1.3 s out of
step with the game trajectory in each of its three VICON recordings. An estimate from the game trajectory and one
from the video alone (VICON wrist speed against the wrist speed of ViTPose+ in the side view) agree within 50 ms, and
the correction brings the median per-trial RMSE of the VICON fingertip against the game trajectory from 15.5 to
3.2 mm (16 and 3 mm in the supplement). These scripts reproduce that analysis for one participant. They use the
internal layout (inputs in `data/<pid>/`, outputs in `outputs/<pid>/`) and the shared modules of `src/`.

The analysis tests the aligned trial table that came with the data, `<pid>_reaching_aligned_provided.mat`, which was
made before and outside `alignment/vicon`. Its VICON crops of the second visit are the ones that are out of step. The
pilot analysis exposed this offset, and `alignment/vicon` corrects it: in the table that `run_alignment.m` writes for
this participant (`outputs/game_vicon_aligned/reaching/<pid>_reaching_aligned.mat`), the VICON clock differs from the
provided one by 0.8-1.3 s in the second visit, the offsets estimated here, and by at most 0.05 s in the first.

Inputs in `data/<pid>/`:

- `<pid>_reaching_aligned_provided.mat`: the provided aligned trial table (a MATLAB table, one row per trial).
- `<pid>_reaching_plain.mat`: the same table as a plain MATLAB struct array `trials` (-v7), written by
  `export_trials.m`: game trajectory (`RMov_path`, `RMov_time_step_ms`), `trial_start_time`, `trial_end_time`,
  `study_day`, `group_no` (VICON recording), `trial_no`, `trial_hand`, and the VICON markers of the trial (`labels`,
  `position_data_interpolated`, `time`, `relative_start_sec`).
- One video per visit and view: `V1_*_Top.mp4`, `V1_*_Side.mp4`, `V2_*_Top.mp4`, `V2_*_Side.mp4`.
- `rois.json`: per video, the polygon (px) of the screen in the top view and of the strip where the finger meets the
  screen in the side view.

```bash
matlab -batch "cd pilot; export_trials('PID')"      # provided trial table -> plain struct for Python
python pilot/align_trials.py --participant PID      # --cut also writes one clip per trial and view
python pilot/run_models.py --participant PID        # pose models; uses the GPU when there is one
python pilot/evaluate_pilot.py --participant PID    # [--reuse_cameras] [--resync]
```

| Script | Writes to `outputs/<pid>/` |
|---|---|
| `align_trials.py` | `alignment/`: `trial_segments.csv`, `alignment_summary.csv`, `alignment_<visit>.png`, flow cache |
| `run_models.py` | `models/<visit>_<view>_T<row>.npz`: ViTPose+-B, RTMPose-WB and MediaPipe keypoints, boxes |
| `evaluate_pilot.py` | `eval/`: `vicon_sync.csv`, `calibration.csv`, `cameras.npz`, joint and trajectory errors |

`evaluate_pilot.py` estimates the offsets when `eval/vicon_sync.csv` is missing or with `--resync`, and fits the cameras
unless `--reuse_cameras` finds `eval/cameras.npz`. `vicon_sync.csv` holds, per trial, the lag from the game trajectory
(`game_lag_s`) and from the video (`video_lag_s`), and the correction of each VICON recording (`group_lag_s`, the
median game lag of its trials). The fingertip RMSE is the median `rmse_mm` of the source `Vicon (3-D)` in
`trajectory_errors.csv`; the uncorrected value comes from `trajectory_errors` with `Context.sync = None`, which the
command line does not offer.

Python packages: numpy, pandas, scipy, opencv-python, matplotlib, tqdm, pyarrow, imageio-ffmpeg (`--cut`), and for
`run_models.py` torch, transformers, onnxruntime and mediapipe with the weights that `src/trial_models.py` loads.
