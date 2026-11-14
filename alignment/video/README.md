# Video alignment

`align_videos.py` places the top and side videos of each visit on the game clock and cuts one clip per trial and view.
A clip covers the movement window of its trial: frame 0 is at the first `RMov_time` sample of the trial
(`move_start_time`) and the last frame at its last sample (`move_end_time`), as for the VICON crop.

## Run

Inputs: `outputs/game_vicon_aligned/reaching/<pid>_reaching_paths_plain.mat` (written by
`alignment/vicon/export_paths_plain.m`) and the visit videos in `data/raw_video_data/<pid>/`,
`*Logitech*/V<visit>_*.mp4` (top) and `*Tripod*/V<visit>_*.mp4` (side), one per visit and sometimes split into several
files. Python packages: numpy, pandas, scipy, opencv-python, imageio-ffmpeg (its ffmpeg binary cuts the clips) and
matplotlib.

```bash
# one participant; --no-cut aligns without writing clips
python alignment/video/align_videos.py --participant XXXXXXX115 [--no-cut]

# all participants, one per array task (submit from the repository root)
python alignment/video/collect_outputs.py participants   # video_participants.txt, one study ID per line
sbatch --array=1-29 slurm/video_align.sbatch

# combine the per-participant tables into video/*_all.csv, then measure the clips
python alignment/video/collect_outputs.py tables
python alignment/video/clip_durations.py
```

Outputs in `outputs/game_vicon_aligned/reaching/video/<pid>/`:

| File | Content |
|---|---|
| `clips/<view>/<pid>_V<visit>_T<k>_P<p>_<hand>.mp4` | clip of trial `P<p>` (1-4) of recording `T<k>` (1-3) |
| `video_segments.csv` | per trial and view: source video, clip start and length, offset, drift, scores, status |
| `video_alignment_summary.csv` | per video: offset, drift, score, margin, residual SD of the refined starts |
| `alignment_V<visit>_<view>_<file>.png` | QA plot: best-cell flow against the game-object velocity |
| `flow_<view>_<file>.npz` | cached cell flow |

`collect_outputs.py tables` concatenates the two tables of all participants into `video/video_segments_all.csv` and
`video/video_alignment_summary_all.csv`. `clip_durations.py` writes `video/clip_durations.csv`: the measured duration
of each clip (frames / fps) next to its movement window (`game_duration_s`) and VICON crop (`vicon_duration_s`), and
their differences in seconds.

## Method

The finger drags the game object, so the image motion around the finger follows the velocity of the game object. Times
are on the visit clock (seconds since the first trial start of the visit), sampled at 30 Hz.

1. Farneback optical flow is computed on frames resized to 320 px width and averaged over a 16 x 9 grid of cells.
2. The flow (u, v) of each cell is correlated with the game-object velocity (vx, vy) at every lag over the whole visit
   (Pearson's r over the samples inside trials, by FFT).
3. The score of a cell, `sqrt((r(u,vx)^2 + r(u,vy)^2 + r(v,vx)^2 + r(v,vy)^2) / 2)`, does not depend on the rotation of
   the camera relative to the screen, so no screen region has to be marked.
4. The lag at which the 6 best cells score highest gives the offset of the video. Its margin over the best lag more
   than 1 s away measures ambiguity.
5. The movement window of each trial, padded by 0.5 s, is refined within +/-1 s on those cells.
6. A line through the refined starts, refitted without the starts more than 0.2 s off it, gives the offset and the clock
   drift: video time = `drift_slope` x visit clock + `offset_s`. The clip times are read off this line; `residual_s` is
   the refined start minus the line.

A visit split over several files is matched file by file, and each trial takes the file that covers it best.

## Validity

`video_status` in `video_segments.csv` is `Need Review` when the match is ambiguous (margin < 0.1), the refined start
is more than 0.2 s off the line, or the video covers less than 90 % of the movement window. The benchmark uses the
stricter flags of the manifest.

## Benchmark manifest

`python alignment/build_benchmark_manifest.py` writes `outputs/game_vicon_aligned/reaching/benchmark_manifest.csv`, one
row per trial (696), after the VICON alignment, `trajectory_check.py`, the video alignment and `clip_durations.py`.
It locates the data of each stream (`aligned_mat_path` and `aligned_mat_row`, the 1-based row of `outputTable`;
`<view>_clip_path`, `<view>_source_video`, `<view>_start_frame`, `<view>_n_frames`), gives the alignment residuals
(`vicon_align_err_s`, `<view>_align_err_s`, in s) and flags each stream, with the reasons in `<stream>_invalid_reason`:

- `vicon_valid`: `review_status` is OK (see `alignment/vicon`), VICON covers the whole movement window, at least two
  estimates support the offset and it is within 0.1 s of the consensus of its recording, and the trial passes the
  trajectory test.
- `top_valid`, `side_valid`: the clip exists, the match is unambiguous (margin >= 0.1), the video covers the whole
  movement window, the refined start is within 0.1 s of the line, the clock drift is at most 0.2 % and the clip
  duration is within 2 frames of the movement window.
- `benchmark_valid`: `vicon_valid` and at least one valid view (`valid_views`, `n_valid_views`).

Video to VICON (pose, kinematics) needs `vicon_valid & <view>_valid`. The videos are aligned to the game directly, so
video to game trajectory needs only `<view>_valid`.

## Trajectory test

`python alignment/trajectory_check.py` writes `trajectory_check.csv`; it needs the marker export
(`alignment/vicon/export_markers.m`). The alignment uses only the speed of the game object, so the game trajectory
itself is an independent test. For every trial with VICON data, the fingertip marker of the active hand (6 Hz
low-pass) is mapped to the screen by an affine transform (mm -> game px) and compared with the game trajectory. The
trial passes (`trajectory_align_pass`) when the affine fitted on the trial reproduces the trajectory within 10 mm RMSE
(`rmse_self_mm`), the affine fitted on the other trials of its recording (else of its visit and hand) within 20 mm
(`rmse_lopo_mm`, not required without other trials), and, with that affine fixed, the best lag within +/-1 s is at
most 0.1 s (`lag_s`).
