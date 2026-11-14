# Benchmark

Zero-shot pose estimation, ViconTraj-MS fine-tuning and the evaluation of the three benchmark tasks. The scripts read
the benchmark manifest `outputs/game_vicon_aligned/reaching/benchmark_manifest.csv`
(`alignment/build_benchmark_manifest.py`) and write to `outputs/benchmark/`. Scripts run from the repository root
(`python benchmark/evaluate.py`) or from this folder, as the Slurm wrappers in `slurm/` do.

The participant splits (8 test participants, five development folds) were drawn once by the rule of the paper (Sec. 5
and the supplement) and are released with public IDs as `derivatives/vikitraj/splits.json`. The first step,
`import_splits.py`, maps them back to study IDs with `release/private/id_map.tsv` and writes
`outputs/benchmark/splits.json`, which the benchmark, dataset and paper scripts read.

A trial is called a path in the code (`n_paths`, `*_paths.csv`); `row` is its `aligned_mat_row` in the manifest.
Experiment codes: E1 and E2 are Task 1 (2D pose per view, two-view 3D pose), E4 is Task 2 (kinematic features), E3 is
Task 3 (end-effector trajectory on the screen), E5 is ViconTraj-MS (fine-tuning on marker and trajectory labels,
temporal head, two-view lifting), E6 the analyses and E7 the qualitative figures.

| Step | Script | Slurm wrapper (`slurm/*.sbatch`) | Output in `outputs/benchmark/` |
|---|---|---|---|
| Participant splits | `import_splits.py` | | `splits.json` |
| VICON markers per trial | `alignment/vicon/export_markers.m` | MATLAB | `markers/<pid>_reaching_markers.mat` |
| Zero-shot pose, participant boxes, marker blobs | `zero_shot.py` | `bench_zero_shot`, `bench_gpu_model`, `bench_onnx_gpu`, `bench_blazepose` | `models/<model>/<pid>/<view>/<clip>.npz`, `boxes/`, `blobs/` |
| Camera of each video | `calibrate.py` | `bench_eval` (`CALIBRATE=1`) | `cameras/` |
| Marker labels and crops | `labels.py` | `bench_labels` | `labels/<pid>/<view>/<clip>.npz` |
| Trajectory labels | `labels_traj.py` | `bench_cpu` | `kp_traj` in the label files, `labels_traj_fit/` |
| E5 fine-tuning | `finetune.py` (ViTPose+), `finetune_sapiens.py` (Sapiens2 heads) | `bench_finetune`, `bench_gpu_step` | `finetune/<run>/` |
| E5 inference of the fine-tuned models | `zero_shot.py --participant <pid> --models <run>` | `bench_gpu_model` (`MODEL=<run>`) | `models/<run>/` |
| E5 temporal head, two-view lifting | `e5_seq.py --part seq2d`, `--part lift3d` (after E4) | `bench_gpu_step` | `models/seq_<base>/`, `results/E5_lift3d_<base>.csv` |
| E3 and the trajectory QA | `e3_screen.py` | `bench_e3`, `bench_final` | `results/E3_screen*.csv`, `qa_vicon_trajectory.csv` |
| E1, E2 | `evaluate.py` | `bench_eval`, `bench_final` | `results/E1_2d*.csv`, `results/E2_3d*.csv` |
| E4 | `e4_kinematics.py` | `bench_cpu`, `bench_final` | `results/E4_kin_*.csv`, `tracks3d/` |
| E6 | `e6_analyses.py` | `bench_final` | `results/E6*.csv` |
| E7 | `e7_qualitative.py`, `e7_path_frames.py` | `bench_final` (`e7_qualitative.py`) | `reports/E7/` |

`e3_screen.py` writes the trajectory QA that `benchlib.manifest()` applies to every other experiment, so on a new
output folder it runs before `evaluate.py`. E6 also reads `evaluate.py --shift-ms <ms> --out-suffix _shift<ms>ms`
(33, 67, 100 and 200 ms), `evaluate.py --no-filter --out-suffix _nofilter`, and the fine-tuned runs
`ft_b_full_pooled_n4`, `_n8`, `_n12` (`finetune.py --n-train`) and `ft_b_full_pooled_shift67`, `_shift200`
(`labels_shift.py`, then `finetune.py --label-shift-ms`).

`benchlib.py` holds the shared code: paths, the manifest, the clocks that place VICON samples and video frames on the
game clock, the joint proxies, the keypoint maps, the per-video left/right decision for BlazePose and the
registration of the fine-tuned (`ft_*`) and temporal-head (`seq_*`) models. `paper/make_results.py` makes the tables
and figures of the paper from `outputs/benchmark/results/` and `reports/E7/`.

## Protocol

- Scoring: per participant first, as the median over the frames of all its trials (E1, E2) or over its trials (E3,
  E4 angles), then the mean over participants with a 1000-sample bootstrap 95% CI over participants. Results are
  given for the test split (8 participants) and for all participants, for TD, UCP and both groups, and in E1 and E2
  for the active and the resting arm.
- Filtering: keypoints get a Hampel filter (7-frame window, 3 scaled MAD) and a zero-lag 4th-order Butterworth
  low-pass at 6 Hz; the VICON reference gets the same low-pass. `E6e_filtering.csv` compares raw keypoints with raw
  markers.
- Labels: `visible`, the primary set of the 2D scores, keeps the blob frames (10 Hz) where every marker of the joint
  has a white image blob within 8 px of its projection; `all` keeps every frame of the movement window with a VICON
  reference. 2D errors are converted to mm with the local scale of the camera; 3D scores use all labels.
- Keypoint decoding of the fine-tuned models: the ViTPose+ runs take the DARK sub-pixel heatmap peak, averaged with
  the horizontally flipped crop (`finetune.decode`, `zero_shot.FineTuned`); the Sapiens2 heads use the model's UDP
  decoding without flip (`zero_shot.FineTunedSapiens`); the temporal heads regress coordinates.
