# Paper tables and figures

These scripts write the generated tables and figures of the paper to `outputs/paper/tables/` and `outputs/paper/fig/`.
`dataset_stats.py` reads only the released dataset and reproduces Table 2. The other scripts read the internal outputs
of the pipeline (`outputs/game_vicon_aligned/reaching/`, `outputs/benchmark/`) and `release/private/id_map.tsv`, which
maps the study IDs to the released participant IDs (sub-NN) used in the tables and figures; `make_fig_dataset.py` also
reads frames of the visit videos in `data/raw_video_data/` and blurs faces with the YuNet model in `models/face/`.
These inputs, apart from the YuNet weights, are not distributed.

Run from the repository root, after the release build and the benchmark (`slurm/bench_final.sbatch`). The scripts do
not depend on each other.

```bash
python paper/dataset_stats.py        # --release <root>, default release/ViKi-Traj; reads every clip header (~1 min)
python paper/make_tables.py
python paper/make_results.py
HF_HUB_OFFLINE=1 python paper/make_fig_dataset.py --pid XXXXXXX110 --row 14  # study ID and aligned_mat_row of the trial
```

| Script | Writes to `outputs/paper/` | Paper |
|---|---|---|
| `dataset_stats.py` | `tables/dataset_stats.tex` | Table 2 |
| `make_fig_dataset.py` | `fig/streams_moment.pdf`, `fig/alignment.pdf` | Figure 1 (bottom row), Figure 2 |
| `make_results.py` | `fig/results_zs_ft.pdf`, `tables/res_kin.tex` | Figure 3, Table 3 |
| `make_results.py` | `tables/splits.tex`, `res_zeroshot.tex`, `res_traj.tex`, `res_finetune_ablation.tex` | Supp. |
| `make_results.py` | `tables/res_effects.tex`, `res_views.tex`, `res_selection.tex` | Supp. |
| `make_results.py` | `fig/analyses.pdf`, `kin_icc.pdf`, `qualitative.pdf` | Supp. |
| `make_tables.py` | `tables/validity_by_group.tex`, `alignment_quality.tex`; `fig/dataset_qa.pdf` | Supp. |

`alignment_panels.py` holds the screen and alignment panels of Figure 2 and the figure style that `make_fig_dataset.py`
uses. `tables/stats_macros.tex` and `tables/res_macros.tex` list the numbers quoted in the text as LaTeX macros. A few
numbers of the text come from no script here: the share of marker samples missing before and after gap filling, the
lag of the movement start after the trial start, and the counts of trials without or with partial VICON coverage
(`vicon_coverage` in the manifest). `fig/qualitative.pdf` is a copy of the figure of `benchmark/e7_qualitative.py`, and
the video-alignment plot of the supplementary material is the per-video plot of `alignment/video/align_videos.py`.
Table 1, the pipeline figure, the top row of Figure 1, and the marker-set table and dataset-layout figure of the
supplementary material are made by hand.
