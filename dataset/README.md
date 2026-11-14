# Dataset release

These scripts build the BIDS release `release/ViKi-Traj` from the raw data in `data/` and the alignment and benchmark
outputs in `outputs/`; `README_card.md` is its dataset card. `common.py` holds what the builders share: participant
IDs, BIDS file names, time origins and paths. Steps 1-5 take `--participant XXXXXXX103` or `--all`, every builder takes
`--out` (default `release/ViKi-Traj`), and all run from the repository root. Steps 1-6 do not read each other's
output; `slurm/release_build.sbatch` runs steps 1, 2, 4, 5 and 6, and `slurm/release_blur.sbatch` runs step 3 with
one participant per array task.

| Step | Script | Writes |
|---|---|---|
| 1 | `build_core.py` | dataset description, participants, phenotype, task sidecars, `sub-*/ses-*/beh/` and `motion/` |
| 2 | `build_sourcedata.py` | `sourcedata/trajectory/` (trajectory files), `sourcedata/vicon/` (raw C3D recordings) |
| 3 | `blur_clips.py` | `sourcedata/videos/`: side clips head-blurred (GPU), top clips re-encoded |
| 4 | `build_frame_tables.py` | `labels2d/`, `keypoints2d/`, `boxes/` and `tracks3d/` in `derivatives/vikitraj/` |
| 5 | `build_derivatives.py` | the rest of `derivatives/vikitraj/`: manifest, splits, cameras, results, checkpoints |
| 6 | `build_docs.py` | `README.md` (copy of `README_card.md`), `LICENSE`, `CHANGES`, `CITATION.cff` |
| 7 | `validate_release.py` | nothing; prints each problem and exits with status 1 if there is one |

De-identification: study codes are mapped to `sub-01` to `sub-29` in sorted order, and the map is saved to
`release/private/id_map.tsv`, which is not released. Calendar dates and clock times are removed (times are seconds
since the first trial start of the session or since the movement start of the trial), ages are whole years, height is
rounded to 10 mm and weight to 1 kg. In the side clips the heads of all persons are blurred (projected VICON head
markers, head keypoints of RTMPose-WB on every person found by RT-DETR, YuNet faces), and the rectangles of
`static_masks.tsv` (fractions of the frame) blur laptop screens in every frame of a session; the clips are written
without container metadata. `blur_clips.py` writes a QA table per participant to `release/private/blur_qa/`, and
`qa_contact_sheets.py` writes contact sheets of the released clips to `release/private/qa_sheets/` for a visual check.
`build_sourcedata.py` and `build_derivatives.py` stop when a file they write contains a study code, date or clock
time, and `validate_release.py` searches the whole tree again (file names, text, C3D parameters, MP4 tags, archive
members), adding the private term list given by `--scrub-terms` (default `release/private/scrub_terms.txt`).

`build_sourcedata.py`, `validate_release.py` and its test write or read C3D files with py-c3d 0.5.2. Its package
metadata requires numpy < 2, but the code runs with numpy 2, so it is installed without dependencies.

```bash
pip install --no-deps c3d==0.5.2
python dataset/build_core.py --all                  # or --participant XXXXXXX103 --out release/test/core
python dataset/build_sourcedata.py --all
python dataset/blur_clips.py --all                  # GPU; or sbatch --array=1-29 slurm/release_blur.sbatch
python dataset/build_frame_tables.py --all
python dataset/build_derivatives.py --all
python dataset/build_docs.py
python dataset/validate_release.py --root release/ViKi-Traj --scrub-terms release/private/scrub_terms.txt --jobs 8
python dataset/test_validate_release.py             # validator test on synthetic trees in release/test/validate/
```

`validate_release.py --parts core sourcedata derivatives videos` restricts the checks to some parts of the tree, and
`--partial` accepts a test tree that holds only some participants. `--scrub-terms FILE` lists terms that must not
appear anywhere in the tree, one per line; the default list is not public, so to check a copy of the release pass a
file of your own. An empty file skips only the term search; the checks for dates, clock times, epoch times, VICON
trial names and paths still run. The test writes its own term file and needs no private or internal data.
