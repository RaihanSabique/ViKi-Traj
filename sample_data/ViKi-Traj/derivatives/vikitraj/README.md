# ViKi-Traj derivatives (vikitraj)

A BIDS derivative dataset by its `dataset_description.json` only: the tables below use a custom layout, listed in
`.bidsignore`. Each table folder has a JSON sidecar at its top describing the columns. See `../../README.md`.

- `manifest.tsv/.json`: one row per trial (run): validity of every stream with the reason, alignment residuals,
  clip frames, and the paths of the trial's files.
- `splits.json`: test participants, development folds 1-5, cross-visit split.
- `cameras/`: projection matrix of every calibrated top and side video (VICON lab frame to image pixels).
- `labels2d/`: per clip frame, the VICON joints and markers projected into the image, with visibility.
- `keypoints2d/<model>/`: per clip frame, the 2D keypoints of each pose model of the paper.
- `boxes/`: per clip frame, the person box the benchmark used to crop the participant.
- `tracks3d/`: two-view 3D tracks of the active arm (VICON and models), mm, lab frame.
- `results/`: the paper's result tables (`results/README.md` defines the columns).
- `checkpoints/`: the ViconTraj-MS fine-tuned weights with their configs (`checkpoints/README.md`).

The video clips are in `../../sourcedata/videos/`.
