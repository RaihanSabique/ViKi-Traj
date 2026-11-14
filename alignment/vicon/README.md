# VICON alignment

MATLAB code that places the VICON recordings on the clock of the reaching game and crops the VICON
markers of every trial over its movement window. The game data call a trial a path (`trial_no` 1-4);
each visit has three rounds of four trials, one VICON recording per round (`Reaching 1` to `Reaching 3`).
Written for MATLAB R2023a; no toolbox is needed.

```bash
cd alignment/vicon
matlab -batch "participants={'XXXXXXX115'}; run_alignment"   # one participant
matlab -batch "participants='all'; run_alignment"            # every data/vicon_data/XXXXXXX*_VICON.mat
matlab -batch "export_paths_plain; summarize_aligned; export_markers; export_vicon_trial_frames"
matlab -batch "plot_review"                                  # optional: recordings with an unverified offset
```

Inputs, relative to the repository root:

- `data/game_data/<pid>.mat`: table `Reaching` of the game, one row per trial
  (`preprocessing/export_game_data.py`)
- `data/vicon_data/<pid>_VICON.mat`: one element per VICON recording with `game`, `trial` (recording
  name), `time`, `framerate`, `labels` and `position_data_interpolated` (3 x markers x frames, mm)
- `data/Vicon Label Marker Tracker Update.xlsx`, sheet `Frame Times`: one row per participant with the
  start of each trial marked during marker labelling (s and 100 Hz frame)

Outputs in `outputs/game_vicon_aligned/reaching/`:

| File | Script | Content |
|---|---|---|
| `<pid>_reaching_aligned.mat` | `run_alignment.m` | `outputTable`: game columns, VICON crop, alignment columns |
| `vicon_alignment_diagnostics.csv` | `run_alignment.m` | offset estimates and review reason per trial |
| `vicon_alignment_summary.csv` | `run_alignment.m` | counts per participant |
| `<pid>_reaching_paths_plain.mat` | `export_paths_plain.m` | plain struct per trial for Python, no markers |
| `aligned_paths.csv` | `summarize_aligned.m` | per trial: round (`vicon_round`), VICON coverage, missing markers |
| `aligned_participants.csv` | `summarize_aligned.m` | VICON coverage and missing markers per participant |
| `vicon_trial_frames.csv` | `export_vicon_trial_frames.m` | frame count of every VICON recording |
| `review_plots/` | `plot_review.m` | plots of the recordings with an unverified offset |

`outputTable` has one row per trial; its columns are described in
`outputTable.Properties.VariableDescriptions`. `export_markers.m` writes the markers of every trial to
`outputs/benchmark/markers/<pid>_reaching_markers.mat`. `vicon_trial_frames.csv` feeds only two internal
statistics of `paper/make_tables.py` (number and total length of the recordings), which the paper does not quote.

## Method

- Movement window: a trial is cropped from the first to the last logged sample of the game object
  (`RMov_time`; `move_start_time`, `move_end_time`), not from `trial_start_time` to `trial_end_time`.
  The object starts moving 0.07 s after the trial start at the median, and more than 1 s later in
  11 % of the trials. `time` is game time from the start of trial 1 of the recording.
- Two offset estimates per trial: the manual start from the marker-labelling sheet, and the
  cross-correlation of finger-marker speed (RFIN or LFIN, by `trial_hand`) with game-object speed
  (`xcorr_offset.m`), which counts when r >= 0.7. The sheet marks either the trial start or the
  movement start; both readings are tried and the one that agrees with its recording is kept
  (`manual_mark`).
- The four trials of a recording share one clock offset. All estimates of the recording vote, and
  the offset shared within 0.1 s by the most estimates is the consensus.
- Fallback order (`align_source`): `manual` if the manual start of the trial agrees with the
  consensus, else `xcorr` if its cross-correlation agrees, else `trial_consensus`. `spike`
  (`spike_offset.m`: the finger lift after trial 1, biased by about +1 s) is used only when the
  recording has no other estimate.
- Review: `align_support` counts the estimates behind the consensus; 1 means that nothing agreed and
  the offset is unverified. `review_status` is `Need Review` when the offset is unverified, VICON
  covers less than 90 % of the movement window, the finger marker of the active hand is missing in
  more than 10 % of its frames, or (manual-only participants) the manual start is more than 0.5 s off
  the median of its recording.
- Every trial is kept; only aborted attempts (an incomplete group of trials restarted later in the
  same visit) are dropped. A trial without VICON data (no recording, no offset, or outside the
  recording) keeps its game data with empty VICON arrays and `vicon_coverage = 0`.
- Manual-only participants (`manualOnlyParticipants` in `run_alignment.m`): four participants with
  UCP whose finger speed does not follow the game object closely enough. Only their manual starts
  are used; a trial without one takes the median manual offset of its recording
  (`manual_trial_median`), and a recording without any manual start falls back to the vote.

The visit comes from the recording name (`... V2 Trial 03`). The trials are grouped by `study_day`
and restarts of `trial_no`, and the groups of a visit are matched to its recordings in chronological
order (by the number in the game name if the counts differ).
