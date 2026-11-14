"""Benchmark manifest: one row per trial with its aligned VICON data, top and side clips and validity flags.

Reads vicon_alignment_diagnostics.csv, aligned_paths.csv, trajectory_check.csv, video/video_segments_all.csv and
video/clip_durations.csv and writes benchmark_manifest.csv, all in outputs/game_vicon_aligned/reaching/. File paths in
the manifest are relative to the repository root; aligned_mat_row is the 1-based row of outputTable in aligned_mat_path.
A false flag has its reasons in <stream>_invalid_reason. The flags are described in alignment/video/README.md.

    python alignment/build_benchmark_manifest.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "outputs" / "game_vicon_aligned" / "reaching"
VIDEO = ALIGNED / "video"
RAW = ROOT / "data" / "raw_video_data"
MAX_ERR = 0.1           # s, alignment error accepted
MAX_DUR_FRAMES = 2      # clip duration vs movement window, in frames (inclusive end frame + rounding + drift)
MAX_DRIFT = 0.002       # |video clock / game clock - 1|; camera clocks drift by < 0.001
MIN_COVERAGE = 0.999    # fraction of the movement window covered by VICON or video


def rel(p):
    return str(Path(p).relative_to(ROOT))


def game_vicon():
    d = pd.read_csv(ALIGNED / "vicon_alignment_diagnostics.csv")
    d = d[d.kept == 1].copy()
    d["aligned_mat_row"] = d.groupby("pid").cumcount() + 1              # outputTable keeps this order
    p = pd.read_csv(ALIGNED / "aligned_paths.csv")
    p["aligned_mat_row"] = p.groupby("pid").cumcount() + 1
    d = d.merge(p[["pid", "aligned_mat_row", "vicon_round", "finger_nan_pct", "any_marker_nan_pct"]],
                on=["pid", "aligned_mat_row"], how="left")
    d["vicon_align_err_s"] = (d.vicon_offset_sec - d.consensus_offset_sec).abs()     # vs the recording's consensus
    # trajectory test (trajectory_check.py): the VICON fingertip must reproduce the game trajectory in time
    # (|lag| <= 0.1 s) and space (also with the screen mapping of the other trials of its recording)
    t = pd.read_csv(ALIGNED / "trajectory_check.csv")[
        ["pid", "aligned_mat_row", "lag_s", "rmse_self_mm", "rmse_lopo_mm", "trajectory_align_pass",
         "trajectory_align_reason"]]
    d = d.merge(t.rename(columns={"lag_s": "trajectory_lag_s", "rmse_self_mm": "trajectory_rmse_self_mm",
                                  "rmse_lopo_mm": "trajectory_rmse_trial_mm"}),
                on=["pid", "aligned_mat_row"], how="left")
    has = d.total_frames > 0
    checks = [("no VICON data", has),
              ("review: ", d.review_status == "OK"),
              ("partial VICON coverage", d.vicon_coverage >= MIN_COVERAGE),
              ("offset not verified", d.align_support >= 2),
              (f"alignment error >= {MAX_ERR} s", d.vicon_align_err_s < MAX_ERR),
              ("game-trajectory test: ", d.trajectory_align_pass.fillna(False).astype(bool))]
    detail = {"review: ": d.review_reason, "game-trajectory test: ": d.trajectory_align_reason}
    why = []
    for i in d.index:
        r = [n + (str(detail[n][i]) if n in detail else "") for n, ok in checks if not ok[i]]
        if "no VICON data" in r:
            r = ["no VICON data"]
        why.append("; ".join(r))
    d["vicon_invalid_reason"] = why
    d["vicon_valid"] = d.vicon_invalid_reason == ""
    return d


def views():
    s = pd.read_csv(VIDEO / "video_segments_all.csv")
    c = pd.read_csv(VIDEO / "clip_durations.csv")[["pid", "clip", "clip_frames", "diff_with_game_duration"]]
    s = s.merge(c, on=["pid", "clip"], how="left")
    s["clip_path"] = [rel(VIDEO / p / c) for p, c in zip(s.pid, s["clip"])]
    s["clip_exists"] = [(ROOT / p).exists() for p in s.clip_path]
    s["source_video"] = [rel(RAW / f) for f in s.video_file]
    return s


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    d = game_vicon()
    s = views()
    vtrial = d.vicon_trial.fillna("").str.split(" | ", regex=False).str[0]

    pid = d.pid
    out = pd.DataFrame({
        # pid: 7-character prefix + study number; pid.str[7] is its first digit (1xx = TD, 2xx = UCP)
        "pid": pid, "motor_impaired": pid.str[7] == "2", "visit": d.visit,
        # trial_in_visit: round of the visit (VICON recording 1-3); path: trial number in the round (trial_no 1-4)
        "trial_in_visit": d.vicon_round, "path": d.trial_no, "hand": d.trial_hand,
        "group_no": d.group_no, "vicon_trial": vtrial,
        "aligned_mat_path": [rel(ALIGNED / f"{p}_reaching_aligned.mat") for p in pid],
        "aligned_mat_file": [f"{p}_reaching_aligned.mat" for p in pid],
        "aligned_mat_row": d.aligned_mat_row,
        "game_raw_path": [rel(ROOT / "data" / "game_data" / f"{p}.mat") for p in pid],
        "vicon_raw_path": [rel(ROOT / "data" / "vicon_data" / f"{p}_VICON.mat") for p in pid],
        "move_duration_s": d.move_duration_s,
        "vicon_start_frame": d.vicon_start_frame, "vicon_frames": d.total_frames,
        "vicon_coverage": d.vicon_coverage, "vicon_align_source": d.align_source,
        "vicon_align_support": d.align_support, "vicon_align_err_s": d.vicon_align_err_s,
        "trajectory_lag_s": d.trajectory_lag_s, "trajectory_rmse_self_mm": d.trajectory_rmse_self_mm,
        "trajectory_rmse_trial_mm": d.trajectory_rmse_trial_mm, "trajectory_align_pass": d.trajectory_align_pass,
        "finger_nan_pct": d.finger_nan_pct, "any_marker_nan_pct": d.any_marker_nan_pct,
        "review_status": d.review_status,
        "vicon_valid": d.vicon_valid, "vicon_invalid_reason": d.vicon_invalid_reason,
    }).reset_index(drop=True)
    key = out[["pid", "aligned_mat_row"]]

    for view in ("top", "side"):
        v = s[s.view == view].rename(columns={"row": "aligned_mat_row"})
        v = key.merge(v, on=["pid", "aligned_mat_row"], how="left")
        present = v.clip_path.notna()
        err = v.residual_s.abs()                          # refined start vs the video's line
        checks = [("no clip", present & v.clip_exists.eq(True)),
                  ("ambiguous video match", v.video_margin >= 0.1),       # score margin over lags > 1 s away
                  ("partial video coverage", v.coverage >= MIN_COVERAGE),
                  (f"alignment error >= {MAX_ERR} s", (err < MAX_ERR) & np.isfinite(v.path_score)),
                  ("implausible clock drift", (v.drift_slope - 1).abs() <= MAX_DRIFT),
                  ("clip duration mismatch", v.diff_with_game_duration.abs() <= MAX_DUR_FRAMES / v.fps + 1e-6)]
        why = []
        for i in range(len(v)):
            r = [n for n, ok in checks if not bool(ok.iloc[i])]
            why.append("no clip" if "no clip" in r else "; ".join(r))
        out[f"{view}_clip_path"] = v.clip_path
        out[f"{view}_clip_file"] = v.clip_path.map(lambda p: Path(p).name, na_action="ignore")
        out[f"{view}_source_video"] = v.source_video
        out[f"{view}_start_frame"] = v.start_frame
        out[f"{view}_n_frames"] = v.n_frames
        out[f"{view}_fps"] = v.fps
        out[f"{view}_align_err_s"] = err.values
        out[f"{view}_valid"] = [w == "" for w in why]
        out[f"{view}_invalid_reason"] = why

    vv = out[["top_valid", "side_valid"]]
    out["valid_views"] = [";".join(n for n, ok in zip(("top", "side"), r) if ok) for r in vv.values]
    out["n_valid_views"] = vv.sum(axis=1)
    out["benchmark_valid"] = out.vicon_valid & (out.n_valid_views > 0)
    out = out.sort_values(["pid", "visit", "trial_in_visit", "path"])
    out.to_csv(ALIGNED / "benchmark_manifest.csv", index=False, float_format="%.4f")

    print(f"wrote {rel(ALIGNED / 'benchmark_manifest.csv')}: {len(out)} paths, {out.pid.nunique()} participants")
    for c in ("vicon_valid", "top_valid", "side_valid", "benchmark_valid"):
        print(f"  {c:16s} {int(out[c].sum()):4d}")
    print("  view combinations of VICON-valid paths:")
    print(out[out.vicon_valid].valid_views.replace("", "(none)").value_counts().to_string())


if __name__ == "__main__":
    main()
