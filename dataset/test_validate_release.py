"""Test of validate_release.py on two synthetic release trees, written under --out with their scrub-terms file.

clean/ is a mini release of two made-up participants, sub-01 (TD) and sub-29 (UCP, runs without VICON data or a clip);
violations/ is a copy with planted problems (study ID in a TSV, date in a JSON, wrong column count, ...). The checker
must pass clean/ with --partial, report the participant count without it, and catch every planted problem.

    python dataset/test_validate_release.py [--out release/test/validate]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import c3d  # noqa: E402

CHECKER = Path(__file__).resolve().parent / "validate_release.py"
RNG = np.random.default_rng(0)
SUBJECTS = {"sub-01": ("TD", "left"), "sub-29": ("UCP", "right")}     # group, affected arm
SCRUB_TERMS = ["XXXXXXX", "Jane Doe"]       # a study-ID prefix and a person name, both planted in violations/
EPOCH = 1500000000.25                    # an absolute time in Unix seconds, as common.game_seconds returns


# ----------------------------------------------------------------------------- clean tree
def manifest_rows():
    """Manifest rows (the columns validate_release.py requires) of the two participants, 12 runs per session. sub-01
    has VICON data and both clips in every run; sub-29 has no top clip in runs 1-4, no side clip in runs 9-12 and, in
    ses-2, no VICON data in runs 9-12 and a VICON recording that covers only part of run 3."""
    rows = []
    for sub, (group, affected) in SUBJECTS.items():
        for ses in ("ses-1", "ses-2"):
            for k in range(1, 13):
                run, mt = f"run-{k:02d}", round(float(RNG.uniform(3.5, 16.0)), 3)
                hand = "right" if k % 4 in (1, 2) else "left"
                gaps = sub == "sub-29"
                clip = {"top": not (gaps and k <= 4), "side": not (gaps and k >= 9)}
                coverage = (0.0 if k >= 9 else 0.75 if k == 3 else 1.0) if gaps and ses == "ses-2" else 1.0
                vicon_ok = coverage == 1.0 and k != 1
                ok = {v: clip[v] and k != 2 for v in common.VIEWS}
                vicon_reason = ("no VICON recording" if coverage == 0 else "VICON recording ends early" if coverage < 1
                                else "fingertip marker missing > 10 %")
                row = {"participant_id": sub, "session": ses, "run": run, "group": group, "hand": hand,
                       "arm_role": "affected" if hand == affected else "less_affected", "movement_time": mt,
                       "vicon_valid": vicon_ok, "vicon_invalid_reason": None if vicon_ok else vicon_reason,
                       "vicon_align_source": "manual" if coverage > 0 else "no_vicon_trial",
                       "vicon_align_support": 6 if coverage > 0 else 0,
                       "vicon_align_err": 0.012 if coverage > 0 else None, "vicon_coverage": coverage,
                       "trajectory_qa_failed": False}
                for v in common.VIEWS:
                    reason = None if ok[v] else "alignment error >= 0.1 s" if clip[v] else "no clip"
                    row.update({f"{v}_valid": ok[v], f"{v}_invalid_reason": reason,
                                f"{v}_align_err": 0.02 if clip[v] else None,
                                f"{v}_n_frames": round(mt * 30) + 1 if clip[v] else None,
                                f"{v}_fps": 30.0 if clip[v] else None,
                                f"{v}_video": video_rel(sub, ses, run, v) if clip[v] else None})
                views = [v for v in common.VIEWS if ok[v]]
                row.update({"benchmark_valid": vicon_ok and bool(views), "valid_views": ";".join(views) or None,
                            "events": f"{sub}/{ses}/motion/" + common.bids_name(sub, ses, "events", ".tsv", run=run)})
                rows.append(row)
    man = pd.DataFrame(rows)
    flags = ["vicon_valid", "trajectory_qa_failed", "benchmark_valid"] + [f"{v}_valid" for v in common.VIEWS]
    man[flags] = man[flags].apply(lambda c: c.map({True: "true", False: "false"}))       # as build_derivatives.py
    return man.astype({f"{v}_n_frames": "Int64" for v in common.VIEWS} | {"vicon_align_support": "Int64"})


def video_rel(sub, ses, run, view):
    return f"{common.VIDEOS}/{sub}/{ses}/" + common.bids_name(sub, ses, "video", ".mp4", acq=view, run=run)


def build_clean(out):
    if out.exists():
        shutil.rmtree(out)
    man = manifest_rows()
    write_root(out)
    recs = []
    for (sub, ses), runs in man.groupby(["participant_id", "session"]):
        write_beh(out, sub, ses, runs)
        start, windows = 0, []
        for row in runs.itertuples():
            n_vicon = write_run(out, row, start)
            window = (str(start), str(n_vicon)) if n_vicon else ("n/a", "n/a")
            windows.append((str(int(row.run[4:])), *window))
            start += n_vicon + 50
        recs.append(write_sourcedata(out, sub, ses, start, windows))
    common.write_tsv(pd.DataFrame(recs), out / "sourcedata/vicon/recordings.tsv")
    write_derivatives(out, man)


def write_root(out):
    out.mkdir(parents=True)
    (out / "README.md").write_text("---\nlicense: cc-by-nc-4.0\npretty_name: ViKi-Traj\n---\n\n# ViKi-Traj (test)\n")
    common.write_json({"Name": "ViKi-Traj", "BIDSVersion": "1.10.0", "DatasetType": "raw"}, out / "dataset_description.json")
    parts = pd.DataFrame({"participant_id": ["sub-01", "sub-29"], "group": ["TD", "UCP"], "age": [9, 11],
                          "affected_arm": ["left", "right"], "split": ["fold-1", "test"]})
    common.write_tsv(parts, out / "participants.tsv")
    common.write_json({c: {"Description": c} for c in parts.columns}, out / "participants.json")
    anthro = pd.DataFrame({c: [1.0, 2.0] for c in ["height", "weight", "upper_arm_length_left", "upper_arm_length_right",
                                                   "forearm_length_left", "forearm_length_right", "hand_length_left",
                                                   "hand_length_right", "shoulder_width"]})
    anthro.insert(0, "participant_id", ["sub-01", "sub-29"])
    common.write_tsv(anthro, out / "phenotype/anthropometrics.tsv")
    common.write_json({c: {"Description": c} for c in anthro.columns}, out / "phenotype/anthropometrics.json")
    common.write_json({c: {"Description": c} for c in ["trial_type", "x", "y"]}, out / f"task-{common.TASK}_events.json")
    beh_columns = ["run", "hand", "arm_role", "trial_start", "movement_start", "movement_end", "movement_time",
                   "time_to_movement", "path_length", "n_object_touches", "n_other_touches"]
    common.write_json({c: {"Description": c} for c in beh_columns}, out / f"task-{common.TASK}_beh.json")


def write_beh(out, sub, ses, runs):
    t0 = np.arange(len(runs)) * 20.0
    beh = pd.DataFrame({"run": runs.run.values, "hand": runs.hand.values, "arm_role": runs.arm_role.values,
                        "trial_start": t0, "movement_start": t0 + 1.5, "movement_end": t0 + 1.5 + runs.movement_time.values,
                        "movement_time": runs.movement_time.values, "time_to_movement": 1.5, "path_length": 250.0,
                        "n_object_touches": 1, "n_other_touches": 0})
    common.write_tsv(beh, out / sub / ses / "beh" / common.bids_name(sub, ses, "beh", ".tsv"))


def write_run(out, row, start):
    """events, iPad motion and (with VICON data) VICON motion of one run; returns the VICON sample count."""
    sub, ses, run, mt = row.participant_id, row.session, row.run, float(row.movement_time)
    d = out / sub / ses / "motion"
    events = pd.DataFrame({"onset": [-1.5, -0.1, 0.0, mt, mt + 0.1], "duration": 0.0,
                           "trial_type": ["trial_start", "object_touch_begin", "movement_start", "movement_end", "trial_end"],
                           "x": [np.nan, 10.0, np.nan, np.nan, np.nan], "y": [np.nan, -20.0, np.nan, np.nan, np.nan]})
    common.write_tsv(events, d / common.bids_name(sub, ses, "events", ".tsv", run=run))
    lat = np.cumsum(np.r_[0, RNG.uniform(0.007, 0.0097, int(mt * 120))])
    ipad = np.c_[lat, RNG.uniform(-90, 90, (len(lat), 2))]                     # mm, on the screen
    name = lambda sfx, ext, ts: d / common.bids_name(sub, ses, sfx, ext, tracksys=ts, run=run)  # noqa: E731
    common.write_tsv(pd.DataFrame(ipad), name("motion", ".tsv", "ipad"), header=False)
    common.write_tsv(pd.DataFrame({"name": ["latency", "x", "y"], "component": ["n/a", "x", "y"],
                                   "type": ["LATENCY", "POS", "POS"], "tracked_point": ["n/a", "fingertip", "fingertip"],
                                   "units": ["s", "mm", "mm"]}), name("channels", ".tsv", "ipad"))
    common.write_json({"TaskName": common.TASK, "TrackingSystemName": "iPad", "SamplingFrequency": "n/a",
                       "SamplingFrequencyEffective": round(1 / float(np.median(np.diff(lat))), 4),
                       "RecordingDuration": round(float(lat[-1]), 4), "MotionChannelCount": 3, "SpatialAxes": "RS_"}, name("motion", ".json", "ipad"))
    if not row.vicon_coverage > 0:
        return 0
    n = int(round(mt * common.VICON_HZ))
    pos = RNG.normal(0, 300, (n, 72))
    pos[RNG.random(pos.shape) < 0.01] = np.nan
    common.write_tsv(pd.DataFrame(pos), name("motion", ".tsv", "vicon"), header=False)
    chans = pd.DataFrame({"name": [f"{m}_{a}" for m in common.MARKERS for a in "xyz"],
                          "component": list("xyz") * 24, "type": "POS",
                          "tracked_point": np.repeat(common.MARKERS, 3), "units": "mm",
                          "placement": np.repeat([common.MARKER_PLACEMENT[m] for m in common.MARKERS], 3)})
    common.write_tsv(chans, name("channels", ".tsv", "vicon"))
    common.write_json({"TaskName": common.TASK, "TrackingSystemName": "VICON", "SamplingFrequency": common.VICON_HZ,
                       "RecordingDuration": n / common.VICON_HZ, "MotionChannelCount": 72, "POSChannelCount": 72,
                       "SpatialAxes": "ALS", "RotationRule": "n/a", "StartFrameInRecording": start,
                       "Recording": common.bids_name(sub, ses, "vicon", ".c3d", acq="rec1"), "MissingPercentage": 1.0},
                      name("motion", ".json", "vicon"))
    return n


def write_sourcedata(out, sub, ses, frames, windows):
    """Trajectory file and one C3D recording of the session; returns its recordings.tsv row (as build_sourcedata)."""
    common.write_json({"trials": [{"run": 1, "trial_start": 0.0}]},
                      out / "sourcedata/trajectory" / sub / ses / common.bids_name(sub, ses, "trajectory", ".json"))
    rel = f"sourcedata/vicon/{sub}/{ses}/" + common.bids_name(sub, ses, "vicon", ".c3d", acq="rec1")
    write_c3d(out / rel, max(frames, 1), "n/a")
    runs, starts, counts = zip(*windows)
    return {"file": rel, "participant_id": sub, "session": ses, "acq": "rec1", "n_frames": max(frames, 1),
            "runs": ",".join(runs), "run_start_frames": ",".join(starts), "run_n_frames": ",".join(counts)}


def write_c3d(path, frames, subject_name):
    w = c3d.Writer(point_rate=float(common.VICON_HZ))
    w.set_point_labels(common.MARKERS)
    pts = np.zeros((len(common.MARKERS), 5), np.float32)
    w.add_frames([(pts, np.zeros((0, 0), np.float32))] * frames)
    w.add_group(10, "SUBJECTS", "Subject").add_str("NAMES", "Subject names", subject_name, len(subject_name), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        w.write(fh)


def write_derivatives(out, man):
    d = out / common.DERIV
    common.write_json({"Name": "ViKi-Traj derivatives", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
                       "GeneratedBy": [{"Name": "test_validate_release.py"}]}, d / "dataset_description.json")
    common.write_tsv(man, d / "manifest.tsv")
    (d / ".bidsignore").write_text("manifest.tsv\nmanifest.json\nsplits.json\ncameras/\nlabels2d/\nkeypoints2d/\nresults/\ntracks3d/\n")
    common.write_json({c: {"Description": c} for c in man.columns}, d / "manifest.json")
    common.write_json({"test": ["sub-29"], "fold-1": ["sub-01"]}, d / "splits.json")
    common.write_tsv(pd.DataFrame({"participant_id": ["sub-01"], "mpjpe_px": [12.3]}), d / "results/E1_2d.tsv")
    for ses in ("ses-1", "ses-2"):
        for view in ("top", "side"):
            runs = sorted(man[(man.participant_id == "sub-01") & (man.session == ses)].run)
            common.write_json({"View": view, "Description": "test camera", "Cameras": [
                {"Runs": runs, "ImageWidth": 1920, "ImageHeight": 1080, "ProjectionMatrix": np.eye(3, 4).tolist(),
                 "CalibrationOK": True, "MarkersInFrontShare": 1.0}]},
                              d / "cameras" / f"sub-01_{ses}_acq-{view}_camera.json")
    for row in man.itertuples():
        sub, ses, run = row.participant_id, row.session, row.run
        for view in common.VIEWS:
            n, fps = getattr(row, f"{view}_n_frames"), getattr(row, f"{view}_fps")
            if pd.isna(n):
                continue
            write_clip(out / getattr(row, f"{view}_video"), int(n), float(fps))
            if sub == "sub-01" and row.vicon_coverage > 0:
                write_frame_tables(d, sub, ses, run, view, int(n), float(fps))
    tracks = pd.DataFrame({"time": [0.0, 0.0], "source": ["vicon", "vitpose_b"], "joint": "wrist",
                           "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]})
    common.write_tsv(tracks, d / "tracks3d/sub-01/ses-1" / common.bids_name("sub-01", "ses-1", "tracks3d", ".tsv", run=1))


def write_frame_tables(d, sub, ses, run, view, n, fps):
    frame = np.arange(n)
    base = {"frame": frame, "time": frame / fps}
    labels = pd.DataFrame({**base, **{f"{p}_{c}": RNG.uniform(0, 1000, n) if c != "visible" else 1
                                      for p in common.JOINT_SET + common.MARKERS for c in ("x", "y", "visible")}})
    common.write_tsv(labels, d / "labels2d" / sub / ses / common.bids_name(sub, ses, "labels", ".tsv", acq=view, run=run))
    kp = pd.DataFrame({**base, **{f"{j}_{c}": RNG.uniform(0, 1, n) for j in common.JOINT_SET for c in ("x", "y", "score")}})
    common.write_tsv(kp, d / "keypoints2d/vitpose_b" / sub / ses /
                     common.bids_name(sub, ses, "keypoints", ".tsv", acq=view, run=run))


def write_clip(path, frames, fps, metadata=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = [a for k, v in metadata for a in ("-metadata", f"{k}={v}")] or ["-map_metadata", "-1"]
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=64x48:rate={fps}", "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    *meta, str(path)], check=True)


# ----------------------------------------------------------------------------- planted violations
def plant(clean, out):
    """Copy clean/ to out/ and plant problems; returns [(description, check, path fragment, message fragment)]."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(clean, out)
    d, m1 = out / common.DERIV, "sub-01/ses-1/motion/sub-01_ses-1_task-reaching"
    planted = []

    def edit(rel, old, new):
        replace_once(out / rel, old, new)

    man = pd.read_csv(d / "manifest.tsv", sep="\t", keep_default_na=False)
    reason = man.loc[0, "vicon_invalid_reason"]
    edit(f"{common.DERIV}/manifest.tsv", f"\t{reason}\t" if reason else "\tn/a\t", "\treview: XXXXXXX218 V2 Trial 03\t")
    planted += [("study ID in a TSV", "deid", "manifest.tsv", "identifier 'XXXXXXX218'"),
                ("VICON trial name in a TSV", "deid", "manifest.tsv", "trial name 'V2 Trial 03'")]
    # benchmark_valid, valid_views and events are the last manifest columns; run-02: no valid view, run-03: side only
    for run, old, new in (("run-02", "false\tn/a", "true\tn/a"), ("run-03", "true\tside", "true\ttop;side")):
        events = "sub-29/ses-1/motion/" + common.bids_name("sub-29", "ses-1", "events", ".tsv", run=run)
        edit(f"{common.DERIV}/manifest.tsv", f"\t{old}\t{events}\n", f"\t{new}\t{events}\n")
    planted += [("benchmark_valid true without a valid view", "numbers", "manifest.tsv",
                 "benchmark_valid is not vicon_valid and (top_valid or side_valid): sub-29 ses-1 run-02 (true)"),
                ("valid_views naming an invalid view", "numbers", "manifest.tsv",
                 "valid_views does not list exactly the views whose <view>_valid is true: "
                 "sub-29 ses-1 run-03 ('top;side', expected 'side')")]
    p = out / f"{m1}_tracksys-vicon_run-01_motion.json"
    meta = json.loads(p.read_text())
    meta.update({"AcquisitionDate": "2001-02-03", "RecordingDuration": meta["RecordingDuration"] + 1})
    p.write_text(json.dumps(meta, indent=2))
    planted += [("date in a JSON", "deid", "run-01_motion.json", "date '2001-02-03'"),
                ("RecordingDuration inconsistent with rows", "format", "run-01_motion.json", "does not match")]
    p = out / f"{m1}_tracksys-vicon_run-02_motion.tsv"
    p.write_text("".join(line.rsplit("\t", 1)[0] + "\n" for line in p.read_text().splitlines()))
    planted.append(("wrong column count", "format", "run-02_motion.tsv", "71 columns but 72 rows in channels.tsv"))
    edit(f"{m1}_run-03_events.tsv", "\tn/a\tn/a\n", "\t\tn/a\n")
    planted.append(("empty cell instead of n/a", "format", "run-03_events.tsv", "empty cell"))
    beh = "sub-29/ses-1/beh/sub-29_ses-1_task-reaching_beh.tsv"
    edit(beh, "\t0.0000\t", "\t12:34:56\t")
    planted.append(("clock time in a TSV", "deid", beh, "clock time '12:34:56'"))
    (out / f"{m1}_run-04_events.tsv").unlink()
    planted.append(("missing events file", "completeness", "run-04_events.tsv", "missing"))
    (d / "results/E1_2d.tsv").rename(d / "results/E1_2d_20010203.tsv")
    planted.append(("date in a file name", "deid", "E1_2d_20010203.tsv", "in the file name"))
    with open(out / "README.md", "a") as fh:
        fh.write("Built from /home/someone/project/data/raw_video_data.\n")
    planted.append(("absolute path in README", "deid", "README.md", "path ' /home'"))
    edit("participants.tsv", "\t9\t", "\t9.5\t")
    planted.append(("non-integer age", "format", "participants.tsv", "age: 1 values outside"))
    edit(f"{common.DERIV}/splits.json", '"sub-29"', "NaN")
    planted.append(("NaN in a JSON", "format", "splits.json", "invalid JSON"))
    c3d_rel = "sourcedata/vicon/sub-29/ses-2/sub-29_ses-2_task-reaching_acq-rec1_vicon.c3d"
    write_c3d(out / c3d_rel, 100, "Jane Doe")
    planted += [("person name in a C3D parameter", "deid", c3d_rel,
                 "identifier 'Jane Doe' in C3D parameter SUBJECTS:NAMES"),
                ("VICON window beyond the C3D", "format", "sub-29/ses-2", "reach outside the 100 frames of the C3D")]
    clip = next(iter(sorted((out / common.VIDEOS / "sub-01/ses-1").glob("*acq-top*.mp4"))))
    n, fps = man.loc[man["top_video"] == f"{common.VIDEOS}/sub-01/ses-1/{clip.name}", ["top_n_frames", "top_fps"]].iloc[0]
    write_clip(clip, int(float(n)), float(fps), [("creation_time", "2001-02-03T12:34:56Z"), ("title", "XXXXXXX103 V1")])
    planted += [("date in MP4 metadata", "deid", clip.name, "date '2001-02-03' in MP4 metadata tag 'creation_time'"),
                ("study ID in MP4 metadata", "deid", clip.name, "identifier 'XXXXXXX103' in MP4 metadata tag 'title'"),
                ("MP4 metadata not stripped", "deid", clip.name, "tags ['creation_time', 'title']")]
    bad = d / "tracks3d/sub-01/ses-1/sub-01_ses-1_run-01_task-reaching_tracks3d.tsv"
    shutil.copy(d / "tracks3d/sub-01/ses-1/sub-01_ses-1_task-reaching_run-01_tracks3d.tsv", bad)
    planted.append(("entity order", "format", bad.name, "Motion-BIDS entity order"))
    return planted + plant_more(out, man)


def replace_once(path, old, new):
    text = path.read_text()
    assert old in text, (path, old)
    path.write_text(text.replace(old, new, 1))


def plant_more(out, man):
    """More planted problems (absolute times, units, vocabularies, truncated files, malformed inputs), each in a file,
    row or line that plant() leaves unchanged."""
    d, planted = out / common.DERIV, []
    m2 = "sub-01/ses-2/motion/sub-01_ses-2_task-reaching"
    vicon = sorted(p.name.split("_run-")[1][:2]                # runs of sub-01 ses-2 with VICON motion
                   for p in (out / "sub-01/ses-2/motion").glob("*tracksys-vicon_*_motion.tsv"))

    def edit(rel, old, new):
        replace_once(out / rel, old, new)

    def change(rel, modify, header=True):
        """Rewrite a TSV after modify(df) changed the table in place."""
        df = pd.read_csv(out / rel, sep="\t", header=0 if header else None, na_values=["n/a"], keep_default_na=False)
        modify(df)
        common.write_tsv(df, out / rel, header=header)

    def add_epoch(cols):
        def modify(df):
            df[cols] += EPOCH
        return modify

    rel = f"{m2}_tracksys-ipad_run-01_motion.tsv"
    change(rel, add_epoch([0]), header=False)
    planted.append(("epoch latency in iPad motion", "deid", rel, "latency: "))
    rel = "sub-01/ses-2/beh/sub-01_ses-2_task-reaching_beh.tsv"
    change(rel, add_epoch(["trial_start", "movement_start", "movement_end"]))
    planted.append(("epoch session times in beh", "deid", rel, "trial_start: 12 values >= 1e+08 s"))
    rel = f"{m2}_run-02_events.tsv"
    change(rel, add_epoch(["onset"]))
    planted += [("epoch onsets in events", "deid", rel, "onset: 5 values >= 1e+08 s"),
                ("movement_start onset not 0", "format", rel, "movement_start onsets")]
    rel = f"{m2}_run-03_events.tsv"
    edit(rel, "\tn/a\tn/a\n", "\t \tn/a\n")
    planted.append(("whitespace cell instead of n/a", "format", rel, "empty cell"))
    rel = f"{m2}_tracksys-ipad_run-04_motion.tsv"
    change(rel, lambda df: df.update(df[[1, 2]] / common.MM_PER_GAME_PT), header=False)
    planted.append(("iPad x, y in game points", "format", rel, "beyond the screen"))

    start = f"{m2}_tracksys-vicon_run-{vicon[0]}_motion.json"
    meta = json.loads((out / start).read_text())
    meta["StartFrameInRecording"] += 1
    common.write_json(meta, out / start)
    planted.append(("StartFrameInRecording one frame off recordings.tsv", "format", start, "recordings.tsv lists"))
    rel = f"{m2}_tracksys-vicon_run-{vicon[1]}_motion.tsv"
    change(rel, lambda df: df.update(df / 1000), header=False)
    planted.append(("VICON positions in m", "format", rel, "positions not in mm"))
    rel = f"{m2}_tracksys-vicon_run-{vicon[2]}_motion.json"
    (out / rel).write_text("[1, 2]\n")
    planted.append(("JSON top level not an object", "format", rel, "top level must be a JSON object"))
    rel = f"{m2}_tracksys-vicon_run-{vicon[3]}_motion.tsv"
    (out / rel).write_bytes(b"")
    planted.append(("empty VICON motion.tsv", "format", rel, "empty file"))
    for ext in ("_motion.tsv", "_channels.tsv", "_motion.json"):
        p = out / f"{m2}_tracksys-vicon_run-{vicon[4]}{ext}"
        p.rename(str(p).replace(f"_run-{vicon[4]}", ""))
    planted.append(("VICON motion without run entity", "format", f"{m2}_tracksys-vicon_motion.tsv", "entities"))

    rel = "sub-29/ses-2/beh/sub-29_ses-2_task-reaching_beh.tsv"
    edit(rel, "\tright\t", "\tRight\t")
    edit(rel, "\tless_affected\t", "\tparetic\t")
    planted += [("beh hand outside the vocabulary", "format", rel, "hand: 1 values outside"),
                ("beh arm_role outside the vocabulary", "format", rel, "arm_role: 1 values outside")]
    rel = "sub-01/ses-1/beh/sub-01_ses-1_task-reaching_beh.tsv"
    run5 = next(line for line in (out / rel).read_text().splitlines() if line.startswith("run-05\t"))
    with open(out / rel, "a") as fh:
        fh.write(f"{run5}\n{run5.replace('run-05', 'run-13')}\n")
    planted += [("duplicate beh row", "completeness", rel, "more than one row for ['run-05']"),
                ("beh row of a run not in the manifest", "completeness", rel, "runs not in manifest.tsv: ['run-13']")]

    rel = "sourcedata/vicon/sub-01/ses-2/sub-01_ses-2_task-reaching_acq-rec1_vicon.c3d"
    data = (out / rel).read_bytes()
    (out / rel).write_bytes(data[:len(data) // 2])
    planted.append(("truncated C3D", "format", rel, "truncated"))
    rel = "sourcedata/trajectory/sub-01/ses-2/sub-01_ses-2_task-reaching_trajectory.json"
    common.write_json({"trials": [{"run": 1, "trial_start": 0.0, "logged_at": EPOCH}]}, out / rel)
    planted.append(("epoch timestamp in a JSON", "deid", rel, "epoch time '1500000000.25'"))

    rel = f"{common.DERIV}/dataset_description.json"
    meta = json.loads((out / rel).read_text())
    meta["SourceDatasets"] = [{"URL": "file:///home/someone/project/outputs/benchmark"}]
    common.write_json(meta, out / rel)
    planted.append(("file:///home URI in a JSON", "deid", rel, "path ':///home'"))
    meta = json.loads((out / "participants.json").read_text())
    meta["age"]["Description"] = "Age in years at the visit of 03.02.2001"
    common.write_json(meta, out / "participants.json")
    planted.append(("day-first date in a JSON", "deid", "participants.json", "date '03.02.2001'"))
    with open(out / "README.md", "a") as fh:
        fh.write("The clip starts at timecode 01:02:03:04.\n")
    planted.append(("SMPTE timecode in README", "deid", "README.md", "clock time '01:02:03:04'"))
    (d / "checkpoints/XXXXXXX216_old").mkdir(parents=True)
    planted.append(("study ID in an empty directory name", "deid", "checkpoints/XXXXXXX216_old/", "in the directory name"))

    clip = sorted((out / common.VIDEOS / "sub-01/ses-2").glob("*acq-top*.mp4"))[0]
    n, fps = man.loc[man["top_video"] == f"{common.VIDEOS}/sub-01/ses-2/{clip.name}", ["top_n_frames", "top_fps"]].iloc[0]
    write_clip(clip, int(float(n)) - 1, float(fps))
    planted.append(("video one frame short of the manifest", "format", clip.name, f"{int(float(n)) - 1} frames, manifest"))
    return planted


# ----------------------------------------------------------------------------- run
def run_checker(root, terms, *flags):
    r = subprocess.run([sys.executable, str(CHECKER), "--root", str(root), "--scrub-terms", str(terms), *flags],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.splitlines(), r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=common.ROOT / "release/test/validate")
    args = ap.parse_args()
    build_clean(args.out / "clean")
    planted = plant(args.out / "clean", args.out / "violations")
    terms = args.out / "scrub_terms.txt"
    terms.write_text("\n".join(SCRUB_TERMS) + "\n")
    ok = True

    code, lines, err = run_checker(args.out / "clean", terms, "--partial")
    problems = [x for x in lines if x.startswith("PROBLEM")]
    print(f"clean --partial: exit {code}, {len(problems)} problems")
    print("\n".join(problems + [err] if problems or err.strip() else []))
    ok &= code == 0 and not problems

    code, lines, _ = run_checker(args.out / "clean", terms)
    counts = [x for x in lines if x.startswith("PROBLEM numbers")]
    print(f"clean (full-release mode): exit {code}; count problems:\n  " + "\n  ".join(counts))
    ok &= code == 1 and any("participants: 2, paper 29" in x for x in counts)

    code, lines, err = run_checker(args.out / "violations", terms, "--partial")
    problems = [x for x in lines if x.startswith("PROBLEM")]
    print(f"violations --partial: exit {code}, {len(problems)} problem lines")
    for what, check, path, msg in planted:
        hit = next((x for x in problems if x.split()[1] == check and path in x and msg in x), None)
        ok &= hit is not None
        print(f"  [{'caught' if hit else 'MISSED'}] {what}: {hit or (check, path, msg)}")
    explained = [x for x in problems if not any(x.split()[1] == c and p in x for _, c, p, _ in planted)]
    print("  other problem lines (side effects of the plants):\n    " + "\n    ".join(explained))
    ok &= code == 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
