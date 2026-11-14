"""Core files of the ViKi-Traj release: participants, phenotype, and the beh, motion and events files of every run.

Writes under --out (default release/ViKi-Traj) the dataset-level files (description, participants, phenotype, task
sidecars) and, per participant and visit, beh.tsv and the motion and events files of every run. Reads the benchmark
manifest, the aligned VICON crops, data/game_data/<pid>.mat, the behavioural sheet and splits.json.

    python dataset/build_core.py --participant XXXXXXX103 --out release/test/core
    python dataset/build_core.py --all
"""
import sys
import warnings
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MM = common.MM_PER_GAME_PT
TOUCHES = {"object_touch_begin": "RT_begin", "object_touch_lift": "RT_lift",
           "other_touch_begin": "OT_begin", "other_touch_lift": "OT_lift"}
EVENT_ORDER = ["trial_start", "movement_start", "object_touch_begin", "object_touch_lift", "other_touch_begin",
               "other_touch_lift", "movement_end", "trial_end"]
HEIGHT_RANGE_MM = (800, 2000)        # plausible body height of the cohort; outside -> n/a (data-entry error)
# released column -> column of the behavioural sheet
ANTHRO = {"height": "Height (mm)", "weight": "Weight (kg)",
          "upper_arm_length_left": "Limb Length (mm) Upper Arm Left",
          "upper_arm_length_right": "Limb Length (mm) Upper Arm Right",
          "forearm_length_left": "Limb Length (mm) Forearm Left",
          "forearm_length_right": "Limb Length (mm) Forearm Right",
          "hand_length_left": "Limb Length (mm) Hand  Left", "hand_length_right": "Limb Length (mm) Hand Right",
          "shoulder_width": "Limb Length (mm) Shoulder"}
# SpatialAxes of the right-handed VICON lab frame (x, y, z up) from the lab axis that points to the participant's
# left: (axis index, sign) -> anatomical direction of x, y, z.
AXES_FROM_LEFT = {(1, 1): "ALS", (1, -1): "PRS", (0, 1): "LPS", (0, -1): "RAS"}
SCREEN_AXES = {"x": "horizontal touch position on the screen; origin at the screen centre, positive to the "
                    "participant's right (logged direction)",
               "y": "vertical touch position on the screen; origin at the screen centre, positive upward "
                    "(logged direction)"}
SCREEN_PLACEMENT = "iPad screen, upright in front of the participant"
TASK_DESCRIPTION = ("Reaching game: the child guides a game object with one finger along a curved path on an upright "
                    "iPad, which logs the position of the object (the end-effector trajectory); a session has 3 rounds "
                    "of 4 trials, and the hand usually alternates every two trials (hand column of beh.tsv).")
DRAG_STEP_S = 0.0125                 # sample intervals below this are steps of a continuous drag (8-9 ms logged)
ROUND_TO = {"height": -1, "weight": 0}   # decimals of the released anthropometrics (10 mm, 1 kg); others whole mm


# ----------------------------------------------------------------------------- formatting
def seconds(x):
    """Times rounded to 4 decimals, without negative zeros."""
    return np.round(np.asarray(x, float), 4) + 0.0


def fixed(values, decimals):
    """Strings with a fixed number of decimals (object array); NaN stays NaN and is written as n/a."""
    a = np.round(np.asarray(values, float), decimals) + 0.0
    out = np.char.mod(f"%.{decimals}f", a).astype(object)
    out[~np.isfinite(a)] = np.nan
    return out


def xy_mm(points):
    """(N, 2) game points (any logged shape, possibly empty) -> (N, 2) mm."""
    return np.asarray(points, float).reshape(-1, 2) * MM


# ----------------------------------------------------------------------------- sources
@lru_cache(None)
def behavior():
    """Rows of the behavioural sheet of the released participants, indexed by study ID in common.pids() order."""
    b = pd.read_csv(common.BEHAVIOR)
    return b[b["Participant ID"].isin(common.pids())].set_index("Participant ID").loc[common.pids()]


@lru_cache(None)
def game_table(pid):
    """Reaching table of data/game_data/<pid>.mat: a struct of column arrays, one element per logged trial."""
    return sio.loadmat(common.GAME / f"{pid}.mat", squeeze_me=True, struct_as_record=False)["Reaching"]


def game_trial(pid, trial_start_time):
    """Index of the logged trial with this start time (aborted attempts in the game data never match)."""
    starts = [str(s) for s in np.atleast_1d(game_table(pid).trial_start_time)]
    hits = [k for k, s in enumerate(starts) if s == str(trial_start_time)]
    if len(hits) != 1:
        raise ValueError(f"{pid}: {len(hits)} logged trials match a manifest trial start")
    return hits[0]


def runs(pid):
    """Manifest rows of a participant in game order. trial_in_visit (game round of the visit), missing for a round
    without VICON recording, is filled from the start order of the rounds within the visit."""
    m = common.manifest()
    m = m[m.pid == pid].copy()
    plain = common.paths_plain(pid)
    m["start"] = [common.game_seconds(plain[int(r)].trial_start_time) for r in m.aligned_mat_row]
    game_start = m.groupby(["visit", "group_no"]).start.transform("min")
    order = game_start.groupby(m.visit).rank(method="dense").astype(int)
    if not ((m.trial_in_visit == order) | m.trial_in_visit.isna()).all():
        raise ValueError(f"{pid}: game order disagrees with trial_in_visit")
    m["trial_in_visit"] = order
    return m.sort_values(["visit", "start"])


# ----------------------------------------------------------------------------- dataset level
def write_dataset_description(out):
    common.write_json({
        "Name": common.DATASET_NAME, "BIDSVersion": "1.10.0", "DatasetType": "raw",
        "License": "ViKi-Traj review terms (see LICENSE)", "Authors": ["Anonymous (double-blind review)"],
        "HowToAcknowledge": "Please cite the paper (citation withheld for double-blind review); see CITATION.cff.",
        "EthicsApprovals": ["Institutional Review Board of the authors' institution (name and protocol number "
                            "withheld for double-blind review); written parental consent and child assent"],
        "GeneratedBy": [{"Name": "ViKi-Traj release builders", "Version": common.VERSION,
                         "Description": "Conversion of the de-identified end-effector trajectory files and gap-filled VICON recordings "
                                        "to Motion-BIDS"}]}, out / "dataset_description.json")


def split_of():
    """{study ID: 'test' or 'fold-k'} from outputs/benchmark/splits.json."""
    s = common.benchlib.splits()
    split = {p: "test" for p in s["test"]}
    split.update({p: f"fold-{i + 1}" for i, fold in enumerate(s["folds"]) for p in fold})
    return split


def write_participants(out):
    b, split = behavior(), split_of()
    group = common.manifest().groupby("pid").group.first()
    if not (b.Condition.str.strip() == group.loc[b.index]).all():
        raise ValueError("group of the manifest disagrees with the behavioural sheet")
    df = pd.DataFrame({"participant_id": [common.sub_id(p) for p in b.index],
                       "group": group.loc[b.index].values,
                       "age": b["Age at PPT"].astype(int).values,
                       "affected_arm": b["Paretic (UCP)/ Nondominant (TD) UL"].str.strip().str.lower().values,
                       "split": [split[p] for p in b.index]})
    common.write_tsv(df, out / "participants.tsv")
    common.write_json({
        "participant_id": {"Description": "Participant label (sub-01 ... sub-29)."},
        "group": {"Description": "Diagnostic group.",
                  "Levels": {"TD": "typically developing", "UCP": "unilateral cerebral palsy"}},
        "age": {"Description": "Age at participation, whole years.", "Units": "year"},
        "affected_arm": {"Description": "The paretic arm (UCP) or the non-dominant arm (TD).",
                         "Levels": {"left": "left arm", "right": "right arm"}},
        "split": {"Description": "Benchmark split of the participant: held-out test set or one of the five "
                                 "cross-validation folds of the remaining participants.",
                  "Levels": {"test": "held-out test participant",
                             **{f"fold-{i}": f"cross-validation fold {i}" for i in range(1, 6)}}},
    }, out / "participants.json")


def write_phenotype(out):
    b = behavior()
    df = pd.DataFrame({"participant_id": [common.sub_id(p) for p in b.index]})
    for col, src in ANTHRO.items():
        v = b[src].astype(float).values
        if col == "height":
            v = np.where((v >= HEIGHT_RANGE_MM[0]) & (v <= HEIGHT_RANGE_MM[1]), v, np.nan)
        df[col] = fixed(np.round(v, ROUND_TO.get(col, 0)), 0)
    common.write_tsv(df, out / "phenotype" / "anthropometrics.tsv")
    length = {"Units": "mm"}
    coarse = "to reduce the risk of re-identification (quasi-identifier)"
    common.write_json({
        "MeasurementToolMetadata": {"Description": "Body height, body weight and segment lengths of both arms, "
                                                   "measured once per participant. Height is rounded to 10 mm and "
                                                   f"weight to 1 kg {coarse}; lengths are whole mm."},
        "participant_id": {"Description": "Participant label (sub-01 ... sub-29)."},
        "height": {"Description": f"Body height, rounded to the nearest 10 mm {coarse}. n/a when not recorded or "
                                  f"outside the plausible range {HEIGHT_RANGE_MM[0]}-{HEIGHT_RANGE_MM[1]} mm "
                                  "(data-entry error).", **length},
        "weight": {"Description": f"Body weight, rounded to the nearest 1 kg {coarse}.", "Units": "kg"},
        **{f"{seg}_length_{side}": {"Description": f"{name} length, {side} arm.", **length}
           for seg, name in (("upper_arm", "Upper arm"), ("forearm", "Forearm"), ("hand", "Hand"))
           for side in ("left", "right")},
        "shoulder_width": {"Description": "Shoulder width.", **length},
    }, out / "phenotype" / "anthropometrics.json")


def write_sidecars(out):
    run_time = "run time: seconds since the movement start of the trial (sample 0 of the motion files)"
    common.write_json({
        "onset": {"Description": f"Time of the event in {run_time}; negative before the movement start.",
                  "Units": "s"},
        "duration": {"Description": "Duration of the event; 0 (instantaneous events).", "Units": "s"},
        "trial_type": {
            "Description": "Event type. Touch events are copied from the trajectory file: begin and lift events are "
                           "logged independently and need not pair up (several fingers, or a touch that began "
                           "before the trial); the last object lift often follows trial_end.",
            "Levels": {"trial_start": "the path appears on the screen",
                       "movement_start": "first end-effector sample (the object starts to follow the finger)",
                       "object_touch_begin": "a touch begins on the object",
                       "object_touch_lift": "a touch on the object is lifted",
                       "other_touch_begin": "a touch begins elsewhere on the screen",
                       "other_touch_lift": "a touch elsewhere on the screen is lifted",
                       "movement_end": "last end-effector sample",
                       "trial_end": "end of the trial as logged by the game"}},
        "x": {"Description": "Touch or end-effector position: " + SCREEN_AXES["x"] + ". n/a for trial_start "
                             "and trial_end.", "Units": "mm"},
        "y": {"Description": "Touch or end-effector position: " + SCREEN_AXES["y"] + ". n/a for trial_start "
                             "and trial_end.", "Units": "mm"},
    }, out / f"task-{common.TASK}_events.json")
    session_time = "session time: seconds since the first trial start of the session"
    common.write_json({
        "TaskName": common.TASK, "TaskDescription": TASK_DESCRIPTION,
        "run": {"Description": "Run label: trial order within the session, run = (round - 1) x 4 + trial of the "
                               "round; 3 game rounds of 4 trials per session."},
        "hand": {"Description": "Hand used for the trial.", "Levels": {"left": "left hand", "right": "right hand"}},
        "arm_role": {"Description": "Role of the arm used.",
                     "Levels": {"affected": "paretic arm (UCP) or non-dominant arm (TD)",
                                "less_affected": "the other arm"}},
        "trial_start": {"Description": f"Start of the trial (path appears), {session_time}.", "Units": "s"},
        "movement_start": {"Description": f"First end-effector sample, {session_time}.", "Units": "s"},
        "movement_end": {"Description": f"Last end-effector sample, {session_time}.", "Units": "s"},
        "movement_time": {"Description": "movement_end - movement_start.", "Units": "s"},
        "time_to_movement": {"Description": "movement_start - trial_start.", "Units": "s"},
        "path_length": {"Description": "Length of the end-effector trajectory on the screen (sum of the distances "
                                       "between consecutive samples of the ipad motion file).", "Units": "mm"},
        "n_object_touches": {"Description": "Number of object_touch_begin events of the trial (trajectory file). A touch "
                                            "already on the screen when the trial starts has no begin event, so 0 "
                                            "is common although the object was moved."},
        "n_other_touches": {"Description": "Number of other_touch_begin events of the trial (touches that began "
                                           "elsewhere on the screen, trajectory file)."},
    }, out / f"task-{common.TASK}_beh.json")


# ----------------------------------------------------------------------------- per run
def ipad_track(path_row, t0):
    """(latency in run time s, (N, 2) positions in mm) of the end-effector trajectory, in logged order."""
    lat = common.game_seconds(path_row.trial_start_time) + np.atleast_1d(path_row.RMov_time_step_sec) - t0
    return seconds(lat), xy_mm(path_row.RMov_path)


def drag_rate(lat):
    """Sampling rate while the finger drags (Hz, 1 decimal; about 120 Hz): 1 / mean of the latency intervals in
    (0, DRAG_STEP_S), which leaves out pauses of the drag, repeated latencies and clock steps back."""
    d = np.diff(lat)
    d = d[(d > 0) & (d < DRAG_STEP_S)]
    if not len(d):
        raise ValueError("no latency interval of a continuous drag")
    return round(1.0 / float(d.mean()), 1)


def latency_note(lat, xy):
    """LatencyNote of an iPad motion.json for latencies that step back or repeat (None when strictly increasing).
    Sample numbers are 0-based rows of motion.tsv."""
    d = np.diff(lat)
    back, same = np.flatnonzero(d < 0), np.flatnonzero(d == 0)
    if not len(back) and not len(same):
        return None
    parts = [f"the game clock steps back at sample {i + 1} (by {float(-d[i]):.4f} s) while the positions continue "
             "smoothly" for i in back]
    parts += [f"samples {i} and {i + 1} share the latency {float(lat[i]):.4f} s (positions "
              f"{float(np.linalg.norm(xy[i + 1] - xy[i])):.1f} mm apart)" for i in same]
    return ("Latency is kept as logged (sample numbers are 0-based rows of motion.tsv): " + "; ".join(parts)
            + ". The samples are kept in logged order, so latency is not strictly increasing.")


def write_ipad(motion_dir, name, lat, xy):
    common.write_tsv(pd.DataFrame({"latency": lat, "x": fixed(xy[:, 0], 2), "y": fixed(xy[:, 1], 2)}),
                     motion_dir / name("motion", ".tsv"), header=False)
    common.write_tsv(pd.DataFrame({
        "name": ["latency", "x", "y"], "component": ["n/a", "x", "y"], "type": ["LATENCY", "POS", "POS"],
        "tracked_point": ["n/a", "touch", "touch"], "units": ["s", "mm", "mm"],
        "placement": ["n/a", SCREEN_PLACEMENT, SCREEN_PLACEMENT],
        "description": ["time of the sample in run time (s since the movement start)", SCREEN_AXES["x"],
                        SCREEN_AXES["y"]]}), motion_dir / name("channels", ".tsv"))
    # SpatialAxes: x right, y up (superior), no z ('_'). SamplingFrequencyEffective: rate while dragging.
    meta = {"TaskName": common.TASK, "TaskDescription": TASK_DESCRIPTION, "TrackingSystemName": "iPad",
            "Manufacturer": "Apple", "ManufacturersModelName": "iPad Pro 12.9-inch", "SamplingFrequency": "n/a",
            "SamplingFrequencyEffective": drag_rate(lat),
            "RecordingDuration": float(np.nanmax(lat)), "MotionChannelCount": 3, "POSChannelCount": 2,
            "LATENCYChannelCount": 1, "TrackedPointsCount": 1, "SpatialAxes": "RS_", "RotationRule": "n/a"}
    note = latency_note(lat, xy)
    if note:
        meta["LatencyNote"] = note
    common.write_json(meta, motion_dir / name("motion", ".json"))


def vicon_window(v, move_duration, t0, at_recording_start):
    """VICON crop on the movement window: W, (N, 24, 3) mm in common.MARKERS order, and lead, its leading n/a samples.

    N = floor(movement_time * 100) + 1, sample k at run time k / 100 s, NaN where a marker is missing or the recording
    does not cover the window. A crop that starts at frame 0 of its recording (at_recording_start) starts late when
    the recording started after the movement start, and is shifted by lead samples; any other crop starts at the
    frame nearest to the movement start (within half a frame). Crop frames past the window are dropped.
    """
    X = np.full((len(v["t"]), len(common.MARKERS), 3), np.nan)
    for j, mk in enumerate(common.MARKERS):
        if mk in v["labels"]:
            X[:, j] = v["X"][:, v["labels"].index(mk)]
    lead = max(0, int(np.round((v["t"][0] - t0) * common.VICON_HZ))) if at_recording_start else 0
    n = int(np.floor(move_duration * common.VICON_HZ + 1e-6)) + 1
    kept = max(0, min(len(X), n - lead))
    W = np.full((n, len(common.MARKERS), 3), np.nan)
    W[lead:lead + kept] = X[:kept]
    return W, lead


def session_axes(pid, rows):
    """SpatialAxes of the VICON lab frame for a session, from the median shoulder line (LSHO - RSHO points to the
    participant's left) over the session's VICON runs; z is up."""
    mk = common.benchlib.markers(pid)[0]
    with warnings.catch_warnings():                                 # runs whose shoulder markers are all missing
        warnings.simplefilter("ignore", RuntimeWarning)
        lines = [np.nanmedian(mk[r]["X"][:, mk[r]["labels"].index("LSHO"), :2]
                              - mk[r]["X"][:, mk[r]["labels"].index("RSHO"), :2], axis=0)
                 for r in rows if r in mk and {"LSHO", "RSHO"} <= set(mk[r]["labels"])]
        left = np.nanmedian(lines, axis=0) if lines else np.full(2, np.nan)
    if not np.isfinite(left).all():
        return "__S"
    k = int(np.argmax(np.abs(left)))
    return AXES_FROM_LEFT[(k, int(np.sign(left[k])))]


def write_vicon(motion_dir, name, W, axes, start_frame, recording):
    common.write_tsv(pd.DataFrame(fixed(W.reshape(len(W), -1), 2)), motion_dir / name("motion", ".tsv"),
                     header=False)
    common.write_tsv(pd.DataFrame([{"name": f"{mk}_{c}", "component": c, "type": "POS", "tracked_point": mk,
                                    "units": "mm", "placement": common.MARKER_PLACEMENT[mk]}
                                   for mk in common.MARKERS for c in "xyz"]), motion_dir / name("channels", ".tsv"))
    common.write_json({"TaskName": common.TASK, "TaskDescription": TASK_DESCRIPTION, "TrackingSystemName": "VICON",
                       "Manufacturer": "Vicon", "SamplingFrequency": common.VICON_HZ,
                       "RecordingDuration": float(seconds(len(W) / common.VICON_HZ)),
                       "MotionChannelCount": W.shape[1] * 3, "POSChannelCount": W.shape[1] * 3,
                       "TrackedPointsCount": W.shape[1], "SpatialAxes": axes, "RotationRule": "n/a",
                       "StartFrameInRecording": start_frame, "Recording": recording,
                       "MissingPercentage": round(100.0 * float(np.isnan(W).mean()), 2)},
                      motion_dir / name("motion", ".json"))


def touch_events(log, k, t0):
    """[(onset in run s, trial_type, x mm, y mm)] of the object and other touches of logged trial k."""
    rows = []
    for kind, field in TOUCHES.items():
        times = np.atleast_1d(getattr(log, field + "_time")[k])
        xy = xy_mm(getattr(log, field + "_pos")[k])
        if len(times) != len(xy):
            raise ValueError(f"{field}: {len(times)} times but {len(xy)} positions")
        rows += [(common.game_seconds(s) - t0, kind, x, y) for s, (x, y) in zip(times, xy)]
    return rows


def write_events(path, path_row, log, k, t0, xy):
    g = common.game_seconds
    rows = [(g(path_row.trial_start_time) - t0, "trial_start", np.nan, np.nan),
            (0.0, "movement_start", *xy[0]),
            (g(path_row.move_end_time) - t0, "movement_end", *xy[-1]),
            (g(path_row.trial_end_time) - t0, "trial_end", np.nan, np.nan)] + touch_events(log, k, t0)
    ev = pd.DataFrame(rows, columns=["onset", "trial_type", "x", "y"])
    ev["onset"] = seconds(ev.onset)
    ev = ev.assign(order=ev.trial_type.map(EVENT_ORDER.index)).sort_values(["onset", "order"], kind="stable")
    common.write_tsv(pd.DataFrame({"onset": ev.onset, "duration": 0.0, "trial_type": ev.trial_type,
                                   "x": fixed(ev.x, 2), "y": fixed(ev.y, 2)}), path)


def beh_row(pid, r, path_row, log, k, xy):
    g, s0 = common.game_seconds, common.session_t0(pid, r.visit)
    start, move0, move1 = g(path_row.trial_start_time), g(path_row.move_start_time), g(path_row.move_end_time)
    return {"run": common.run_id(r.trial_in_visit, r.path), "hand": r.hand.lower(),
            "arm_role": common.benchlib.limb_of(pid, r.hand),
            "trial_start": start - s0, "movement_start": move0 - s0, "movement_end": move1 - s0,
            "movement_time": move1 - move0, "time_to_movement": move0 - start,
            "path_length": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "n_object_touches": np.atleast_1d(log.RT_begin_time[k]).size,
            "n_other_touches": np.atleast_1d(log.OT_begin_time[k]).size}


def build_participant(pid, out):
    plain, log = common.paths_plain(pid), game_table(pid)
    vicon = common.benchlib.markers(pid)[0]
    m = runs(pid)
    sub = common.sub_id(pid)
    for visit, rows in m.groupby("visit"):
        ses = common.ses_id(visit)
        motion_dir = common.subses_dir(out, sub, ses, "motion")
        beh, axes = [], session_axes(pid, [int(x) for x in rows.aligned_mat_row])
        for r in rows.itertuples():
            run = common.run_id(r.trial_in_visit, r.path)
            row = int(r.aligned_mat_row)
            path_row, t0 = plain[row], common.run_t0(pid, row)
            k = game_trial(pid, path_row.trial_start_time)
            lat, xy = ipad_track(path_row, t0)
            write_ipad(motion_dir, partial(common.bids_name, sub, ses, tracksys="ipad", run=run), lat, xy)
            if row in vicon:
                W, lead = vicon_window(vicon[row], float(path_row.move_duration_s), t0, r.vicon_start_frame == 0)
                write_vicon(motion_dir, partial(common.bids_name, sub, ses, tracksys="vicon", run=run), W, axes,
                            start_frame=int(r.vicon_start_frame) - lead,
                            recording=common.bids_name(sub, ses, "vicon", ".c3d", acq=f"rec{r.trial_in_visit}"))
            write_events(motion_dir / common.bids_name(sub, ses, "events", ".tsv", run=run), path_row, log, k, t0, xy)
            beh.append(beh_row(pid, r, path_row, log, k, xy))
        df = pd.DataFrame(beh).sort_values("run")
        for col in ("trial_start", "movement_start", "movement_end", "movement_time", "time_to_movement"):
            df[col] = seconds(df[col])
        df["path_length"] = fixed(df.path_length, 2)
        common.write_tsv(df, common.subses_dir(out, sub, ses, "beh") / common.bids_name(sub, ses, "beh", ".tsv"))
    print(f"{pid} -> {sub}: {len(m)} runs, {sum(int(x) in vicon for x in m.aligned_mat_row)} with VICON")


def main():
    args = common.cli(__doc__.splitlines()[0]).parse_args()
    out = Path(args.out)
    write_dataset_description(out)
    write_participants(out)
    write_phenotype(out)
    write_sidecars(out)
    for pid in common.selected(args):
        build_participant(pid, out)


if __name__ == "__main__":
    main()
