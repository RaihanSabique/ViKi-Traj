"""Source data of the ViKi-Traj release: de-identified end-effector trajectory files and full VICON recordings.

Writes under --out (default release/ViKi-Traj), per visit, one trajectory JSON (the kept fields of
data/game_data/<pid>.mat; times in session seconds, positions in game points) and one C3D file per VICON recording
(acq-rec<k>, raw positions, gaps invalid), and updates sourcedata/vicon/recordings.tsv (runs and movement windows).

    python dataset/build_sourcedata.py --participant XXXXXXX103 --out release/test/sourcedata
    python dataset/build_sourcedata.py --all
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common      # noqa: E402
import build_core  # noqa: E402  (vicon_window, the movement window of the VICON motion files)
import c3d         # noqa: E402  (py-c3d 0.5.2: pip install --no-deps c3d==0.5.2)

# ----------------------------------------------------------------------------- fields of the game data
# Logged field -> (kind, description, units). kind: time (absolute -> session s), xy (one point),
# xys (list of points), value (number).
FIELDS = {
    "trial_no": ("value", "Trial within the game round (1-4); run = (round - 1) x 4 + trial_no", "n/a"),
    "trial_start_time": ("time", "Trial start logged by the game", "s"),
    "trial_end_time": ("time", "Trial end logged by the game (in most trials the time of the last trajectory_xy "
                               "sample)", "s"),
    "time_diff_s": ("value", "Trial duration logged by the game (trial_end_time - trial_start_time)", "s"),
    "road_path": ("xys", "Road polyline the object has to follow", "pt"),
    "target_pos": ("xy", "Target position", "pt"),
    "rocket_pos": ("xy", "Object position at trial start, as logged; its y is 39.03 pt below the object position "
                         "of the other_touch_*_object_pos fields (see Notes)", "pt"),
    "RT_begin_pos": ("xys", "Object touches: position where each touch on the object began", "pt"),
    "RT_begin_time": ("time", "Object touches: time each touch on the object began", "s"),
    "RT_lift_pos": ("xys", "Object touches: position where each touch on the object was lifted", "pt"),
    "RT_lift_time": ("time", "Object touches: time each touch on the object was lifted", "s"),
    "RMov_path": ("xys", "Object movement: finger position samples while dragging the object (the object follows "
                         "the finger), irregular, about 120 Hz", "pt"),
    "RMov_time": ("time", "Object movement: time of each trajectory_xy sample", "s"),
    "RMov_dist_px": ("value", "Length of the trajectory_xy polyline, in points (the game names it px)", "pt"),
    "OT_begin_pos": ("xys", "Other touches (not on the object): position where each began", "pt"),
    "OT_begin_time": ("time", "Other touches: time each began", "s"),
    "OT_begin_rocket_pos": ("xys", "Other touches: object position when each began", "pt"),
    "OT_lift_pos": ("xys", "Other touches: position where each was lifted", "pt"),
    "OT_lift_time": ("time", "Other touches: time each was lifted", "s"),
    "OT_lift_rocket_pos": ("xys", "Other touches: object position when each was lifted", "pt"),
    "OTMov_path": ("xys", "Other-touch movement: finger position samples of the other touches", "pt"),
    "OTMov_time": ("time", "Other-touch movement: time of each other_touch_xy sample", "s"),
    "OTMov_rocket_pos": ("xys", "Other-touch movement: object position at each other_touch_xy sample", "pt"),
}
SCALAR_TIMES = ("trial_start_time", "trial_end_time")
# Released name of each logged field: the game object is called "object", its logged movement the trajectory.
RENAME = {"rocket_pos": "object_pos", "RT_begin_pos": "object_touch_begin_pos", "RT_begin_time": "object_touch_begin_time",
          "RT_lift_pos": "object_touch_lift_pos", "RT_lift_time": "object_touch_lift_time",
          "RMov_path": "trajectory_xy", "RMov_time": "trajectory_time", "RMov_dist_px": "trajectory_length",
          "OT_begin_pos": "other_touch_begin_pos", "OT_begin_time": "other_touch_begin_time",
          "OT_begin_rocket_pos": "other_touch_begin_object_pos", "OT_lift_pos": "other_touch_lift_pos",
          "OT_lift_time": "other_touch_lift_time", "OT_lift_rocket_pos": "other_touch_lift_object_pos",
          "OTMov_path": "other_touch_xy", "OTMov_time": "other_touch_time",
          "OTMov_rocket_pos": "other_touch_object_pos"}
# Not released: identifying fields, and fields that others carry (trial_hand is released as hand;
# RMov_time_step_ms / _sec are RMov_time - trial_start_time in ms / s).
DROPPED = ("pid", "user_id", "age", "study_location", "study_day", "dob", "Properties", "trial_hand",
           "RMov_time_step_ms", "RMov_time_step_sec")
NOTES = [
    "The game's per-sample times relative to the trial start are omitted: they equal trajectory_time - "
    "trial_start_time. The game's trial_hand is given as hand.",
    "object_pos and the other_touch_*_object_pos fields use different reference points on the object: before the "
    "object is first dragged, every other_touch_*_object_pos equals object_pos + (0, 39.03) pt (39.03 pt = 7.51 mm), "
    "and the y of object_pos is 39.03 pt below the y of the previous path's target_pos in the same game. Add "
    "(0, 39.03) to object_pos to express it in the reference point of the other_touch_*_object_pos fields.",
    "Field names: 'object' is the game object the finger drags (object_pos, object_touch_*); 'trajectory' is its "
    "logged movement (trajectory_xy, trajectory_time, trajectory_length); 'other_touch' fields are touches "
    "elsewhere on the screen.",
]

POINTS_PER_INCH = round(25.4 / common.MM_PER_GAME_PT)
SCREEN = {
    "Device": f"12.9-inch iPad Pro, landscape, {POINTS_PER_INCH} points per inch",
    "SizePoints": list(common.SCREEN_PT),
    "SizeMillimetres": [round(s * common.MM_PER_GAME_PT, 2) for s in common.SCREEN_PT],
    "MillimetresPerPoint": round(common.MM_PER_GAME_PT, 6),
    "Origin": "screen centre",
    "XAxis": "+x towards the right edge of the screen as seen by the participant",
    "YAxis": "+y towards the top edge of the screen; the screen stood upright in front of the participant, "
             "so +y points up",
}
TIME_ORIGIN = "session time: seconds since the first trial start of the session (trial_start_time of run 1 = 0)"

# ----------------------------------------------------------------------------- VICON recordings
# Residual written for every measured point: the source holds no residuals, and a C3D residual of 0 would mark
# the point as modelled (interpolated); -1 marks a gap.
MEASURED_RESIDUAL = 1.0
REC_COLUMNS = {
    "file": "C3D file, relative to the dataset root. Raw VICON marker trajectories (no gap filling), 100 Hz, mm, "
            "VICON lab frame (Z up); points ordered as the 24 released markers; gaps and markers absent from the "
            "recording are invalid points (residual -1, coordinates 0). Residuals were not exported, so every "
            f"measured point carries the placeholder residual {MEASURED_RESIDUAL:.1f} (a positive residual marks "
            "measured data; no point is interpolated or modelled)",
    "participant_id": "Participant (sub-NN)",
    "session": "Session (visit): ses-1 or ses-2",
    "acq": "Recording within the session, in recording order (rec1, rec2, ...); one round of the reaching game "
           "(4 runs) per recording",
    "n_frames": "Number of frames in the recording (100 Hz)",
    "runs": "Comma-separated run indices recorded (the run-NN entity of the released files)",
    "run_start_frames": "Per run in runs: 0-based frame of the recording at the run's movement start, i.e. of "
                        "sample 0 of its vicon motion file (= StartFrameInRecording of its motion.json; C3D frame "
                        "number = value + 1). Negative when the recording started after the movement start; n/a "
                        "when the run has no VICON samples (and no vicon motion file)",
    "run_n_frames": "Per run in runs: length of the run's movement window in frames (= samples of its vicon motion "
                    "file); sample k of the motion file is recording frame start + k. Only frames 0 to n_frames - 1 "
                    "are recorded, so a window that starts before or ends after the recording is recorded only in "
                    "part; n/a when the run has no VICON samples",
}
FORBIDDEN = re.compile(r"XXXXXXX|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}:\d{2}")     # study IDs, dates, clock times


def check_deidentified(text, what, extra=()):
    """Raise if a study ID, date, clock time or other participant token appears in released text."""
    hits = FORBIDDEN.findall(text) + [t for t in extra if t and t in text]
    if hits:
        raise ValueError(f"identifying content {sorted(set(hits))[:3]} in {what}")


# ----------------------------------------------------------------------------- manifest
def session_rows(pid):
    """Manifest rows of a participant with their run index, sorted by session and run.

    The game round of a trial comes from group_no (running round index over both visits), because trial_in_visit
    is missing for rounds without VICON recording; both agree wherever both exist."""
    m = common.manifest()
    m = m[m.pid == pid].copy()
    game = m.groupby("visit").group_no.rank(method="dense")
    known = m.trial_in_visit.notna()
    if not (game[known] == m.trial_in_visit[known]).all():
        raise ValueError(f"{pid}: game order from group_no differs from trial_in_visit")
    m["run"] = [common.run_index(g, p) for g, p in zip(game, m.path)]
    return m.sort_values(["visit", "run"])


# ----------------------------------------------------------------------------- game data
def load_game(pid):
    """Reaching table of data/game_data/<pid>.mat; fails on a field that neither FIELDS nor DROPPED lists."""
    ab = sio.loadmat(common.GAME / f"{pid}.mat", squeeze_me=True, struct_as_record=False)["Reaching"]
    unknown = set(ab._fieldnames) - set(FIELDS) - set(DROPPED)
    if unknown:
        raise ValueError(f"{pid}: undocumented game fields {sorted(unknown)}")
    return ab


def game_index(pid, ab):
    """{aligned_mat_row: index of that trial in the Reaching struct}, matched on trial_start_time."""
    starts = [str(s).strip() for s in np.atleast_1d(ab.trial_start_time)]
    if len(set(starts)) != len(starts):
        raise ValueError(f"{pid}: duplicate trial_start_time in the game data")
    return {row: starts.index(str(e.trial_start_time).strip()) for row, e in common.paths_plain(pid).items()}


def session_seconds(ts, t0):
    """Session time (s, 4 decimals) of logged timestamp strings (empty cell -> [])."""
    ts = np.atleast_1d(ts)
    if ts.size == 0:
        return []
    return np.round(pd.to_datetime(ts.astype(str)).asi8 / 1e9 - t0, 4).tolist()


def convert(kind, units, value, t0):
    if kind == "time":
        return session_seconds(value, t0)
    if kind == "xys":
        return np.round(np.asarray(value, float).reshape(-1, 2), 2).tolist()
    if kind == "xy":
        return np.round(np.asarray(value, float).reshape(2), 2).tolist()
    v = float(value)
    return None if np.isnan(v) else round(v, 2 if units == "pt" else 4)


def trial_entry(ab, i, r, t0):
    """One released trial: run, hand and every kept logged field."""
    if str(ab.trial_hand[i]) != r.hand or int(ab.trial_no[i]) != r.path:
        raise ValueError(f"game row {i} does not match manifest run {r.run}")
    out = {"run": int(r.run), "hand": r.hand.lower()}
    for name, (kind, _, units) in FIELDS.items():
        out[RENAME.get(name, name)] = convert(kind, units, getattr(ab, name)[i], t0)
    for name in SCALAR_TIMES:
        out[name] = out[name][0]
    out["trial_no"] = int(out["trial_no"])
    return out


def trajectory_file(ab, rows, index, t0):
    return {
        "TaskName": common.TASK,
        "Description": "De-identified end-effector trajectory file of one session of the reaching game: every "
                       "released trial (run) with the game's logged fields (object trajectory, touches, path and "
                       "target). Identifiers, dates and clock times are removed.",
        "Screen": SCREEN,
        "TimeOrigin": TIME_ORIGIN,
        "PositionUnits": "pt (game points, see Screen)",
        "Fields": {"run": {"Description": "Run index (the run-NN entity of the released files)", "Units": "n/a"},
                   "hand": {"Description": "Hand that played the trial: left | right", "Units": "n/a"},
                   **{RENAME.get(k, k): {"Description": d, "Units": u} for k, (_, d, u) in FIELDS.items()}},
        "Notes": NOTES,
        "Trials": [trial_entry(ab, index[int(r.aligned_mat_row)], r, t0) for r in rows.itertuples()],
    }


def dumps_compact(obj):
    """JSON with indent 2 and every list of numbers on one line (one point per line in point lists)."""
    num = r"(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|null)"
    text = json.dumps(obj, indent=2)
    return re.sub(rf"\[\s*{num}(?:,\s*{num})*\s*\]", lambda m: re.sub(r"\s+", " ", m.group(0))
                  .replace("[ ", "[").replace(" ]", "]"), text) + "\n"


def write_trajectories(pid, ab, m, out, private):
    """One trajectory file per session; private = participant tokens that must not appear in the output."""
    index = game_index(pid, ab)
    files = []
    for visit, rows in m.groupby("visit"):
        sub, ses = common.sub_id(pid), common.ses_id(visit)
        text = dumps_compact(trajectory_file(ab, rows, index, common.session_t0(pid, visit)))
        path = (common.subses_dir(out / "sourcedata" / "trajectory", sub, ses)
                / common.bids_name(sub, ses, "trajectory", ".json"))
        check_deidentified(text, path.name, private)
        path.write_text(text)
        files.append(path)
    return files


# ----------------------------------------------------------------------------- VICON
class GapWriter(c3d.Writer):
    """c3d.Writer for float point data without analog channels whose invalid points are written as
    (0, 0, 0, residual -1); the writer of c3d 0.5.2 leaves the previous frame's coordinates there."""

    def _write_frames(self, handle):
        pts = np.stack([p for p, _ in self._frames])
        raw = np.where(pts[..., 3:4] < 0, np.array([0, 0, 0, -1]), pts[..., :4])
        handle.write(raw.astype("<f4").tobytes())
        self._pad_block(handle)


def load_vicon(pid):
    """The recordings of <pid>_VICON.mat, in recording order."""
    v = sio.loadmat(common.VICON_RAW / f"{pid}_VICON.mat", squeeze_me=True, struct_as_record=False)
    (key,) = [k for k in v if not k.startswith("__")]
    return list(np.atleast_1d(v[key]))


def session_recordings(recs, rows):
    """[(recording, its manifest rows)] of one session, in recording order (k = 1, 2, ...)."""
    names = [str(r.trial) for r in recs]
    found = []
    for name, rr in rows.dropna(subset=["vicon_trial"]).groupby("vicon_trial"):
        idx = [i for i, n in enumerate(names) if n == name]
        if len(idx) != 1 or not str(recs[idx[0]].game).lower().startswith("reaching") or rr.group_no.nunique() != 1:
            raise ValueError("ambiguous VICON recording for a manifest game")
        if int(recs[idx[0]].framerate) != common.VICON_HZ:
            raise ValueError("unexpected VICON frame rate")
        found.append((idx[0], rr))
    return [(recs[i], rr) for i, rr in sorted(found, key=lambda x: x[0])]


def marker_frames(rec):
    """(n_frames, 24, 3) raw positions (mm) in common.MARKERS order; NaN = gap or marker absent."""
    labels = [str(s).strip() for s in np.atleast_1d(rec.labels)]
    P = np.asarray(rec.position_data, float)
    X = np.full((P.shape[2], len(common.MARKERS), 3), np.nan)
    for j, m in enumerate(common.MARKERS):
        if m in labels:
            X[:, j] = P[:, labels.index(m)].T
    return X


def write_c3d(path, X):
    w = GapWriter(point_rate=common.VICON_HZ, point_scale=-1.0, point_units="mm")
    w.set_point_labels(common.MARKERS)
    desc, width = c3d.Writer.pack_labels([common.MARKER_PLACEMENT[m] for m in common.MARKERS])
    w.point_group.add_str("DESCRIPTIONS", "Marker placement", desc, width, len(common.MARKERS))
    w.set_screen_axis("+X", "+Z")
    ok = np.isfinite(X).all(2)
    pts = np.zeros(X.shape[:2] + (5,), np.float32)
    pts[..., :3] = np.where(ok[..., None], X, 0)
    pts[..., 3] = np.where(ok, MEASURED_RESIDUAL, -1)
    no_analog = np.zeros((0, 0), np.float32)
    w.add_frames([(p, no_analog) for p in pts])
    with open(path, "wb") as f, warnings.catch_warnings():
        warnings.filterwarnings("ignore", "No analog data found in file.")
        w.write(f)


def check_c3d(path, X, extra):
    """Read the file back: same labels, rate, frames and data (gaps invalid), no identifier in the parameters.
    The parameters are parsed by c3d.Reader; the point block is read directly, because the frame iterator of
    the c3d 0.5.2 Reader overflows under numpy 2 (TRIAL:ACTUAL_END_FIELD)."""
    data = path.read_bytes()
    with open(path, "rb") as f, warnings.catch_warnings():
        warnings.filterwarnings("ignore", "No analog data found in file.")
        r = c3d.Reader(f)
        # C3D files are 512-byte blocks: header (block 1), parameters, then the data from block header.data_block
        start = 512 * (r.header.data_block - 1)
        n = int(r.get("POINT:FRAMES").uint16_value)
        labels = [s.strip() for s in r.point_labels]
        rate, used = float(r.point_rate), int(r.point_used)
    check_deidentified(data[512:start].decode("latin-1"), path.name, extra)
    pts = np.frombuffer(data, "<f4", count=n * used * 4, offset=start).reshape(n, used, 4)
    ok = np.isfinite(X).all(2)
    if labels != common.MARKERS or rate != common.VICON_HZ or n != len(X) \
            or not np.array_equal(pts[..., 3] == MEASURED_RESIDUAL, ok) \
            or (pts[~ok] != [0, 0, 0, -1]).any() or np.abs(pts[ok][:, :3] - X[ok]).max(initial=0) > 1e-3:
        raise ValueError(f"{path.name}: read-back mismatch")


def movement_window(pid, r):
    """(start, length) in frames of a run's movement window in its recording, from build_core.vicon_window:
    start = 0-based recording frame of the movement start (negative when the recording started late), length =
    samples of the VICON motion file. None when the run has no VICON crop."""
    row = int(r.aligned_mat_row)
    crop = common.benchlib.markers(pid)[0].get(row)
    if crop is None:
        return None
    if len(crop["t"]) != r.vicon_frames:
        raise ValueError(f"run {r.run}: VICON crop length differs from the manifest")
    window, lead = build_core.vicon_window(crop, float(common.paths_plain(pid)[row].move_duration_s),
                                           common.run_t0(pid, row), r.vicon_start_frame == 0)
    return int(r.vicon_start_frame) - lead, len(window)


def run_windows(pid, rr, n_frames):
    """Comma-separated runs of one recording, and the start frame and length of each run's movement window.
    Every run with VICON frames must have a crop that lies inside the recording (vicon_start_frame, vicon_frames)."""
    starts, counts = [], []
    for r in rr.sort_values("run").itertuples():
        win = movement_window(pid, r)
        if (win is None and r.vicon_frames > 0) or (win and r.vicon_start_frame + r.vicon_frames > n_frames):
            raise ValueError(f"run {r.run}: VICON crop missing or beyond the recording")
        starts.append("n/a" if win is None else str(win[0]))
        counts.append("n/a" if win is None else str(win[1]))
    return ",".join(map(str, sorted(rr.run))), ",".join(starts), ",".join(counts)


def write_recordings(pid, m, out, private):
    """Write the session recordings of one participant; returns their recordings.tsv rows."""
    recs = load_vicon(pid)
    private = list(private) + [str(r.trial) for r in recs]
    rows = []
    for visit, srows in m.groupby("visit"):
        sub, ses = common.sub_id(pid), common.ses_id(visit)
        for k, (rec, rr) in enumerate(session_recordings(recs, srows), 1):
            X = marker_frames(rec)
            path = common.subses_dir(out / "sourcedata" / "vicon", sub, ses) / \
                common.bids_name(sub, ses, "vicon", ".c3d", acq=f"rec{k}")
            write_c3d(path, X)
            check_c3d(path, X, private)
            runs, starts, counts = run_windows(pid, rr, len(X))
            rows.append({"file": common.rel(path, out), "participant_id": sub, "session": ses, "acq": f"rec{k}",
                         "n_frames": len(X), "runs": runs, "run_start_frames": starts, "run_n_frames": counts})
    return rows


def update_recordings_tsv(rows, built_subs, out):
    """recordings.tsv: the rows of this call plus the kept rows of the participants not built in it."""
    path = out / "sourcedata" / "vicon" / "recordings.tsv"
    new = pd.DataFrame(rows, columns=list(REC_COLUMNS))
    if path.exists():
        old = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        new = pd.concat([old[~old.participant_id.isin(built_subs)], new.astype(str)])
    new = new.sort_values(["participant_id", "session", "acq"])
    check_deidentified(new.to_csv(sep="\t", index=False), path.name)
    common.write_tsv(new, path)
    common.write_json({k: {"Description": d} for k, d in REC_COLUMNS.items()}, path.with_suffix(".json"))
    return path


def main():
    args = common.cli(__doc__.splitlines()[0]).parse_args()
    out = args.out.resolve()
    rows, subs = [], []
    for pid in common.selected(args):
        m, ab = session_rows(pid), load_game(pid)
        private = [str(u) for u in np.atleast_1d(ab.user_id)]
        logs = write_trajectories(pid, ab, m, out, private)
        recs = write_recordings(pid, m, out, private)
        rows += recs
        subs.append(common.sub_id(pid))
        print(f"{common.sub_id(pid)}: {len(logs)} trajectory files, {len(recs)} VICON recordings", flush=True)
    print(f"wrote {update_recordings_tsv(rows, subs, out)}")


if __name__ == "__main__":
    main()
