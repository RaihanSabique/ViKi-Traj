"""Read-only checker of a release tree: de-identification, completeness, format and the paper's counts.

Checks de-identification (scrub terms, dates, clock and epoch times, absolute paths in names, text, C3D parameters,
MP4 tags and archives), completeness against the manifest, file formats, the paper's counts and the manifest's
validity columns. Prints one line per problem, exit status 1 if any.

    python dataset/validate_release.py [--root release/ViKi-Traj] [--parts core sourcedata derivatives videos]
                                       [--partial] [--scrub-terms FILE] [--jobs 8]
"""
import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import c3d  # noqa: E402

PARTS = ("core", "sourcedata", "derivatives", "videos")
CHECKS = ("deid", "completeness", "format", "numbers")
DERIV = common.DERIV
TASK = common.TASK
PAPER = {"participants": 29, "TD": 18, "UCP": 11, "trials": 696,
         "vicon_valid": 556, "top_valid": 522, "side_valid": 488, "vicon+top+side valid": 416}

TEXT_SUFFIXES = {"", ".tsv", ".json", ".md", ".csv", ".txt", ".cff", ".yaml", ".yml", ".py", ".cfg", ".toml"}
ZIP_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".zip"}
NUMERIC_TABLES = ("_motion.tsv", "_labels.tsv", "_keypoints.tsv")      # every cell a number or n/a
RECORDINGS = "sourcedata/vicon/recordings.tsv"
KEEP_TABLES = ("_channels.tsv", "_beh.tsv", "participants.tsv", "phenotype/anthropometrics.tsv", RECORDINGS,
               f"{DERIV}/manifest.tsv")                                                 # rows kept as facts
ROOT_FILES = {"README.md", "dataset_description.json", "participants.tsv", "participants.json", "CHANGES",
              "LICENSE", "LICENSE.md", "CITATION.cff", ".bidsignore", ".gitattributes",
              f"task-{TASK}_events.json", f"task-{TASK}_beh.json"}
ROOT_DIRS = {"phenotype", "sourcedata", "derivatives"}
MP4_TAGS = {"major_brand", "minor_version", "compatible_brands", "encoder", "handler_name", "vendor_id", "language", ""}
EVENT_TYPES = {"trial_start", "object_touch_begin", "object_touch_lift", "other_touch_begin", "other_touch_lift",
               "movement_start", "movement_end", "trial_end"}

# ----------------------------------------------------------------------------- time origins, units and values
TIME_COLUMNS = {"onset", "duration", "trial_start", "movement_start", "movement_end", "movement_time",
                "time_to_movement", "time", "latency"}               # seconds from a session or run origin
EPOCH_S = 1e8               # |time| >= 1e8 s (3 years) can only be an absolute (Unix epoch) time
MAX_VISIT_S = 7200          # no session or run time exceeds a 2-h visit
SCREEN_HALF_MM = [n / 2 * common.MM_PER_GAME_PT + 0.01 for n in common.SCREEN_PT]   # iPad |x|, |y| bound (mm)
VICON_P99_MM = (50, 5000)   # 99th percentile of |marker coordinate| in mm (m, cm or game units fall outside)
BOOL = r"true|false|n/a"
RUN = r"run-(?:0[1-9]|1[0-2])"
SIDES, ROLES = r"left|right", r"affected|less_affected"
VOCAB = {   # table -> column -> allowed values (regex, whole cell)
    "participants.tsv": {"participant_id": r"sub-\d\d", "group": r"TD|UCP", "age": r"\d{1,2}",
                         "affected_arm": SIDES, "split": r"test|fold-[1-5]"},
    "_beh.tsv": {"run": RUN, "hand": SIDES, "arm_role": ROLES},
    f"{DERIV}/manifest.tsv": {"participant_id": r"sub-\d\d", "session": r"ses-[12]", "run": RUN, "group": r"TD|UCP",
                              "hand": SIDES, "arm_role": ROLES, "vicon_valid": BOOL, "trajectory_qa_failed": BOOL,
                              "benchmark_valid": BOOL, **{f"{v}_valid": BOOL for v in common.VIEWS}},
}

# ----------------------------------------------------------------------------- column rules
VIEW_COLS = [f"{v}_{c}" for v in common.VIEWS
             for c in ("valid", "invalid_reason", "align_err", "n_frames", "fps", "video")]
REQUIRED_COLUMNS = {   # "_suffix.tsv" or path -> required columns; BIDS_ORDER: these must come first, in this order
    "participants.tsv": ["participant_id", "group", "age", "affected_arm", "split"],
    "phenotype/anthropometrics.tsv": ["participant_id", "height", "weight", "upper_arm_length_left",
                                      "upper_arm_length_right", "forearm_length_left", "forearm_length_right",
                                      "hand_length_left", "hand_length_right", "shoulder_width"],
    "_beh.tsv": ["run", "hand", "arm_role", "trial_start", "movement_start", "movement_end", "movement_time",
                 "time_to_movement", "path_length", "n_object_touches", "n_other_touches"],
    "_events.tsv": ["onset", "duration", "trial_type", "x", "y"],
    "_channels.tsv": ["name", "component", "type", "tracked_point", "units"],
    f"{DERIV}/manifest.tsv": ["participant_id", "session", "run", "group", "hand", "arm_role", "movement_time",
                              "vicon_valid", "vicon_invalid_reason", "vicon_align_source", "vicon_align_support",
                              "vicon_align_err", "vicon_coverage", "trajectory_qa_failed", *VIEW_COLS, "benchmark_valid",
                              "valid_views"],
    "_labels.tsv": ["frame", "time"],
    "_keypoints.tsv": ["frame", "time"],
    "_tracks3d.tsv": ["time", "source", "joint", "x", "y", "z"],
    # the columns of build_sourcedata.py, needed to cross-check motion.json
    RECORDINGS: ["file", "participant_id", "session", "n_frames", "runs", "run_start_frames", "run_n_frames"],
}
BIDS_ORDER = {"participants.tsv": 1, "_events.tsv": 2, "_channels.tsv": 5, "_labels.tsv": 2, "_keypoints.tsv": 2}
MOTION_JSON_KEYS = {"vicon": ["TaskName", "TrackingSystemName", "SamplingFrequency", "RecordingDuration",
                              "MotionChannelCount", "POSChannelCount", "SpatialAxes", "StartFrameInRecording",
                              "Recording", "MissingPercentage"],
                    "ipad": ["TaskName", "SamplingFrequency", "SamplingFrequencyEffective", "RecordingDuration",
                             "MotionChannelCount"]}
VICON_CHANNELS = [f"{m}_{a}" for m in common.MARKERS for a in "xyz"]
IPAD_CHANNELS = [("latency", "LATENCY", "s"), ("x", "POS", "mm"), ("y", "POS", "mm")]

# ----------------------------------------------------------------------------- Motion-BIDS names
NAME_RE = re.compile(
    r"^(?P<sub>sub-[0-9A-Za-z]+)(?:_(?P<ses>ses-[0-9A-Za-z]+))?(?:_task-(?P<task>[0-9A-Za-z]+))?"
    r"(?:_tracksys-(?P<tracksys>[0-9A-Za-z]+))?(?:_acq-(?P<acq>[0-9A-Za-z]+))?(?:_run-(?P<run>[0-9]+))?"
    r"_(?P<suffix>[0-9A-Za-z]+)(?P<ext>\.[0-9a-z]+)$")
SUFFIX_RULES = {   # suffix -> (directory pattern, extensions, entities besides sub and ses)
    "beh": ("{sub}/{ses}/beh", {".tsv"}, {"task"}),
    "motion": ("{sub}/{ses}/motion", {".tsv", ".json"}, {"task", "tracksys", "run"}),
    "channels": ("{sub}/{ses}/motion", {".tsv"}, {"task", "tracksys", "run"}),
    "events": ("{sub}/{ses}/motion", {".tsv"}, {"task", "run"}),
    "trajectory": ("sourcedata/trajectory/{sub}/{ses}", {".json"}, {"task"}),
    "vicon": ("sourcedata/vicon/{sub}/{ses}", {".c3d"}, {"task", "acq"}),
    "video": (common.VIDEOS + "/{sub}/{ses}", {".mp4"}, {"task", "acq", "run"}),
    "boxes": (DERIV + "/boxes/{sub}/{ses}", {".tsv"}, {"task", "acq", "run"}),
    "labels": (DERIV + "/labels2d/{sub}/{ses}", {".tsv"}, {"task", "acq", "run"}),
    "keypoints": (DERIV + "/keypoints2d/*/{sub}/{ses}", {".tsv"}, {"task", "acq", "run"}),
    "tracks3d": (DERIV + "/tracks3d/{sub}/{ses}", {".tsv"}, {"task", "run"}),
    "camera": (DERIV + "/cameras", {".json"}, {"acq"}),
}
ENTITY_VALUES = {"ses": r"ses-[12]", "task": TASK, "tracksys": r"vicon|ipad", "run": r"0[1-9]|1[0-2]",
                 "acq": {"vicon": r"rec[0-9]+", "labels": r"top|side"}}

# ----------------------------------------------------------------------------- de-identification patterns
MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
         r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DAY, MON = r"(?:0?[1-9]|[12]\d|3[01])", r"(?:0?[1-9]|1[0-2])"
DATE = "|".join([
    rf"(?<![\d.])(?:19|20)\d\d[-/]{MON}[-/]{DAY}(?!\d)",                                          # 2001-02-03, 2001-2-3
    r"(?<![\d.])(?:19|20)\d\d[._](?:0[1-9]|1[0-2])[._](?:0[1-9]|[12]\d|3[01])(?!\d)",                    # 2001.02.03
    r"(?<![\d.])(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:[T_-]?[0-2]\d[0-5]\d(?:[0-5]\d)?)?(?!\d)",
    rf"(?<![\d./-]){DAY}[-/.]{DAY}[-/.](?:19|20)\d\d(?!\d)",                       # 03.02.2001, 03-02-2001, 02/03/2001
    rf"(?<![\d/.]){DAY}/{DAY}/\d\d(?![\d/])",                                                    # 2/3/01, 3/2/01
    rf"(?i:\b\d{{1,2}}[ -]{MONTH}\.?[ ,-]+(?:19|20)\d\d\b)",                                              # 3-Feb-2001
    rf"(?i:\b{MONTH}\.? \d{{1,2}}(?:st|nd|rd|th)?,? (?:19|20)\d\d\b)",                                    # Feb 3, 2001
])
CLOCK = (r"(?<![\d:.])(?:(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d(?:[:;]\d\d)?|(?:[01]\d|2[0-3]):[0-5]\d)"  # 12:34:56, 12:34,
         r"(?![\d:])")                                                                             # 12:34:56:12 (SMPTE)
ABSPATH = (r"(?:^|[^\w.])/{1,3}(?:home|Users)(?![\w.-])|/home/"           # /home/..., file:///home/..., /<mount>/home/
           r"|(?<![A-Za-z0-9])[A-Za-z]:\\{1,2}[A-Za-z0-9_]|(?:^|[\s\"'=(:])~/\w"           # C:\Users, ~/project
           r"|\bdata/(?:raw_video_data|game_data|vicon_data)\b")
EPOCH = r"(?<![\d.])1[4-9]\d{8}(?:\d{3}){0,3}(?:\.\d+)?(?!\d)"     # Unix time 2014-2033 in s, ms, us or ns
TRIAL_NAME = r"(?i:\bV[12][ _-]*(?:Trial|T)[ _-]*\d+\b)"                                            # V1 Trial 01, V1_T1
YEAR_HINT = re.compile(r"(?:19|20)\d\d[-/._]?\d|\d[-/.](?:19|20)\d\d(?!\d)")
SLASH_HINT = re.compile(r"\d/\d")
MONTHS3 = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
TORCH_STORAGE = re.compile(r"(?:^|/)data/\d+$")
_KINDS = _FFMPEG = None            # set per process by init_worker


def deid_kinds(terms):
    """[(kind, regex, hint)]: the regex runs only where the cheap hint(text, lowercase text) fires.
    Terms match case-insensitively as whole words; a word ends where letters meet digits (XXXXXXX103 matches and
    is reported with its digits)."""
    def bound(t):
        left = r"(?<![A-Za-z])" if t[0].isalpha() else r"(?<![0-9])"
        right = r"(?![A-Za-z])\d*" if t[-1].isalpha() else r"(?![0-9])"
        return left + re.escape(t) + right
    low_terms = [t.lower() for t in terms]
    ident = "(?i:" + "|".join(bound(t) for t in sorted(terms, key=len, reverse=True)) + ")"
    return [
        ("identifier", re.compile(ident), lambda s, low: any(t in low for t in low_terms)),
        ("trial name", re.compile(TRIAL_NAME), lambda s, low: "v1" in low or "v2" in low),
        ("date", re.compile(DATE), lambda s, low: bool(YEAR_HINT.search(s) or ("/" in s and SLASH_HINT.search(s))
                                                       or any(m in low for m in MONTHS3))),
        ("clock time", re.compile(CLOCK), lambda s, low: ":" in s),
        ("epoch time", re.compile(EPOCH), lambda s, low: True),
        ("path", re.compile(ABSPATH), lambda s, low: any(h in low for h in ("/home", "/users", ":\\", "~/", "data/"))),
    ]


def init_worker(terms, ffmpeg):
    global _KINDS, _FFMPEG
    _KINDS, _FFMPEG = deid_kinds(terms), ffmpeg
    warnings.simplefilter("ignore")


def deid_problems(text, where=""):
    """One problem line per kind of hit (first three hits quoted, with line numbers)."""
    low, out = text.lower(), []
    for kind, regex, hint in _KINDS:
        ms = list(regex.finditer(text)) if hint(text, low) else []
        if ms:
            shown = ", ".join(f"'{m.group()}'" + (f" (line {line_of(text, m.start())})" if "\n" in text else "")
                              for m in ms[:3])
            more = f" and {len(ms) - 3} more" if len(ms) > 3 else ""
            out.append(("deid", f"{kind} {shown}{more}{where}"))
    return out


# ----------------------------------------------------------------------------- per-file checks (worker processes)
NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
EMPTY_CELL = re.compile(r"(?m)(?:^|\t)[^\S\t\n]*(?=\t|$)")                # blank or whitespace-only cell
BAD_NA = re.compile(r"(?m)(?:^|\t)(?!n/a(?:\t|$))((?i:-?nan|na|n/a|none|null|[+-]?inf(?:inity)?))(?=\t|$)")


def ends(rel, endings):
    return any(rel.endswith(e) for e in endings)


def matches(rel, key):
    """Keys starting with '_' match file name endings, other keys whole relative paths."""
    return rel.endswith(key) if key.startswith("_") else rel == key


def rule_for(rel, table):
    return next((v for k, v in table.items() if matches(rel, k)), None)


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def check_file(args):
    """(rel, problems [(check, message)], facts) of one file; an exception in a check becomes a problem."""
    root, rel = args
    path = Path(root, rel)
    suffix = path.suffix.lower()
    out, facts = [], {"size": path.stat().st_size}
    try:
        if facts["size"] == 0:
            out.append(("format", "empty file"))
        elif suffix in TEXT_SUFFIXES:
            facts.update(check_text(rel, path.read_bytes(), out))
        elif suffix == ".c3d":
            facts.update(check_c3d(path, out))
        elif suffix == ".mp4":
            facts.update(check_mp4(path, out))
        elif suffix in ZIP_SUFFIXES and zipfile.is_zipfile(path):
            check_zip(path, out)
        else:
            check_binary(path, out)
    except Exception as e:  # noqa: BLE001 (report and go on with the other files)
        out.append(("format", checker_error(e)))
    return rel, out, facts


def checker_error(e):
    return f"checker error {type(e).__name__}: {e}"


def check_text(rel, raw, out):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        out.append(("format", f"not UTF-8 ({e})"))
        text = raw.decode("latin-1")
    if text.startswith("\ufeff"):
        out.append(("format", "UTF-8 byte-order mark"))
    if "\r" in text:
        out.append(("format", "CR line endings (use LF)"))
    if rel.endswith(".csv"):
        out.append(("format", "CSV file: tabular data must be TSV"))
    if rel.endswith(".tsv"):
        facts = check_tsv(rel, text, out)
        # a body of numbers and n/a holds no name, path, clock time or separated date; numeric_table checks
        # 8-digit dates and values >= 1e8 (epoch times)
        out.extend(deid_problems(text.split("\n", 1)[0] if facts.get("numeric") else text))
        return facts
    out.extend(deid_problems(text))
    if rel.endswith(".json"):
        return check_json(rel, text, out)
    if rel == "README.md" and not re.match(r"---\n.*?\n---\n", text, re.S):
        out.append(("format", "README.md does not start with a YAML header (--- ... ---)"))
    return {}


def check_tsv(rel, text, out):
    body = text[:-1] if text.endswith("\n") else text
    lines = body.split("\n")
    widths = [line.count("\t") + 1 for line in lines]
    if len(set(widths)) > 1:
        common_w = Counter(widths).most_common(1)[0][0]
        bad = [i + 1 for i, w in enumerate(widths) if w != common_w]
        out.append(("format", f"{len(bad)} rows do not have {common_w} columns (first: line {bad[0]})"))
    headerless = rel.endswith("_motion.tsv")
    data = body if headerless else body.partition("\n")[2]              # cheap hints before the regexes
    empty_hint = any(h in body for h in ("\t\t", "\n\t", "\t\n", "\n\n", " \t", "\t ", " \n", "\n ")) \
        or body[:1] in ("", "\t", "\n", " ") or body[-1:] in ("\t", " ")
    na_hint = data.lower().count("n") != data.count("n/a")      # every token BAD_NA looks for has an n or N
    for regex, what, hint in ((EMPTY_CELL, "empty cell", empty_hint),
                              (BAD_NA, "missing-value token other than n/a", na_hint)):
        hits = list(regex.finditer(body)) if hint else []
        if hits:
            out.append(("format", f"{len(hits)} x {what} (first: line {line_of(body, hits[0].start())}"
                                  f"{', ' + repr(hits[0].group(1)) if hits[0].groups() else ''})"))
    first = lines[0].split("\t")
    numeric_first = [bool(NUM.match(c)) or c == "n/a" for c in first]
    if headerless and not all(numeric_first):
        out.append(("format", "motion.tsv must not have a header row"))
    header = None if headerless else first
    if header is not None:
        if any(numeric_first):
            out.append(("format", "missing header row (numeric cell in line 1)"))
        dup = [c for c, n in Counter(header).items() if n > 1]
        if dup:
            out.append(("format", f"duplicate columns {dup}"))
        check_columns(rel, header, out)
    facts = {"header": header, "n_rows": len(lines) - (header is not None), "n_cols": widths[0]}
    if ends(rel, NUMERIC_TABLES):
        facts.update(numeric_table(rel, body, header is not None, out))
    elif header is not None:
        cells = [line.split("\t") for line in lines[1:]]
        for i, col in enumerate(header):
            if col in TIME_COLUMNS:
                check_times(col, pd.to_numeric(pd.Series([c[i] if i < len(c) else "" for c in cells]),
                                               errors="coerce").to_numpy(float), out)
        if any(matches(rel, k) for k in KEEP_TABLES) or rel.endswith("_events.tsv"):
            facts["rows"] = check_rows(rel, header, [dict(zip(header, c)) for c in cells], out)
    return facts


def check_rows(rel, header, rows, out):
    """Category values and the table's own rules; returns the rows (dicts)."""
    for col, pattern in (rule_for(rel, VOCAB) or {}).items():
        bad = [r[col] for r in rows if col in r and not re.fullmatch(pattern, r[col])]
        if bad:
            out.append(("format", f"{col}: {len(bad)} values outside {pattern} (e.g. {sorted(set(bad))[:3]})"))
    if rel.endswith("_events.tsv"):
        check_event_values(rows, out)
    elif rel.endswith("_beh.tsv"):
        check_beh_values(rows, out)
    elif rel == "participants.tsv":
        out.extend(("deid", f"column '{c}' looks like a date of birth / date") for c in header
                   if re.search(r"birth|dob|date", c, re.I))
    return rows


def check_times(col, values, out):
    """Times (s) count from a session or run origin: never epoch-sized, never beyond a 2-h visit."""
    a = np.abs(values[np.isfinite(values)])
    if (a >= EPOCH_S).any():
        out.append(("deid", f"{col}: {np.sum(a >= EPOCH_S)} values >= {EPOCH_S:.0e} s (max {a.max():.4f}): "
                            "absolute (epoch) time, not session or run time"))
    elif (a > MAX_VISIT_S).any():
        out.append(("format", f"{col}: {np.sum(a > MAX_VISIT_S)} values beyond a {MAX_VISIT_S} s visit "
                              f"(max {a.max():.4f}): wrong time origin or unit"))


def check_event_values(rows, out):
    """trial_type in EVENT_TYPES, onset numeric, duration numeric or n/a, one movement_start at onset 0 (the
    run-time origin), x and y on the screen."""
    types = {r.get("trial_type") for r in rows} - EVENT_TYPES - {None}
    if types:
        out.append(("format", f"unknown trial_type values {sorted(types)}"))
    bad = [r.get("onset") for r in rows if not NUM.match(r.get("onset", ""))]
    bad += [r.get("duration") for r in rows if not (NUM.match(r.get("duration", "")) or r.get("duration") == "n/a")]
    if bad:
        out.append(("format", f"non-numeric onset / duration {bad[:3]}"))
    starts = [as_float(r.get("onset")) for r in rows if r.get("trial_type") == "movement_start"]
    if starts != [0.0]:
        out.append(("format", f"movement_start onsets {starts}: expected one, at 0 (the run-time origin)"))
    check_screen([as_float(r.get("x")) for r in rows], [as_float(r.get("y")) for r in rows], out)


def check_beh_values(rows, out):
    """Session time starts at the first trial start (0); time_to_movement and movement_time agree with it."""
    def col(name):
        return np.array([as_float(r.get(name)) for r in rows], float)
    start = col("trial_start")
    if np.isfinite(start).any() and abs(np.nanmin(start)) > 1e-3:
        out.append(("format", f"earliest trial_start {np.nanmin(start):.4f} s: session time starts at the first "
                              "trial start (0)"))
    for diff, end, begin in (("time_to_movement", "movement_start", "trial_start"),
                             ("movement_time", "movement_end", "movement_start")):
        d = np.abs(col(end) - col(begin) - col(diff))
        if (d[np.isfinite(d)] > 2e-3).any():
            out.append(("format", f"{diff} != {end} - {begin} in {np.sum(d > 2e-3)} rows (max {np.nanmax(d):.4f} s)"))


def check_screen(x, y, out):
    """iPad positions (mm) lie on the screen: |x|, |y| at most half its size (game points are 5.2 times larger)."""
    for name, v, half in (("x", x, SCREEN_HALF_MM[0]), ("y", y, SCREEN_HALF_MM[1])):
        a = np.abs(np.asarray(v, float))
        n = np.sum(a > half)               # n/a (nan) never counts
        if n:
            out.append(("format", f"{name}: {n} values beyond the screen (|{name}| <= {half:.2f} mm, max "
                                  f"{np.nanmax(a):.2f}): game points not converted to mm?"))


def check_columns(rel, header, out):
    required = rule_for(rel, REQUIRED_COLUMNS)
    if required is None:
        return
    missing = [c for c in required if c not in header]
    if missing:
        out.append(("format", f"missing columns {missing}"))
    n = rule_for(rel, BIDS_ORDER)
    if n and header[:n] != required[:n]:
        out.append(("format", f"first columns must be {required[:n]}, found {header[:n]}"))


def numeric_table(rel, body, has_header, out):
    """motion, labels and keypoints tables: numbers only; time column origin and range, units of iPad and VICON."""
    try:
        a = pd.read_csv(io.StringIO(body), sep="\t", header=None, skiprows=int(has_header), na_values=["n/a"],
                        keep_default_na=False, dtype=float).to_numpy()
    except ValueError as e:
        out.append(("format", f"non-numeric cell ({str(e).splitlines()[0]})"))
        return {}
    facts = {"numeric": True}
    big = np.abs(a[np.isfinite(a)])
    out.extend(deid_problems(" ".join(map(str, np.unique(big[(big >= 1e7) & (big < 1e8)].astype(np.int64)))),
                             " (8-digit number in the table)"))
    if not a.size:
        return facts
    time_col = 0 if "_tracksys-ipad_" in rel else 1 if ends(rel, ("_labels.tsv", "_keypoints.tsv", "_boxes.tsv")) else -1
    if 0 <= time_col < a.shape[1]:
        check_times("latency" if time_col == 0 else "time", a[:, time_col], out)
    rest = np.abs(a[:, np.arange(a.shape[1]) != time_col])
    if (rest >= EPOCH_S).any():
        out.append(("deid", f"{np.sum(rest >= EPOCH_S)} values >= {EPOCH_S:.0e} (max {np.nanmax(rest):.4f}): "
                            "absolute (epoch) time?"))
    if "_tracksys-ipad_" in rel:
        facts["latency"] = check_ipad(a, out)
        facts["latency_back"] = int((np.diff(a[:, 0]) <= 0).sum())          # steps back or repeats
    elif "_tracksys-vicon_" in rel:
        check_vicon_units(a, out)
    else:
        facts["first_frame"] = a[0, 0]
    return facts


def check_ipad(a, out):
    """latency increasing from 0 (sample 0 is the movement start), x and y on the screen; returns
    (first, last, median step) of the latency."""
    lat = a[:, 0]
    if np.isnan(lat).any():
        out.append(("format", "latency column has n/a values"))
    if not abs(lat[0]) <= 1e-3:
        out.append(("format", f"first latency {lat[0]:.4f} s: sample 0 is the movement start (run time 0)"))
    if a.shape[1] >= 3:
        check_screen(a[:, 1], a[:, 2], out)
    return float(np.nanmin(lat)), float(np.nanmax(lat)), float(np.nanmedian(np.diff(lat))) if len(lat) > 1 else None


def check_vicon_units(a, out):
    """Marker coordinates in mm: the 99th percentile of |value| of a reaching trial is a few hundred mm."""
    v = np.abs(a[np.isfinite(a)])
    lo, hi = VICON_P99_MM
    if v.size and not lo <= np.quantile(v, 0.99) <= hi:
        out.append(("format", f"99th percentile of |position| {np.quantile(v, 0.99):.2f}, expected {lo}-{hi} mm: "
                              "positions not in mm?"))


def _reject_constant(name):
    raise ValueError(f"{name} is not valid JSON")


def check_json(rel, text, out):
    try:
        obj = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:
        out.append(("format", f"invalid JSON: {e}"))
        return {}
    if not isinstance(obj, dict):
        out.append(("format", f"top level must be a JSON object, found {type(obj).__name__}"))
        return {}
    name = Path(rel).name
    if name == "dataset_description.json":
        dtype = "derivative" if rel.startswith("derivatives/") else "raw"
        for key in ("Name", "BIDSVersion"):
            if key not in obj:
                out.append(("format", f"missing {key}"))
        if obj.get("DatasetType") != dtype:
            out.append(("format", f"DatasetType must be '{dtype}'"))
        if dtype == "derivative" and not obj.get("GeneratedBy"):
            out.append(("format", "derivative dataset_description.json needs GeneratedBy"))
    if name.endswith("_motion.json"):
        tracksys = re.search(r"_tracksys-([a-z]+)_", name)
        keys = MOTION_JSON_KEYS.get(tracksys.group(1) if tracksys else "", [])
        missing = [k for k in keys if k not in obj]
        if missing:
            out.append(("format", f"missing keys {missing}"))
        if re.sub(r"[^0-9A-Za-z]", "", str(obj.get("TaskName", ""))).lower() != TASK:
            out.append(("format", f"TaskName '{obj.get('TaskName')}' does not match task-{TASK}"))
    return {"json": obj} if len(text) < 500_000 else {}


def check_c3d(path, out):
    with open(path, "rb") as fh:
        try:
            r = c3d.Reader(fh)
        except Exception as e:  # noqa: BLE001 (any parse failure is a format problem)
            out.append(("format", f"unreadable C3D ({e})"))
            return {}
        fields = [("header events", " ".join(map(str, r.header.event_labels)))]
        for _, group in r.group_listed():
            fields.append((f"group {group.name}", f"{group.name} {_str(group.desc)}"))
            fields += [(f"{group.name}:{name}", f"{name} {_str(p.desc)} {_param_text(p)}") for name, p in group.param_items()]
        for label, text in fields:
            out.extend(deid_problems(text, f" in C3D parameter {label}"))
        labels = [s.strip() for s in r.point_labels] if r.get("POINT:LABELS") is not None else []
        frames = _c3d_field(r, "ACTUAL_END_FIELD", r.header.last_frame) - \
            _c3d_field(r, "ACTUAL_START_FIELD", r.header.first_frame) + 1
        rate = float(r.point_rate)
        data_end = c3d_data_end(r, frames)
    size = path.stat().st_size
    if size < data_end:
        out.append(("format", f"truncated: {size} bytes, the header and parameters describe {frames} frames "
                              f"ending at byte {data_end}"))
    if abs(rate - common.VICON_HZ) > 1e-6:
        out.append(("format", f"point rate {rate} Hz, expected {common.VICON_HZ}"))
    missing = [m for m in common.MARKERS if m not in labels]
    if missing:
        out.append(("format", f"markers missing from POINT:LABELS {missing}"))
    return {"frames": frames, "rate": rate}


def _str(v):
    return v.decode("latin-1") if isinstance(v, bytes) else (v or "")


def _param_text(p):
    """String parameters as text (string arrays joined by spaces); numeric parameters as ''."""
    if p.bytes_per_element != -1:
        return ""
    return " ".join(map(str, np.ravel(p.string_array))) if len(p.dimensions) > 1 else p.string_value


def c3d_data_end(r, frames):
    """Byte offset where the data section ends: it starts at block POINT:DATA_START (header copy) and holds per
    frame 4 words (x, y, z, residual) per point plus the analog samples; words are 4-byte floats when
    POINT:SCALE < 0, else 2-byte integers. Python ints: the reader returns uint16, which overflows with numpy 2."""
    word = 4 if r.point_scale < 0 else 2
    per_frame = 4 * int(r.point_used) + int(r.analog_used) * int(r.analog_per_frame)
    return (int(r.header.data_block) - 1) * 512 + frames * per_frame * word


def _c3d_field(reader, name, header_value):
    """TRIAL:<name> as two 16-bit words (the c3d 0.5.2 reader overflows on this with numpy 2)."""
    p = reader.get(f"TRIAL:{name}")
    if p is None:
        return int(header_value)
    w = p.uint16_array
    return int(w[0]) + 65536 * int(w[1])


def check_mp4(path, out):
    """Metadata tags and streams from the input section of ffmpeg's stderr; the exact frame count from the
    framecrc lines of the video stream (one per packet = frame; stream copy, no decoding)."""
    run = subprocess.run([_FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                          "-f", "framecrc", "-"], capture_output=True, text=True, errors="replace")
    meta, streams, meta_indent = [], [], None
    for line in run.stderr.split("\nStream mapping:")[0].splitlines():      # the output section repeats the tags
        indent, s = len(line) - len(line.lstrip()), line.strip()
        if meta_indent is not None and indent > meta_indent:
            meta.append(s)
            continue
        meta_indent = indent if s == "Metadata:" else None
        if s.startswith("Stream #"):
            streams.append(s)
    for tag in meta:
        out.extend(deid_problems(tag, f" in MP4 metadata tag '{tag.split(':')[0].strip()}'"))
    extra = sorted({t.split(":")[0].strip() for t in meta} - MP4_TAGS)
    if extra:
        out.append(("deid", f"MP4 metadata not stripped: tags {extra}"))
    video = [s for s in streams if "Video:" in s]
    if len(video) != 1 or len(streams) != 1:
        out.append(("format", f"expected one video stream and nothing else, found {len(streams)} streams "
                              f"({len(video)} video)"))
    if not video:
        return {}
    if not re.search(r"Video: h264\b", video[0]) or "yuv420p" not in video[0]:
        out.append(("format", "video stream is not H.264 yuv420p"))
    fps = re.search(r"([\d.]+) fps", video[0])
    facts = {"fps": float(fps[1]) if fps else None}
    if run.returncode:
        out.append(("format", f"ffmpeg cannot read the video stream (exit {run.returncode})"))
    else:
        facts["frames"] = sum(1 for line in run.stdout.splitlines() if line and not line.startswith("#"))
    return facts


def check_zip(path, out):
    with zipfile.ZipFile(path) as z:
        out.extend(deid_problems("\n".join(z.namelist()), " in archive member names"))
        for info in z.infolist():
            if not TORCH_STORAGE.search(info.filename):
                out.extend(deid_problems(z.read(info).decode("latin-1"), f" in archive member {info.filename}"))


def check_binary(path, out, chunk=1 << 24, overlap=256):
    with open(path, "rb") as fh:
        tail = b""
        while True:
            block = fh.read(chunk)
            if not block:
                break
            out.extend(deid_problems((tail + block).decode("latin-1"), " in binary content"))
            tail = block[-overlap:]


# ----------------------------------------------------------------------------- report
class Report:
    def __init__(self, partial):
        self.partial, self.counts = partial, Counter()

    def problem(self, check, rel, msg):
        self.counts[check] += 1
        print(f"PROBLEM {check:<12} {rel}: {msg}")

    @staticmethod
    def info(msg):
        print(f"INFO    {msg}")

    def lenient(self, check, rel, msg):
        """A problem for a full release; INFO for a --partial test tree."""
        if self.partial:
            self.info(f"{rel}: {msg} (partial tree)")
        else:
            self.problem(check, rel, msg)

    def expect(self, check, rel, what, found, expected):
        if found == expected:
            self.info(f"{what}: {found} (paper {expected})")
        else:
            self.lenient(check, rel, f"{what}: {found}, paper {expected}")


def guarded(report, rel, check, *args):
    """Run one cross-file check; an exception in the checker becomes a problem of rel and the other checks go on."""
    try:
        check(*args)
    except Exception as e:  # noqa: BLE001
        report.problem("format", rel, checker_error(e))


# ----------------------------------------------------------------------------- helpers for cross-file checks
def part_of(rel):
    if rel.startswith(f"{common.VIDEOS}/"):
        return "videos"
    if rel.startswith("sourcedata/"):
        return "sourcedata"
    return "derivatives" if rel.startswith("derivatives/") else "core"


def list_files(root, parts, report):
    """Sorted relative paths of the regular files and of the directories of the selected parts; symbolic links
    are reported."""
    files, dirs = [], []
    for d, subdirs, names in os.walk(root):
        for n in subdirs + names:
            p = Path(d, n)
            rel = p.relative_to(root).as_posix()
            is_dir = n in subdirs
            if part_of(rel + "/" if is_dir else rel) not in parts:
                continue
            if p.is_symlink():
                report.problem("format", rel, "symbolic link")
            else:
                (dirs if is_dir else files).append(rel)
    return sorted(files), sorted(dirs)


def norm(value, key):
    """'1' / 'ses-1' -> 'ses-1', '3' / 'run-03' -> 'run-03' (None if not an index)."""
    v = str(value).strip()
    v = v[len(key) + 1:] if v.startswith(f"{key}-") else v
    if not v.isdigit():
        return None
    return f"{key}-{int(v)}" if key == "ses" else f"{key}-{int(v):02d}"


def as_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_rows(path):
    lines = path.read_text(encoding="utf-8", errors="replace").rstrip("\n").split("\n")
    header = lines[0].split("\t")
    return header, [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def load_terms(path):
    """Scrub terms (comment lines skipped) plus the game user ids of the study participants."""
    terms = [t.strip() for t in path.read_text().splitlines() if t.strip() and not t.lstrip().startswith("#")]
    id_map = common.PRIVATE / "id_map.tsv"
    if not id_map.exists():
        return terms, 0
    import scipy.io as sio
    users = set()
    for row in read_rows(id_map)[1]:
        mat = common.GAME / f"{row['study_id']}.mat"
        if mat.exists():
            s = sio.loadmat(mat, variable_names=["Reaching"], squeeze_me=True, struct_as_record=False)
            users |= {str(u) for u in np.atleast_1d(s["Reaching"].user_id) if str(u).strip()}
    return terms + sorted(users), len(users)


def bids_validator_status(report):
    tools = [t for t in ("deno", "node", "bids-validator") if shutil.which(t)]
    pkgs = [p for p in ("bids_validator", "bids") if importlib.util.find_spec(p)]
    if tools or pkgs:
        report.info(f"BIDS validator available ({', '.join(tools + pkgs)}): run it for the official check")
    else:
        report.info("BIDS validator not run: deno and node are not installed, no bids-validator executable, and the "
                    "Python packages bids-validator (bids_validator) and pybids (bids) are absent; names, entities "
                    "and sidecars are checked by this script's Motion-BIDS rules only")


# ----------------------------------------------------------------------------- name checks
def check_names(files, dirs, report):
    for rel in dirs:                                       # also catches empty directories
        for check, msg in deid_problems(rel, " in the directory name"):
            report.problem(check, rel + "/", msg)
    for rel in files:
        for check, msg in deid_problems(rel, " in the file name"):
            report.problem(check, rel, msg)
        parts = rel.split("/")
        if len(parts) == 1 and rel not in ROOT_FILES:
            report.problem("format", rel, "unexpected file at the dataset root")
        elif parts[0] not in ROOT_DIRS and not parts[0].startswith("sub-") and len(parts) > 1:
            report.problem("format", rel, "unexpected directory at the dataset root")
        if any(p.startswith("sub-") for p in parts) or rel.startswith(f"{DERIV}/cameras/"):
            for msg in bids_name_problems(rel):
                report.problem("format", rel, msg)


def bids_name_problems(rel):
    d, name = rel.rsplit("/", 1) if "/" in rel else ("", rel)
    m = NAME_RE.match(name)
    if not m:
        return ["name does not follow the Motion-BIDS entity order sub, ses, task, tracksys, acq, run"]
    e = {k: v for k, v in m.groupdict().items() if v is not None}
    rule = SUFFIX_RULES.get(e["suffix"])
    if rule is None:
        return [f"unknown suffix '{e['suffix']}'"]
    pattern, exts, entities = rule
    out = []
    if e["ext"] not in exts:
        out.append(f"extension {e['ext']} not allowed for _{e['suffix']}")
    present = set(e) - {"sub", "suffix", "ext"}
    if present != entities | {"ses"}:
        out.append(f"entities {sorted(present)}, expected {sorted(entities | {'ses'})} besides sub")
    folder = pattern.format(sub=e["sub"], ses=e.get("ses", "ses-?"))
    if not fnmatch(d, folder):
        out.append(f"belongs in {folder}/")
    if not (re.fullmatch(r"sub-\d\d", e["sub"]) and 1 <= int(e["sub"][4:]) <= PAPER["participants"]):
        out.append(f"participant label {e['sub']} outside sub-01 ... sub-{PAPER['participants']}")
    for key in ("ses", "task", "tracksys", "run", "acq"):
        allowed = ENTITY_VALUES.get(key)
        if key == "acq":
            allowed = ENTITY_VALUES["acq"].get(e["suffix"], "|".join(common.VIEWS))
        if key in e and not re.fullmatch(allowed, e[key]):
            out.append(f"{key} '{e[key]}' not in {allowed}")
    return out


# ----------------------------------------------------------------------------- cross-file checks
class Tree:
    """Files, per-file facts and the manifest runs of the checked tree."""

    def __init__(self, root, parts, files, facts, report):
        self.root, self.parts, self.files, self.facts, self.report = root, parts, set(files), facts, report
        # participants with a sub-NN/ directory (flat files such as cameras/sub-NN_* do not count)
        self.subjects = {m[1] for f in files for m in re.finditer(r"(?:^|/)(sub-\d+)/", f)}
        path = root / DERIV / "manifest.tsv"
        self.manifest = read_rows(path)[1] if path.exists() else None
        self.runs = self._runs()

    def _runs(self):
        """{(sub, ses, run): manifest row (or {} from beh.tsv)}, limited to the participants present if partial."""
        runs = {}
        if self.manifest is not None:
            for row in self.manifest:
                key = (norm(row.get("participant_id"), "sub"), norm(row.get("session"), "ses"), norm(row.get("run"), "run"))
                if key in runs:
                    self.report.problem("format", f"{DERIV}/manifest.tsv", f"duplicate run {key}")
                runs[key] = row
        else:
            self.report.lenient("completeness", f"{DERIV}/manifest.tsv",
                                "missing; runs taken from beh.tsv, VICON motion and videos not checked for completeness")
            for rel, f in self.facts.items():
                if rel.endswith("_beh.tsv"):
                    sub, ses = rel.split("/")[:2]
                    runs.update({(sub, ses, norm(r.get("run"), "run")): {} for r in f.get("rows", [])})
        if self.report.partial:
            runs = {k: v for k, v in runs.items() if k[0] in self.subjects}
        return runs

    def has(self, rel):
        return rel in self.files

    def need(self, rel, why):
        if not self.has(rel):
            self.report.problem("completeness", rel, f"missing ({why})")


def has_vicon(row):
    cov = as_float(row.get("vicon_coverage"))
    return cov is not None and not np.isnan(cov) and cov > 0


def clip_exists(row, view):
    n = as_float(row.get(f"{view}_n_frames"))
    return n is not None and not np.isnan(n) and n > 0


def run_path(sub, ses, datatype, **kw):
    return f"{sub}/{ses}/{datatype}/{common.bids_name(sub, ses, **kw)}"


def video_path(sub, ses, run, view):
    return f"{common.VIDEOS}/{sub}/{ses}/" + common.bids_name(sub, ses, "video", ".mp4", acq=view, run=run)


def camera_path(sub, ses, view):
    return f"{DERIV}/cameras/{sub}_{ses}_acq-{view}_camera.json"


def camera_ok(t, sub, ses, view, run):
    """The run's camera of this view has CalibrationOK (cameras/*.json lists the runs of each camera)."""
    j = t.facts.get(camera_path(sub, ses, view), {}).get("json") or {}
    return any(e.get("CalibrationOK") and run in e.get("Runs", []) for e in j.get("Cameras", []))


def check_core(t):
    for rel in ("dataset_description.json", "participants.tsv", "participants.json", f"task-{TASK}_events.json",
                f"task-{TASK}_beh.json", "phenotype/anthropometrics.tsv", "phenotype/anthropometrics.json"):
        t.need(rel, "release layout")
    if not t.has("README.md"):
        t.report.lenient("completeness", "README.md", "missing (dataset card)")
    check_run_files(t)
    for rel in sorted(f for f in t.files if f.endswith("_motion.tsv")):
        guarded(t.report, rel, check_motion_set, t, rel)
    check_sidecars(t)
    check_participants(t)


def check_run_files(t):
    """events, iPad motion, one beh row and (with VICON data) VICON motion of every run; no files of other runs."""
    r = t.report
    for sub, ses in sorted({(s, e) for s, e, _ in t.runs}):
        beh = run_path(sub, ses, "beh", suffix="beh", ext=".tsv")
        t.need(beh, "every session has a beh.tsv")
        if not t.has(beh):
            continue
        listed = [norm(x.get("run"), "run") for x in t.facts.get(beh, {}).get("rows", [])]
        expected = {run for s, e, run in t.runs if (s, e) == (sub, ses)}
        for what, runs in (("no row for", expected - set(listed)),
                           ("more than one row for", {k for k, n in Counter(listed).items() if n > 1}),
                           ("rows for runs not in manifest.tsv:", set(listed) - expected)):
            if runs:
                r.problem("completeness", beh, f"{what} {sorted(map(str, runs))}")
    for (sub, ses, run), row in sorted(t.runs.items()):
        t.need(run_path(sub, ses, "motion", suffix="events", ext=".tsv", run=run), "every run has events")
        for sfx, ext in (("motion", ".tsv"), ("channels", ".tsv"), ("motion", ".json")):
            t.need(run_path(sub, ses, "motion", suffix=sfx, ext=ext, tracksys="ipad", run=run), "every run has iPad motion")
            vicon = run_path(sub, ses, "motion", suffix=sfx, ext=ext, tracksys="vicon", run=run)
            if row and has_vicon(row):
                t.need(vicon, "the run has VICON data")
            elif row and t.has(vicon):
                r.problem("completeness", vicon, "VICON motion for a run without VICON data (vicon_coverage 0)")
    for rel in sorted(t.files):
        m = re.match(r"(sub-\d+)/(ses-\d+)/.*_(run-\d+)_", rel)
        if m and m.groups() not in t.runs and (not r.partial or m[1] in t.subjects):
            r.problem("completeness", rel, "run not in manifest.tsv")


def check_participants(t):
    r = t.report
    participants = t.facts.get("participants.tsv", {}).get("rows")
    if participants is None:
        return
    groups = Counter(p.get("group") for p in participants)
    r.expect("numbers", "participants.tsv", "participants", len(participants), PAPER["participants"])
    for g in ("TD", "UCP"):
        r.expect("numbers", "participants.tsv", f"{g} participants", groups.get(g, 0), PAPER[g])
    if t.manifest is not None:
        ids, mids = {p.get("participant_id") for p in participants}, {m.get("participant_id") for m in t.manifest}
        if ids != mids and not r.partial:
            r.problem("completeness", "participants.tsv", f"participants differ from manifest.tsv: {sorted(ids ^ mids)}")


def check_motion_set(t, rel):
    """motion.tsv against its channels.tsv and motion.json (and the C3D it was cut from)."""
    r, stem = t.report, rel[:-len("_motion.tsv")]
    mot, chan = t.facts.get(rel, {}), t.facts.get(stem + "_channels.tsv", {}).get("rows")
    meta = t.facts.get(stem + "_motion.json", {}).get("json")
    if chan is None or meta is None:
        r.problem("completeness", rel, "channels.tsv or motion.json missing or unreadable")
        return
    if not re.fullmatch(r"[ALPRSI_]{3}", str(meta.get("SpatialAxes", ""))):
        r.problem("format", stem + "_motion.json", f"SpatialAxes {meta.get('SpatialAxes')!r}: three of A L P R S I _")
    if "n_rows" not in mot:
        return                                  # empty or unreadable motion.tsv: reported by its file check
    n_rows, n_cols = mot["n_rows"], mot["n_cols"]
    if n_cols != len(chan):
        r.problem("format", rel, f"{n_cols} columns but {len(chan)} rows in channels.tsv")
    if meta.get("MotionChannelCount") != n_cols:
        r.problem("format", stem + "_motion.json", f"MotionChannelCount {meta.get('MotionChannelCount')} != {n_cols} columns")
    names = [c.get("name") for c in chan]
    if "_tracksys-vicon_" in rel:
        if names != VICON_CHANNELS:
            r.problem("format", stem + "_channels.tsv", "channel names are not common.MARKERS x (x, y, z)")
        n_pos = sum(c.get("type") == "POS" for c in chan)
        if meta.get("POSChannelCount") != n_pos:
            r.problem("format", stem + "_motion.json", f"POSChannelCount {meta.get('POSChannelCount')} != {n_pos}")
        check_regular_rate(r, stem, meta, n_rows)
        check_recording(t, stem, meta, n_rows)
    else:
        found = [(c.get("name"), c.get("type"), c.get("units")) for c in chan]
        if found != IPAD_CHANNELS:
            r.problem("format", stem + "_channels.tsv", f"channels {found}, expected {IPAD_CHANNELS}")
        check_irregular_rate(r, stem, meta, mot.get("latency"))
        if mot.get("latency_back") and "LatencyNote" not in meta:
            r.problem("format", rel, f"latency steps back or repeats {mot['latency_back']} times and the "
                                     "motion.json has no LatencyNote")


def check_regular_rate(r, stem, meta, n_rows):
    sf, dur = meta.get("SamplingFrequency"), meta.get("RecordingDuration")
    if sf != common.VICON_HZ:
        r.problem("format", stem + "_motion.json", f"SamplingFrequency {sf}, expected {common.VICON_HZ}")
    elif not isinstance(dur, (int, float)) or abs(dur * sf - n_rows) > 1.01:
        r.problem("format", stem + "_motion.json", f"RecordingDuration {dur} s x {sf} Hz does not match {n_rows} rows")


def check_irregular_rate(r, stem, meta, latency):
    sf, sfe, dur = meta.get("SamplingFrequency"), meta.get("SamplingFrequencyEffective"), meta.get("RecordingDuration")
    rel = stem + "_motion.json"
    if sf != "n/a":
        r.problem("format", rel, f"SamplingFrequency {sf!r}: irregular sampling must be \"n/a\"")
    if latency is None or latency[2] is None:
        return
    lo, hi, dt = latency
    if not isinstance(sfe, (int, float)) or dt <= 0 or abs(sfe * dt - 1) > 0.1:
        r.problem("format", rel, f"SamplingFrequencyEffective {sfe} Hz vs median latency step {dt:.4f} s")
    if not isinstance(dur, (int, float)) or dur < hi - lo - 1e-3:
        r.problem("format", rel, f"RecordingDuration {dur} s shorter than the latency span {hi - lo:.4f} s")


def check_recording(t, stem, meta, n_rows):
    """The motion window is the one recordings.tsv lists for the run and lies in its C3D recording; samples outside
    the recording (partial VICON coverage) must be the share 1 - vicon_coverage of the manifest."""
    m = re.search(r"_(run-\d+)$", stem)
    if "sourcedata" not in t.parts or m is None:        # a name without run is reported by check_names
        return
    (sub, ses), run = stem.split("/")[:2], m[1]
    rel, c3d_rel = stem + "_motion.json", f"sourcedata/vicon/{sub}/{ses}/{meta.get('Recording')}"
    if not t.has(c3d_rel):
        t.report.problem("completeness", rel, f"Recording {meta.get('Recording')!r} not in sourcedata")
        return
    frames, start = t.facts[c3d_rel].get("frames"), meta.get("StartFrameInRecording")
    if not isinstance(start, int) or frames is None or not n_rows:
        t.report.problem("format", rel, f"StartFrameInRecording {start!r} / C3D frames {frames} / {n_rows} rows")
        return
    listed = recording_window(t, c3d_rel, run)
    if listed != (start, n_rows):
        t.report.problem("format", rel, f"StartFrameInRecording {start} and {n_rows} samples, but recordings.tsv lists "
                                        f"(start, frames) {listed} for {run} of {meta.get('Recording')}")
    outside = (max(0, -start) + max(0, start + n_rows - frames)) / n_rows
    coverage = as_float(t.runs.get((sub, ses, run), {}).get("vicon_coverage"))
    if outside and (coverage is None or abs(outside - (1 - coverage)) > 0.01 + 1 / n_rows):
        t.report.problem("format", rel, f"samples {start}..{start + n_rows - 1} reach outside the {frames} frames of "
                                        f"the C3D ({outside:.1%}) but vicon_coverage is {coverage}")


def recording_window(t, c3d_rel, run):
    """(run_start_frames, run_n_frames) entries of recordings.tsv for a run of a recording, or None."""
    for row in t.facts.get(RECORDINGS, {}).get("rows", []):
        runs = [norm(x, "run") for x in row.get("runs", "").split(",")]
        if row.get("file") == c3d_rel and run in runs:
            return tuple(nth_int(row.get(c, ""), runs.index(run)) for c in ("run_start_frames", "run_n_frames"))
    return None


def nth_int(cell, i):
    """Integer i of a comma-separated cell, else None (n/a or missing)."""
    values = cell.split(",")
    return int(values[i]) if i < len(values) and re.fullmatch(r"-?\d+", values[i]) else None


def check_sidecars(t):
    """Every column of a TSV is described by its JSON sidecar (inherited at the root for events and beh)."""
    pairs = {"participants.tsv": "participants.json", "phenotype/anthropometrics.tsv": "phenotype/anthropometrics.json"}
    pairs.update({rel: f"task-{TASK}_{kind}.json" for rel in t.files for kind in ("events", "beh")
                  if rel.endswith(f"_{kind}.tsv") and rel.startswith("sub-")})
    missing = defaultdict(set)
    for tsv, side in pairs.items():
        header = t.facts.get(tsv, {}).get("header")
        if header is not None:
            missing[side] |= {c for c in header if c not in ("participant_id", "onset", "duration")}
    for side, cols in sorted(missing.items()):
        desc = t.facts.get(side, {}).get("json", {})
        if t.has(side) and cols - set(desc):
            t.report.problem("format", side, f"columns not described: {sorted(cols - set(desc))}")


def check_sourcedata(t):
    """One end-effector trajectory file per session; recordings.tsv names existing C3D files with their frame counts."""
    for sub, ses in sorted({(s, e) for s, e, _ in t.runs}):
        t.need(f"sourcedata/trajectory/{sub}/{ses}/" + common.bids_name(sub, ses, "trajectory", ".json"),
               "one trajectory file per session")
    t.need(RECORDINGS, "recording -> runs table")
    check_references(t, RECORDINGS)
    for row in t.facts.get(RECORDINGS, {}).get("rows", []):
        frames = t.facts.get(row.get("file"), {}).get("frames")
        if frames is not None and as_float(row.get("n_frames")) != frames:
            t.report.problem("format", RECORDINGS, f"{row.get('file')}: n_frames {row.get('n_frames')}, the C3D has "
                                                   f"{frames}")
    n_c3d = sum(f.endswith(".c3d") for f in t.files)
    t.report.info(f"sourcedata: {n_c3d} C3D recordings, "
                  f"{sum(f.endswith('_trajectory.json') for f in t.files)} trajectory files")


def check_references(t, table_rel):
    """Cells naming files must name existing files: relative paths (to the root, the table or derivatives/vikitraj)
    or bare BIDS file names (looked up by name)."""
    path = t.root / table_rel
    if not path.exists():
        return
    base, names = str(Path(table_rel).parent), {Path(f).name for f in t.files}
    for row in read_rows(path)[1]:
        for col, v in row.items():
            if not re.fullmatch(r"[\w.\-/]+\.(?:tsv|json|mp4|c3d)", v or ""):
                continue
            if v in names or any((t.root / c).exists() for c in (v, f"{base}/{v}", f"{DERIV}/{v}")):
                continue
            sub = re.search(r"sub-\d+", v)
            if reference_part(v) in t.parts and (not t.report.partial or (sub and sub[0] in t.subjects)):
                t.report.problem("completeness", table_rel, f"{col} names a missing file: {v}")


def reference_part(v):
    """Release part a referenced file belongs to (bare names: from the location rule of their suffix)."""
    m = NAME_RE.match(v)
    if "/" not in v and m and m["suffix"] in SUFFIX_RULES:
        return part_of(SUFFIX_RULES[m["suffix"]][0])
    in_deriv = v.startswith(("videos/", "labels2d/", "keypoints2d/", "tracks3d/", "cameras/"))
    return part_of(f"{DERIV}/{v}" if in_deriv else v)


def check_derivatives(t):
    r = t.report
    if t.manifest is None:
        return
    rel = f"{DERIV}/manifest.tsv"
    check_references(t, rel)
    t.need(f"{DERIV}/.bidsignore", "the custom derivative layout")
    desc = t.facts.get(f"{DERIV}/manifest.json", {}).get("json")
    if desc is None:
        t.need(f"{DERIV}/manifest.json", "manifest sidecar")
    else:
        undescribed = [c for c in t.facts.get(rel, {}).get("header") or [] if c not in desc]
        if undescribed:
            r.problem("format", f"{DERIV}/manifest.json", f"columns not described: {undescribed}")
    for (sub, ses, run), row in sorted(t.runs.items()):
        for view in common.VIEWS:
            video = row.get(f"{view}_video", "n/a")
            if clip_exists(row, view) and video in ("n/a", ""):
                r.problem("completeness", rel, f"{sub} {ses} {run}: {view}_video is n/a although the clip exists")
            if not clip_exists(row, view) and video not in ("n/a", "", None):
                r.problem("completeness", rel, f"{sub} {ses} {run}: {view}_video set but no clip ({view}_n_frames n/a)")
    check_manifest_flags(t)
    rows = list(t.manifest)
    r.expect("numbers", rel, "trials", len(rows), PAPER["trials"])
    groups = {g: len({x["participant_id"] for x in rows if x.get("group") == g}) for g in ("TD", "UCP")}
    for g, n in groups.items():
        r.expect("numbers", rel, f"{g} participants in manifest", n, PAPER[g])
    for col in ("vicon_valid", "top_valid", "side_valid"):
        r.expect("numbers", rel, col, sum(as_bool(x.get(col)) for x in rows), PAPER[col])
    all3 = sum(all(as_bool(x.get(c)) for c in ("vicon_valid", "top_valid", "side_valid")) for x in rows)
    r.expect("numbers", rel, "vicon+top+side valid", all3, PAPER["vicon+top+side valid"])
    r.info("not in the paper table: " + ", ".join(f"{c} {sum(as_bool(x.get(c)) for x in rows)}"
                                                  for c in ("benchmark_valid", "trajectory_qa_failed")))
    presence_summary(t)


def check_manifest_flags(t):
    """benchmark_valid = vicon_valid and (top_valid or side_valid); valid_views lists the views whose <view>_valid is
    true, in the order of common.VIEWS, n/a when none. One problem line per column, with the first three rows."""
    bad = {"benchmark_valid": [], "valid_views": []}
    for row in t.manifest:
        run = " ".join(str(row.get(k)) for k in ("participant_id", "session", "run"))
        views = [v for v in common.VIEWS if as_bool(row.get(f"{v}_valid"))]
        flag, listed, expected = row.get("benchmark_valid"), row.get("valid_views"), ";".join(views) or "n/a"
        if flag is not None and as_bool(flag) != (as_bool(row.get("vicon_valid")) and bool(views)):
            bad["benchmark_valid"].append(f"{run} ({flag})")
        if listed is not None and listed != expected:
            bad["valid_views"].append(f"{run} ('{listed}', expected '{expected}')")
    rules = {"benchmark_valid": "benchmark_valid is not vicon_valid and ("
                                + " or ".join(f"{v}_valid" for v in common.VIEWS) + ")",
             "valid_views": "valid_views does not list exactly the views whose <view>_valid is true"}
    for col, runs in bad.items():
        if runs:
            more = f" and {len(runs) - 3} more" if len(runs) > 3 else ""
            t.report.problem("numbers", f"{DERIV}/manifest.tsv", f"{rules[col]}: {', '.join(runs[:3])}{more}")


def presence_summary(t):
    """labels2d, keypoints2d and tracks3d files found against the clips they could exist for."""
    r, clips = t.report, defaultdict(set)
    for (sub, ses, run), row in t.runs.items():
        for view in common.VIEWS:
            if clip_exists(row, view):
                clips[view].add((sub, ses, run))
    for view in ("top", "side"):
        expected = {k for k in clips[view] if has_vicon(t.runs[k]) and camera_ok(t, k[0], k[1], view, k[2])}
        found = {k for k in clips[view] if t.has(f"{DERIV}/labels2d/{k[0]}/{k[1]}/" + common.bids_name(
            k[0], k[1], "labels", ".tsv", acq=view, run=k[2]))}
        r.info(f"labels2d {view}: {len(found & expected)}/{len(expected)} clips with VICON data and a usable camera "
               f"(CalibrationOK) have labels; {len(found - expected)} labels files outside that set")
        if found - expected:
            r.problem("completeness", f"{DERIV}/labels2d", f"{len(found - expected)} {view} labels files without VICON "
                                                           "data or a usable camera")
        for k in sorted(found):
            guarded(r, f"{DERIV}/labels2d", check_labels_rows, t, k, view)
    models = sorted({f.split("/")[3] for f in t.files if f.startswith(f"{DERIV}/keypoints2d/") and f.count("/") > 4})
    for model in models:
        n = Counter(m[1] for f in t.files if f.startswith(f"{DERIV}/keypoints2d/{model}/")
                    for m in [re.search(r"_acq-([a-z]+)_.*_keypoints\.tsv$", f)] if m)
        r.info(f"keypoints2d {model}: " + ", ".join(f"{v} {n.get(v, 0)}/{len(clips[v])}" for v in common.VIEWS))
    absent = [m for m in common.ZERO_SHOT + common.WITH_METHOD if m not in models]
    if absent:
        r.info(f"keypoints2d models absent: {absent}")
    unknown = [m for m in models if m not in common.ZERO_SHOT + common.WITH_METHOD]
    if unknown:
        r.problem("format", f"{DERIV}/keypoints2d", f"models not in common.ZERO_SHOT or common.WITH_METHOD: {unknown}")
    nb = Counter(m[1] for f in t.files if f.startswith(f"{DERIV}/boxes/")
                 for m in [re.search(r"_acq-([a-z]+)_.*_boxes\.tsv$", f)] if m)
    r.info("boxes: " + ", ".join(f"{v} {nb.get(v, 0)}/{len(clips[v])}" for v in common.VIEWS))
    n3d = sum(f.endswith("_tracks3d.tsv") for f in t.files)
    r.info(f"tracks3d: {n3d} runs; cameras: {sum(f.startswith(f'{DERIV}/cameras/') for f in t.files)} files")


def check_labels_rows(t, key, view):
    sub, ses, run = key
    rel = f"{DERIV}/labels2d/{sub}/{ses}/" + common.bids_name(sub, ses, "labels", ".tsv", acq=view, run=run)
    n = as_float(t.runs[key].get(f"{view}_n_frames"))
    first = as_float(t.runs[key].get(f"{view}_lead_black_frames")) or 0     # black frames before the video starts
    f = t.facts.get(rel, {})
    rows = f.get("n_rows")
    # no rows past the end of the source video: allow up to 10 % fewer rows than image frames
    if rows is None or n is None or f.get("first_frame", 0) != first or rows > n - first or rows < 0.9 * (n - first):
        t.report.problem("format", rel, f"{rows} rows from frame {f.get('first_frame')}, video has {n} frames "
                                        f"of which frames {first} on show the clip")


def check_videos(t):
    r = t.report
    if t.manifest is None:
        return
    expected = set()
    for (sub, ses, run), row in sorted(t.runs.items()):
        for view in common.VIEWS:
            rel = video_path(sub, ses, run, view)
            if not clip_exists(row, view):
                continue
            expected.add(rel)
            t.need(rel, f"the {view} clip exists")
            f = t.facts.get(rel, {})
            n, fps = as_float(row.get(f"{view}_n_frames")), as_float(row.get(f"{view}_fps"))
            if f.get("fps") and fps and abs(f["fps"] / fps - 1) > 0.005:
                r.problem("format", rel, f"{f['fps']} fps, manifest {fps}")
            if f.get("frames") is not None and n and f["frames"] != n:
                r.problem("format", rel, f"{f['frames']} frames, manifest {view}_n_frames {n:.0f}")
    for rel in sorted(f for f in t.files if f.endswith(".mp4") and f not in expected):
        if not t.report.partial or rel.split("/")[3] in t.subjects:
            r.problem("completeness", rel, "video for a run / view without a clip in manifest.tsv")
    r.info(f"videos: {len(expected & t.files)}/{len(expected)} expected clips present")


# ----------------------------------------------------------------------------- main
def run_file_checks(root, files, terms, jobs, report):
    tasks = [(str(root), rel) for rel in files]
    if jobs > 1:
        with ProcessPoolExecutor(jobs, initializer=init_worker, initargs=(terms, _FFMPEG)) as pool:
            results = list(pool.map(check_file, tasks, chunksize=8))
    else:
        results = list(map(check_file, tasks))
    facts = {}
    for rel, problems, f in results:
        for check, msg in problems:
            report.problem(check, rel, msg)
        facts[rel] = f
    return facts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=common.RELEASE)
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--partial", action="store_true",
                    help="test tree with some participants: completeness only for those present, count mismatches "
                         "as INFO")
    ap.add_argument("--scrub-terms", type=Path, default=common.PRIVATE / "scrub_terms.txt",
                    help="terms that must not appear, one per line (default: the private list, not released); an "
                         "empty file or /dev/null skips the term search")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"no release tree at {root}")
    if not args.scrub_terms.exists():
        sys.exit(f"no scrub-terms file at {args.scrub_terms}: pass one with --scrub-terms FILE (one term per line; "
                 "an empty file skips the term check)")
    report = Report(args.partial)
    terms, n_users = load_terms(args.scrub_terms)
    report.info(f"checking {root} parts {args.parts}{' (partial)' if args.partial else ''}; "
                f"{len(terms) - n_users} scrub terms + {n_users} game user ids")
    files, dirs = list_files(root, args.parts, report)
    init_worker(terms, imageio_ffmpeg.get_ffmpeg_exe())
    check_names(files, dirs, report)
    facts = run_file_checks(root, files, terms, args.jobs, report)
    tree = Tree(root, args.parts, files, facts, report)
    report.info(f"{len(files)} files, {sum(f['size'] for f in facts.values()) / 1e6:.1f} MB; "
                f"{len(tree.runs)} runs of {len({k[0] for k in tree.runs})} participants checked for completeness")
    for part, check in (("core", check_core), ("sourcedata", check_sourcedata), ("derivatives", check_derivatives),
                        ("videos", check_videos)):
        if part in args.parts:
            guarded(report, part, check, tree)
    bids_validator_status(report)
    total = sum(report.counts.values())
    print(f"SUMMARY {total} problems ({', '.join(f'{c} {report.counts[c]}' for c in CHECKS)}) in {len(files)} files")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
