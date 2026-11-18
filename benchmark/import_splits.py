"""Write outputs/benchmark/splits.json, the participant splits of the paper, from the splits.json of the release.

The split was drawn once (test set and five development folds, paper Sec. 5) and is released with public IDs in
derivatives/vikitraj/splits.json. The private ID map (columns study_id, participant_id) gives back the study IDs
that the benchmark, dataset and paper scripts use; the sessions of cross_visit become visit numbers.

    python benchmark/import_splits.py [--release release/ViKi-Traj] [--id-map release/private/id_map.tsv]
"""
import argparse
import csv
import json
from pathlib import Path

from benchlib import BENCH, ROOT

SPLITS_NOTE = ("Participant splits of the paper with study IDs (derivatives/vikitraj/splits.json of the release). "
               "folds lists fold 1 ... fold 5; ViTMS trains on folds 2-5, stops early on fold 1 and is scored "
               "on the test participants. cross_visit: train and test visit (not used in the paper).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, default=ROOT / "release" / "ViKi-Traj", help="root of the released dataset")
    ap.add_argument("--id-map", type=Path, default=ROOT / "release" / "private" / "id_map.tsv",
                    help="TSV with the columns study_id and participant_id")
    args = ap.parse_args()
    with open(args.id_map, newline="") as f:
        study = {r["participant_id"]: r["study_id"] for r in csv.DictReader(f, delimiter="\t")}
    rel = json.loads((args.release / "derivatives" / "vikitraj" / "splits.json").read_text())

    def ids(sub):
        missing = [s for s in sub if s not in study]
        if missing:
            raise SystemExit(f"not in {args.id_map}: {', '.join(missing)}")
        return [study[s] for s in sub]

    visit = lambda ses: int(str(ses).split("-")[1])                     # "ses-1" -> 1
    out = {"note": SPLITS_NOTE, "test": ids(rel["test"]), "folds": [ids(f) for f in rel["folds"]],
           "cross_visit": {"train_visit": visit(rel["cross_visit"]["train_session"]),
                           "test_visit": visit(rel["cross_visit"]["test_session"])}}
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "splits.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {BENCH / 'splits.json'}: {len(out['test'])} test participants, "
          f"folds of {', '.join(str(len(f)) for f in out['folds'])}")


if __name__ == "__main__":
    main()
