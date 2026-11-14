"""Copy one trial of the released dataset into a small tree with the same layout.

Keeps the trial's rows of the participant, phenotype, recordings and manifest tables, and copies the visit's beh,
trajectory and camera files, the C3D recording that contains the trial and the trial's motion, events, video clips and
derivative files, with their sidecars. Checkpoints, results and the root documents are left out.

    python scripts/make_sample.py [--root release/ViKi-Traj] [--trial sub-03 ses-1 run-06]
        [--out sample_data/ViKi-Traj]
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIV = "derivatives/vikitraj"


def read_rows(path):
    """Header line and (line, row dict) per data line of a TSV."""
    lines = path.read_bytes().decode().splitlines(keepends=True)
    names = lines[0].rstrip("\r\n").split("\t")
    return lines[0], [(s, dict(zip(names, s.rstrip("\r\n").split("\t")))) for s in lines[1:]]


def copy_rows(root, out, rel, keep):
    """Copy a TSV with its header and the lines whose row passes keep, unchanged; returns those rows."""
    header, data = read_rows(root / rel)
    kept = [(s, r) for s, r in data if keep(r)]
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    (out / rel).write_bytes((header + "".join(s for s, _ in kept)).encode())
    return [r for _, r in kept]


def glob(root, pattern):
    return sorted(p.relative_to(root).as_posix() for p in root.glob(pattern))


def trial_files(root, t):
    """Files of the trial that are copied whole, relative to the root."""
    sub, ses, run = t["participant_id"], t["session"], t["run"]
    patterns = ["dataset_description.json", "participants.json", "task-*.json", "phenotype/*.json",
                f"{sub}/{ses}/beh/*", f"{sub}/{ses}/motion/*_{run}_*", f"sourcedata/trajectory/{sub}/{ses}/*",
                "sourcedata/vicon/recordings.json",
                *(f"{DERIV}/{name}" for name in ("dataset_description.json", ".bidsignore", "README.md",
                                                 "manifest.json", "splits.json")),
                f"{DERIV}/cameras/{sub}_{ses}_*", f"{DERIV}/*/task-*.json", f"{DERIV}/keypoints2d/*/task-*.json",
                f"{DERIV}/tracks3d/{sub}/{ses}/*_{run}_tracks3d.tsv"]
    for view in ("top", "side"):
        if t[f"{view}_video"] != "n/a":
            patterns.append(t[f"{view}_video"])
        labels = t[f"{view}_labels"]
        if labels != "n/a":
            stem = labels[len(f"{DERIV}/labels2d/"):-len("_labels.tsv")]       # sub-XX/ses-N/<clip>
            patterns += [labels, f"{DERIV}/boxes/{stem}_boxes.tsv", f"{DERIV}/keypoints2d/*/{stem}_keypoints.tsv"]
    return [rel for p in patterns for rel in glob(root, p)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT / "release" / "ViKi-Traj", help="released dataset")
    ap.add_argument("--trial", nargs=3, default=["sub-03", "ses-1", "run-06"], metavar=("SUB", "SES", "RUN"))
    ap.add_argument("--out", type=Path, default=ROOT / "sample_data" / "ViKi-Traj")
    args = ap.parse_args()
    root, out = args.root, args.out
    sub, ses, run = args.trial
    number = str(int(run.split("-")[1]))                    # recordings.tsv lists runs as 1,2,3,4

    def is_trial(r):
        return (r["participant_id"], r["session"], r["run"]) == (sub, ses, run)

    def is_participant(r):
        return r["participant_id"] == sub

    def has_trial(r):
        return r["participant_id"] == sub and r["session"] == ses and number in r["runs"].split(",")

    trial = [r for _, r in read_rows(root / DERIV / "manifest.tsv")[1] if is_trial(r)]
    if not trial:
        raise SystemExit(f"{sub} {ses} {run} is not in the manifest")
    copy_rows(root, out, f"{DERIV}/manifest.tsv", is_trial)
    for rel in ["participants.tsv"] + glob(root, "phenotype/*.tsv"):
        copy_rows(root, out, rel, is_participant)
    recordings = copy_rows(root, out, "sourcedata/vicon/recordings.tsv", has_trial)
    for rel in trial_files(root, trial[0]) + [r["file"] for r in recordings]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, out / rel)
    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"{out}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
