"""Root documents of the ViKi-Traj release: README.md, LICENSE, CHANGES and CITATION.cff.

README.md is a copy of README_card.md, the dataset card with its Hugging Face YAML header. All four documents are
anonymous for double-blind review.

    python dataset/build_docs.py [--out release/ViKi-Traj]
"""
import argparse
import shutil
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent

LICENSE = """ViKi-Traj review terms (version 1.0, review copy)

This copy of ViKi-Traj is shared with the reviewers of an anonymous submission. The license of the public release
will be stated here at publication.

Until then, the data may be used only to review the submission. By accessing the data you agree:
  1. not to attempt to identify any participant or any other person shown in the data;
  2. not to share the data with anyone else;
  3. to delete the data at the end of the review or on request.

The data are provided as is, without warranty of any kind.
"""

CHANGES = """1.0.0 Unknown Release Date (review copy)
  - Initial release: 29 participants, 696 trials of the reaching task (VICON, end-effector trajectory, top and
    side video clips, labels, model keypoints, person boxes, 3D tracks, results and ViTMS checkpoints).
"""

CITATION = """cff-version: 1.2.0
message: "If you use ViKi-Traj, please cite the paper (citation withheld for double-blind review)."
title: "ViKi-Traj"
type: dataset
version: "1.0.0"
authors:
  - name: "Anonymous (double-blind review)"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=common.RELEASE)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "README_card.md", out / "README.md")
    for name, text in (("LICENSE", LICENSE), ("CHANGES", CHANGES), ("CITATION.cff", CITATION)):
        (out / name).write_text(text)
    print(f"wrote README.md, LICENSE, CHANGES, CITATION.cff to {out}")


if __name__ == "__main__":
    main()
