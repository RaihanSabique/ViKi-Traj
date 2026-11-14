"""Contact sheets of the released clips for a visual check of the de-identification (kept in release/private).

Five frames of every clip (first, 1/4, 1/2, 3/4 and last; black lead frames skipped) as 480 px wide tiles labelled
with clip and frame, 4 x 4 tiles per sheet, one series of sheets per participant and view; index.tsv lists the sheets.

    python dataset/qa_contact_sheets.py [--root release/ViKi-Traj] [--out release/private/qa_sheets]
                                        [--participant sub-01 sub-02 ...]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import common

TILE_W, GRID = 480, 4
POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def frames_of(path, positions):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for p in positions:
        k = int(round(p * (n - 1)))
        for _ in range(60):                     # skip black lead frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            if not ok or img.mean() > 2 or k >= n - 1:
                break
            k += max(1, n // 50)
        if ok:
            out.append((k, img))
    cap.release()
    return out


def tile(img, label):
    h = int(img.shape[0] * TILE_W / img.shape[1])
    t = cv2.resize(img, (TILE_W, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (TILE_W, 18), (0, 0, 0), -1)
    cv2.putText(t, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return cv2.resize(t, (TILE_W, 270)) if t.shape[0] != 270 else t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=common.RELEASE)
    ap.add_argument("--out", type=Path, default=common.PRIVATE / "qa_sheets")
    ap.add_argument("--participant", nargs="*", help="released participant IDs (sub-NN)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    vids = a.root / common.VIDEOS
    rows = []
    for sub_dir in sorted(vids.glob("sub-*")):
        if a.participant and sub_dir.name not in a.participant:
            continue
        for view in common.VIEWS:
            tiles = []
            for f in sorted(sub_dir.glob(f"*/*_acq-{view}_*_video.mp4")):
                short = f.name.replace(f"_task-{common.TASK}", "").replace("_video.mp4", "")
                for k, img in frames_of(f, POSITIONS):
                    tiles.append(tile(img, f"{short} f{k}"))
            for s in range(0, len(tiles), GRID * GRID):
                chunk = tiles[s:s + GRID * GRID]
                chunk += [np.zeros_like(tiles[0])] * (GRID * GRID - len(chunk))
                sheet = np.vstack([np.hstack(chunk[r * GRID:(r + 1) * GRID]) for r in range(GRID)])
                name = f"{sub_dir.name}_{view}_{s // (GRID * GRID):02d}.jpg"
                cv2.imwrite(str(a.out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
                rows.append({"sheet": name, "participant_id": sub_dir.name, "view": view, "tiles": len(tiles[s:s + 16])})
    pd.DataFrame(rows).to_csv(a.out / "index.tsv", sep="\t", index=False)
    print(f"{len(rows)} sheets in {a.out}")


if __name__ == "__main__":
    main()
