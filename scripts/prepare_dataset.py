"""Extract real labeled evaluation windows from a video manifest CSV."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from gym_tracker.features import extract_features
from gym_tracker.pose import PoseDetector
from gym_tracker.repetitions import EXERCISES


def read_manifest(path, split):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "exercise", "subject_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path, exercise, subject_id, split columns")
        rows = list(reader)
    subjects, videos = {}, set()
    for row in rows:
        if (row["exercise"] not in EXERCISES or not row["subject_id"].strip()
                or row["split"] not in ("train", "validation", "test")):
            raise ValueError("Invalid exercise, subject or split in manifest")
        subject = row["subject_id"]
        if subject in subjects and subjects[subject] != row["split"]:
            raise ValueError(f"Subject {subject} appears in multiple splits")
        subjects[subject] = row["split"]
        video = (path.parent / row["path"]).resolve()
        if video in videos:
            raise ValueError(f"Duplicate video in manifest: {video}")
        videos.add(video)
        row["path"] = video
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError("No videos in selected split")
    return selected


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("data/test.npz"))
    args = parser.parse_args()
    rows = read_manifest(args.manifest, args.split)
    windows, labels, sources = [], [], []
    for row in rows:
        detector = PoseDetector()
        cap = cv2.VideoCapture(str(row["path"]))
        sequence, frames, invalid, count = [], 0, 0, 0
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open {row['path']}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                features = extract_features(detector.detect(frame))
                if features is None:
                    invalid += 1
                    sequence.clear()
                    continue
                sequence.append(features)
                if len(sequence) == 30:
                    windows.append(np.stack(sequence))
                    labels.append(row["exercise"])
                    count += 1
                    sequence.clear()
        finally:
            cap.release()
            detector.close()
        sources.append({"path": str(row["path"]), "subject_id": row["subject_id"],
                        "exercise": row["exercise"], "frames": frames,
                        "invalid_pose_frames": invalid, "windows": count})
    if not windows:
        raise ValueError("No complete valid windows extracted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, X=np.stack(windows), y=np.asarray(labels))
    args.output.with_suffix(".json").write_text(json.dumps({"split": args.split, "sources": sources}, indent=2) + "\n")
    print(f"Saved {len(windows)} windows to {args.output}")


if __name__ == "__main__":
    main()
