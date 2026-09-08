"""Evaluate the bundled model on held-out, unscaled (N, 30, 22) windows."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from gym_tracker.classification import MODEL_DIR, get_classifier
from gym_tracker.evaluation import classification_metrics
from gym_tracker.repetitions import EXERCISES


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if digest is not None:
            return digest.hexdigest()
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        x, y = data["X"], data["y"]
    if x.ndim != 3 or x.shape[1:] != (30, 22) or not len(x) or not np.isfinite(x).all():
        raise ValueError("X must contain finite unscaled features with shape (N, 30, 22)")
    if y.shape != (len(x),) or y.dtype.kind not in "US" or not set(y).issubset(EXERCISES):
        raise ValueError("y must contain N canonical exercise-name strings")
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split-description", required=True, help="Explain held-out subjects/sessions and dataset provenance")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    x, y = load_dataset(args.dataset)
    classifier = get_classifier()
    predicted = [classifier.predict(window).exercise for window in x]
    report = classification_metrics(y, predicted, list(EXERCISES))
    report.update({"created_at": datetime.now(timezone.utc).isoformat(),
                   "evaluation_unit": "30-frame window", "split_description": args.split_description,
                   "dataset_sha256": sha256(args.dataset),
                   "artifact_sha256": {p.name: sha256(p) for p in MODEL_DIR.iterdir() if p.suffix in (".h5", ".pkl")},
                   "methodology": "Fixed bundled scaler/model; no fitting on evaluation data. Matrix rows=true, columns=predicted. Undefined precision/recall/F1 set to zero."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
