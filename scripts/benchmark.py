"""Measure actual decode + pose + features + counting + annotation latency."""
import argparse
import json
import platform
import os
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
from gym_tracker.classification import MODEL_DIR, get_classifier
from gym_tracker.pipeline import WorkoutTracker
from gym_tracker.pose import PoseDetector
from gym_tracker.repetitions import EXERCISES
from scripts.evaluate import sha256


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--exercise", choices=(*EXERCISES, "auto"), default="auto")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.max_frames <= 0:
        parser.error("warmup must be nonnegative and max-frames positive")
    start = perf_counter()
    classifier = get_classifier() if args.exercise == "auto" else None
    tracker = WorkoutTracker(PoseDetector(), None if classifier else args.exercise, classifier, "benchmark")
    initialization = perf_counter() - start
    cap = cv2.VideoCapture(str(args.video))
    samples = []
    windows_before = 0
    dimensions = None
    try:
        if not cap.isOpened():
            raise ValueError("Unable to open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has invalid FPS")
        for index in range(args.warmup + args.max_frames):
            start = perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            dimensions = [frame.shape[1], frame.shape[0]]
            tracker.process(frame, (index + 1) / fps)
            elapsed = (perf_counter() - start) * 1000
            if index < args.warmup:
                windows_before = sum(m["classification_windows"] for m in tracker.snapshot()["exercises"])
            else:
                samples.append(elapsed)
    finally:
        cap.release()
        tracker.close()
    if not samples:
        raise ValueError("No frames measured after warmup; use a longer video or reduce --warmup")
    snapshot = tracker.snapshot()
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "video_sha256": sha256(args.video),
              "versions": {name: version(name) for name in ("numpy", "tensorflow-cpu", "keras", "mediapipe", "opencv-contrib-python", "scikit-learn")},
              "cpu_count": os.cpu_count(), "frame_size": dimensions, "source_fps": fps,
              "thread_settings": {key: os.getenv(key) for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
              "artifact_sha256": {p.name: sha256(p) for p in MODEL_DIR.iterdir() if p.suffix in (".h5", ".pkl")},
              "mode": args.exercise, "initialization_seconds": initialization,
              "warmup_frames": args.warmup, "measured_frames": len(samples),
              "mean_ms": float(np.mean(samples)), "p50_ms": float(np.percentile(samples, 50)),
              "p95_ms": float(np.percentile(samples, 95)), "throughput_fps": 1000 / float(np.mean(samples)),
              "classification_windows_measured": sum(m["classification_windows"] for m in snapshot["exercises"]) - windows_before,
              "scope": "Video decode + pose + features + scheduled classification + counting + annotation; excludes UI, network, output encoding. Results apply only to this input and machine.",
              "session": snapshot}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
