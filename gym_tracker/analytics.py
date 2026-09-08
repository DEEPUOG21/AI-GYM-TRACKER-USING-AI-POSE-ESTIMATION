"""Serializable session metrics; elapsed time is supplied by the video clock."""
from datetime import datetime, timezone
from uuid import uuid4
import copy
import math
from gym_tracker.repetitions import EXERCISES


class WorkoutSession:
    def __init__(self, source):
        self.session_id = str(uuid4())
        self.source = source
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at = None
        self.duration = 0.0
        self.frames = self.valid_pose_frames = 0
        self.metrics = {name: {"exercise": name, "reps": 0, "duration_seconds": 0.0,
                               "first_seen_seconds": None, "last_seen_seconds": None,
                               "rep_timestamps_seconds": [], "confidence_sum": 0.0,
                               "classification_windows": 0} for name in EXERCISES}

    def record(self, timestamp, exercise=None, rep=False, prediction=None, pose_valid=False):
        if self.ended_at is not None:
            raise ValueError("Session has ended")
        if not math.isfinite(timestamp) or timestamp < self.duration:
            raise ValueError("Timestamps must be finite, nonnegative and monotonic")
        if exercise is not None and exercise not in self.metrics:
            raise ValueError("Unknown exercise")
        delta = timestamp - self.duration
        self.duration = float(timestamp)
        self.frames += 1
        self.valid_pose_frames += int(pose_valid)
        if exercise is not None:
            metric = self.metrics[exercise]
            metric["duration_seconds"] += delta
            if metric["first_seen_seconds"] is None:
                metric["first_seen_seconds"] = timestamp - delta
            metric["last_seen_seconds"] = timestamp
            if rep:
                metric["reps"] += 1
                metric["rep_timestamps_seconds"].append(timestamp)
        if prediction is not None:
            metric = self.metrics[prediction.exercise]
            metric["classification_windows"] += 1
            metric["confidence_sum"] += prediction.confidence

    def finish(self):
        if self.ended_at is None:
            self.ended_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self):
        exercises = []
        for value in self.metrics.values():
            metric = copy.deepcopy(value)
            total = metric.pop("confidence_sum")
            count = metric["classification_windows"]
            metric["confidence"] = total / count if count else None
            exercises.append(metric)
        return {"schema_version": 1, "session_id": self.session_id, "source": self.source,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "duration_seconds": self.duration, "frames": self.frames,
                "valid_pose_frames": self.valid_pose_frames,
                "unassigned_duration_seconds": max(0.0, self.duration - sum(m["duration_seconds"] for m in exercises)),
                "exercises": exercises}
