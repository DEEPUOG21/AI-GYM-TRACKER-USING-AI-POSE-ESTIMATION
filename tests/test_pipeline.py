from types import SimpleNamespace
import numpy as np
import pytest
from gym_tracker.classification import Prediction
from gym_tracker.pipeline import WorkoutTracker


class Detector:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.calls = self.closed = 0

    def detect(self, frame):
        self.calls += 1
        return self.landmarks

    def close(self):
        self.closed += 1


def test_one_detector_and_nonoverlapping_windows(landmarks):
    detector = Detector(landmarks)
    predictions = []

    def predict(window):
        predictions.append(np.asarray(window))
        return Prediction("squat", 0.8, {})

    tracker = WorkoutTracker(detector, classifier=SimpleNamespace(predict=predict))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(60):
        tracker.process(frame, (i + 1) / 30)
    assert len(predictions) == 2
    assert detector.calls == 60
    assert tracker.snapshot()["duration_seconds"] == 2
    detector.landmarks = None
    tracker.process(frame, 3)
    assert tracker.prediction is None and tracker.window == []
    detector.landmarks = landmarks
    for i in range(29):
        tracker.process(frame, 4 + i)
    assert len(predictions) == 2
    tracker.close()
    tracker.close()
    assert detector.closed == 1
    assert tracker.snapshot()["ended_at"] is not None


def test_session_isolation(landmarks):
    first = WorkoutTracker(Detector(landmarks), exercise="squat")
    second = WorkoutTracker(Detector(landmarks), exercise="squat")
    first.counters["squat"].update_angles(180, 180)
    first.counters["squat"].update_angles(120, 230)
    assert second.counters["squat"].reps == 0
    assert first.session.session_id != second.session.session_id
    first.close()
    second.close()


def test_manual_shoulder_press_counts_without_visible_legs(shoulder_pose):
    detector = Detector(shoulder_pose())
    tracker = WorkoutTracker(detector, exercise="shoulder_press")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.process(frame, 0)
    detector.landmarks = shoulder_pose(extended=True)
    tracker.process(frame, 1)
    assert tracker.counters["shoulder_press"].reps == 1
    metric = next(m for m in tracker.snapshot()["exercises"] if m["exercise"] == "shoulder_press")
    assert metric["reps"] == 1
    assert metric["rep_timestamps_seconds"] == [1]
    tracker.close()


def test_auto_still_requires_full_body(shoulder_pose):
    def unexpected_prediction(window):
        raise AssertionError("Partial poses must not reach the classifier")

    tracker = WorkoutTracker(Detector(shoulder_pose()),
                             classifier=SimpleNamespace(predict=unexpected_prediction))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for timestamp in range(31):
        tracker.process(frame, timestamp)
    assert tracker.window == []
    assert tracker.current_exercise is None
    tracker.close()


@pytest.mark.parametrize("exercise,finish", [("bicep_curl", 40), ("push_up", 100), ("squat", 120)])
def test_manual_partial_pose_reaches_session(exercise_pose, exercise, finish):
    detector = Detector(exercise_pose(exercise, 180))
    tracker = WorkoutTracker(detector, exercise=exercise)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.process(frame, 0, stop_on_gesture=True)
    assert not tracker.stopped  # Invisible wrists in squat mode cannot end the session.
    detector.landmarks = exercise_pose(exercise, finish)
    tracker.process(frame, 1, stop_on_gesture=True)
    metric = next(m for m in tracker.snapshot()["exercises"] if m["exercise"] == exercise)
    assert metric["reps"] == 1
    assert metric["rep_timestamps_seconds"] == [1]
    assert not tracker.stopped
    tracker.close()
