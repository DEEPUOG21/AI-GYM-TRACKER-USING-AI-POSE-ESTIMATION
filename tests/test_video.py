import cv2
import numpy as np
import pytest
from gym_tracker.pipeline import WorkoutTracker
from gym_tracker.video import analyze_video


class MissingPose:
    closed = False

    def detect(self, frame):
        return None

    def close(self):
        self.closed = True


def test_encode_duration_and_cleanup(tmp_path):
    source, output = tmp_path / "source.avi", tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    detector = MissingPose()
    tracker = WorkoutTracker(detector, exercise="squat")
    report = analyze_video(source, output, tracker)
    assert report["frames"] == 10 and report["duration_seconds"] == 1
    assert report["valid_pose_frames"] == 0
    assert report["unassigned_duration_seconds"] == 1
    assert output.stat().st_size > 0 and detector.closed
    decoded = cv2.VideoCapture(str(output))
    try:
        assert decoded.read()[0]
    finally:
        decoded.release()


def test_bad_video_releases_tracker(tmp_path):
    detector = MissingPose()
    tracker = WorkoutTracker(detector, exercise="squat")
    with pytest.raises(ValueError):
        analyze_video(tmp_path / "missing.mp4", tmp_path / "out.mp4", tracker)
    assert detector.closed and tracker.snapshot()["ended_at"]
