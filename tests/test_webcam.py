from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
import pytest

pytest.importorskip("streamlit_webrtc")
import av  # noqa: E402
from gym_tracker.ui.webcam import WorkoutVideoProcessor  # noqa: E402


def test_initialization_failure_is_reported():
    def fail(*args):
        raise OSError("Model missing")

    processor = WorkoutVideoProcessor("squat", fail)
    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="bgr24")
    assert processor.tracker is None and processor.error is None
    assert processor.recv(frame) is frame
    assert processor.error and processor.finished
    assert processor.snapshot() is None
    processor.on_ended()


def test_factory_is_lightweight_and_stop_before_first_frame():
    calls = []
    processor = WorkoutVideoProcessor("auto", lambda *args: calls.append(args))
    assert not calls  # Negotiating the camera must not import or load ML models.
    processor.on_ended()
    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="bgr24")
    assert processor.recv(frame) is frame
    assert not calls and processor.snapshot() is None


def test_shutdown_waits_for_frame_and_closes_once():
    entered, release = Event(), Event()

    class Tracker:
        stopped = False
        closed = 0

        def process(self, frame, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            assert self.closed == 0
            return frame

        def close(self):
            self.closed += 1

    tracker = Tracker()
    processor = WorkoutVideoProcessor("squat", lambda *a: tracker)
    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="bgr24")
    with ThreadPoolExecutor(2) as pool:
        processing = pool.submit(processor.recv, frame)
        assert entered.wait(5)
        stopping = pool.submit(processor.on_ended)
        release.set()
        processing.result(timeout=5)
        stopping.result(timeout=5)
    processor.on_ended()
    assert tracker.closed == 1 and processor.error is None
    assert processor.recv(frame) is frame
