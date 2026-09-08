"""WebRTC adapter with synchronized frame processing and shutdown."""
import logging
from threading import RLock
from time import monotonic

import av
from streamlit_webrtc import VideoProcessorBase

logger = logging.getLogger(__name__)


class WorkoutVideoProcessor(VideoProcessorBase):
    def __init__(self, mode, tracker_factory):
        self.mode = mode
        self._lock = RLock()
        self.started = None
        self.error = None
        self.finished = False
        self.tracker = None
        # The WebRTC factory runs during SDP negotiation, which has a 10s timeout.
        # Load models only when the asynchronous video worker receives its first frame.
        self._tracker_factory = tracker_factory

    def recv(self, frame):
        with self._lock:
            if self.finished:
                return frame
            try:
                if self.tracker is None:
                    try:
                        self.tracker = self._tracker_factory(self.mode, "browser_webcam")
                    except Exception:
                        logger.exception("Webcam tracker initialization failed")
                        self.error = "Webcam model could not initialize. Check dependencies and server logs."
                        self.finished = True
                        return frame
                if self.started is None:
                    self.started = monotonic()
                output = self.tracker.process(frame.to_ndarray(format="bgr24"),
                                              monotonic() - self.started, stop_on_gesture=True)
                if self.tracker.stopped:
                    self.on_ended()
                return av.VideoFrame.from_ndarray(output, format="bgr24")
            except Exception:
                logger.exception("Webcam inference failed")
                self.error = "Webcam inference stopped. See server logs for details."
                self.on_ended()
                return frame

    def on_ended(self):
        with self._lock:
            if self.finished:
                return
            self.finished = True
            try:
                if self.tracker is not None:
                    self.tracker.close()
            except Exception:
                logger.exception("Webcam cleanup failed")
                self.error = "Webcam cleanup failed. See server logs for details."

    def snapshot(self):
        with self._lock:
            return self.tracker.snapshot() if self.tracker is not None else None
