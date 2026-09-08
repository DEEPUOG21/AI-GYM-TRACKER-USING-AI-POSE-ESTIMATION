"""One stateful MediaPipe detector per video/stream, never per frame."""
import numpy as np


class PoseDetector:
    def __init__(self):
        import mediapipe as mp
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._closed = False

    def detect(self, frame):
        import cv2
        if self._closed:
            raise RuntimeError("Pose detector is closed")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Expected a nonempty BGR image")
        result = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        self._mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, self._mp.solutions.pose.POSE_CONNECTIONS)
        return np.asarray([[p.x, p.y, p.z, p.visibility]
                           for p in result.pose_landmarks.landmark], dtype=float)

    def close(self):
        if not self._closed:
            self._pose.close()
            self._closed = True
