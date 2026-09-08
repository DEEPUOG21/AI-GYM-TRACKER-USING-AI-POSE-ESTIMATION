"""Own stream state and orchestrate inference without importing Streamlit."""
from threading import RLock
import numpy as np
from gym_tracker.analytics import WorkoutSession
from gym_tracker.features import extract_features, valid_landmarks
from gym_tracker.repetitions import EXERCISES, RepCounter


class WorkoutTracker:
    def __init__(self, detector, exercise=None, classifier=None, source="video"):
        if exercise is not None and exercise not in EXERCISES:
            raise ValueError("Unknown exercise")
        if exercise is None and classifier is None:
            raise ValueError("Auto mode requires a classifier")
        self.detector, self.classifier = detector, classifier
        self.exercise = exercise
        self.counters = {name: RepCounter(name) for name in EXERCISES}
        self.session = WorkoutSession(source)
        self.window = []
        self.prediction = None
        self.current_exercise = exercise
        self.stopped = False
        self._closed = False
        self._lock = RLock()

    def process(self, frame, timestamp, *, stop_on_gesture=False):
        with self._lock:
            if self._closed:
                raise RuntimeError("Tracker is closed")
            if self.stopped:
                return frame
            if not np.isfinite(timestamp) or timestamp < self.session.duration:
                raise ValueError("Timestamps must be finite and monotonic")
            landmarks = self.detector.detect(frame)
            valid = (self.counters[self.exercise].accepts_landmarks(landmarks)
                     if self.exercise else valid_landmarks(landmarks))
            new_prediction = None
            previous = self.current_exercise
            if not valid:
                self.window.clear()
                self.prediction = None
                self.current_exercise = self.exercise
                for counter in self.counters.values():
                    counter.reset_phase()
            elif self.exercise is None:
                features = extract_features(landmarks)
                if features is None:
                    valid = False
                    self.window.clear()
                    self.prediction = None
                    self.current_exercise = None
                    for counter in self.counters.values():
                        counter.reset_phase()
                else:
                    self.window.append(features)
                    if len(self.window) == 30:
                        new_prediction = self.classifier.predict(self.window)
                        self.window.clear()
                        self.prediction = new_prediction
                        self.current_exercise = new_prediction.exercise
            if previous != self.current_exercise:
                for counter in self.counters.values():
                    counter.reset_phase()
            active = self.current_exercise if valid else None
            rep = False
            if active:
                rep = self.counters[active].update(landmarks, (frame.shape[1], frame.shape[0]))
            self.session.record(timestamp, active, rep, new_prediction, valid)
            if stop_on_gesture and valid and valid_landmarks(landmarks, required_ids=(15, 16)):
                wrists = landmarks[[15, 16], :2] * (frame.shape[1], frame.shape[0])
                self.stopped = bool(np.linalg.norm(wrists[0] - wrists[1]) < 30)
            return self._annotate(frame, active)

    def _annotate(self, frame, active):
        import cv2
        text = active or "Waiting for a clear pose / 30-frame window"
        if active:
            text += f" | Reps: {self.counters[active].reps}"
        if self.exercise == "shoulder_press":
            if not active:
                cue = "Keep both shoulders, elbows and wrists visible"
            elif self.counters[active].stage == "ready":
                cue = "Extend both arms overhead"
            else:
                cue = "Bend both elbows with hands above shoulders"
            cv2.putText(frame, cue, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1)
        elif self.exercise:
            visibility = {"bicep_curl": "Keep both shoulders, elbows and wrists visible",
                          "push_up": "Keep a shoulder, elbow and wrist visible from the side",
                          "squat": "Keep a hip, knee and ankle visible from the side"}
            if not active:
                cue = visibility[self.exercise]
            elif active == "bicep_curl":
                counter = self.counters[active]
                cue = "Curl both arms" if counter.right_ready and counter.left_ready else "Extend both arms to begin"
            elif self.counters[active].stage == "ready":
                cue = "Lower into the movement to count"
            else:
                cue = "Return to extension to begin"
            cv2.putText(frame, cue, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1)
        if self.prediction:
            text += f" | Confidence: {self.prediction.confidence:.2f}"
        cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2)
        return frame

    def snapshot(self):
        with self._lock:
            return self.session.snapshot()

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    self.detector.close()
                finally:
                    self.session.finish()
                    self._closed = True
