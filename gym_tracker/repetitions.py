"""Exercise-specific angle thresholds and deterministic repetition phases."""
import numpy as np
from gym_tracker.features import calculate_angle, valid_landmarks

EXERCISES = ("bicep_curl", "push_up", "squat", "shoulder_press")
ARM_TRIPLES = ((12, 14, 16), (11, 13, 15))
LEG_TRIPLES = ((24, 26, 28), (23, 25, 27))


class RepCounter:
    def __init__(self, exercise):
        if exercise not in EXERCISES:
            raise ValueError(f"Unsupported exercise: {exercise}")
        self.exercise = exercise
        self.reps = 0
        self.reset_phase()

    def reset_phase(self):
        self.stage = None
        self.right_ready = self.left_ready = False
        self._tracked_side = None

    def accepts_landmarks(self, landmarks):
        if self.exercise in ("shoulder_press", "bicep_curl"):
            return valid_landmarks(landmarks, required_ids=range(11, 17))
        triples = LEG_TRIPLES if self.exercise == "squat" else ARM_TRIPLES
        return any(valid_landmarks(landmarks, required_ids=ids) for ids in triples)

    def update(self, landmarks, frame_size):
        if not self.accepts_landmarks(landmarks):
            self.reset_phase()
            return False
        # Pixel coordinates retain the original aspect ratio and integer rounding.
        p = (np.asarray(landmarks)[:, :2] * np.asarray(frame_size)).astype(int)
        if self.exercise == "shoulder_press" and not (
                p[15, 1] < p[11, 1] and p[16, 1] < p[12, 1]):
            # Lowering the arms to the sides must not complete an overhead press.
            self.reset_phase()
            return False
        triples = LEG_TRIPLES if self.exercise == "squat" else ARM_TRIPLES
        if self.exercise in ("push_up", "squat"):
            visible = [i for i, ids in enumerate(triples)
                       if valid_landmarks(landmarks, required_ids=ids)]
            side = self._tracked_side
            if side not in visible:
                self.reset_phase()
                side = max(visible, key=lambda i: np.asarray(landmarks)[list(triples[i]), 3].min())
                self._tracked_side = side
            a, b, c = triples[side]
            angle = calculate_angle(p[a], p[b], p[c])
            return self.update_angles(angle, angle)
        angles = [calculate_angle(p[a], p[b], p[c], directed=True) for a, b, c in triples]
        return self.update_angles(*angles)

    def update_angles(self, right, left):
        """Consume right/left directed angles; emit True only on a new rep."""
        if any(a is None or not np.isfinite(a) or not 0 <= a < 360 for a in (right, left)):
            self.reset_phase()
            return False
        right, left = min(right, 360 - right), min(left, 360 - left)
        completed = False
        if self.exercise == "bicep_curl":
            if right > 160:
                self.right_ready = True
            if left > 140:
                self.left_ready = True
            completed = self.right_ready and self.left_ready and right < 60 and left < 60
            if completed:
                self.right_ready = self.left_ready = False
        else:
            if self.exercise == "push_up":
                ready, finish = left >= 140, left <= 120
            elif self.exercise == "squat":
                ready, finish = right >= 160 and left >= 160, right <= 140 and left <= 140
            else:
                ready = right <= 110 and left <= 110
                finish = right >= 155 and left >= 155
            if ready:
                self.stage = "ready"
            completed = finish and self.stage == "ready"
            if completed:
                self.stage = "complete"
        if completed:
            self.reps += 1
        return bool(completed)
