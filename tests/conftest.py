import numpy as np
import pytest


@pytest.fixture
def exercise_pose():
    def make(exercise, angle, *, mirrored=False, side=None):
        points = np.zeros((33, 4))
        triples = ((24, 26, 28), (23, 25, 27)) if exercise == "squat" else ((12, 14, 16), (11, 13, 15))
        for i, (a, b, c) in enumerate(triples):
            points[b, :2] = (0.3 + i * 0.4, 0.5)
            points[a, :2] = points[b, :2] + (0.15, 0)
            radians = np.radians(angle)
            points[c, :2] = points[b, :2] + (0.15 * np.cos(radians), 0.2 * np.sin(radians))
            if side is None or side == i:
                points[[a, b, c], 3] = 0.99
        if mirrored:
            points[:, 0] = 1 - points[:, 0]
        return points
    return make


@pytest.fixture
def landmarks():
    # Synthetic geometry for unit tests only, never evaluation data.
    rng = np.random.default_rng(42)
    points = rng.uniform(0.1, 0.9, (33, 4))
    points[:, 3] = 0.99
    return points


@pytest.fixture
def shoulder_pose():
    def make(*, extended=False, mirrored=False):
        points = np.zeros((33, 4))
        points[11:17, 3] = 0.99  # Legs are outside the camera frame.
        points[[11, 12], :2] = [[0.4, 0.6], [0.6, 0.6]]
        if extended:
            points[[13, 14], :2] = [[0.4, 0.4], [0.6, 0.4]]
            points[[15, 16], :2] = [[0.4, 0.2], [0.6, 0.2]]
        else:
            points[[13, 14], :2] = [[0.2, 0.6], [0.8, 0.6]]
            points[[15, 16], :2] = [[0.2, 0.4], [0.8, 0.4]]
        if mirrored:
            points[:, 0] = 1 - points[:, 0]
        return points
    return make
