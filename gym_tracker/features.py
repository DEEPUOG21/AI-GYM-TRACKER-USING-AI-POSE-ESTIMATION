"""The original classifier's ordered 22-feature contract.

Coordinates are normalized MediaPipe x/y/z. Angles use x/y; distances use x/y/z.
Zero is a valid coordinate. Invalid poses are rejected instead of fabricated.
"""
import numpy as np

LANDMARK_IDS = tuple(range(11, 17)) + tuple(range(23, 29))
ANGLE_TRIPLES = ((0, 2, 4), (1, 3, 5), (6, 8, 10), (7, 9, 11),
                 (0, 6, 8), (1, 7, 9), (6, 0, 2), (7, 1, 3))
DISTANCE_PAIRS = ((0, 1), (6, 7), (6, 8), (7, 9), (0, 6), (1, 7),
                  (2, 8), (3, 9), (4, 0), (5, 1), (4, 6), (5, 7))


def calculate_angle(a, b, c, *, directed=False):
    """Return a 2D joint angle, or None for missing/degenerate points.

    directed=True retains the legacy counter's orientation-sensitive 0–360°.
    """
    try:
        points = np.asarray([a, b, c], dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] not in (2, 3) or not np.isfinite(points).all():
        return None
    u, v = points[0, :2] - points[1, :2], points[2, :2] - points[1, :2]
    if np.linalg.norm(u) < 1e-10 or np.linalg.norm(v) < 1e-10:
        return None
    angle = float(np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])) % 360)
    return angle if directed else min(angle, 360 - angle)


def valid_landmarks(landmarks, min_visibility=0.5, *, required_ids=LANDMARK_IDS):
    """Validate pose data and visibility of the joints required by the caller."""
    try:
        points = np.asarray(landmarks, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(points.shape == (33, 4) and np.isfinite(points).all()
                and np.all((points[:, 3] >= 0) & (points[:, 3] <= 1))
                and np.all(points[list(required_ids), 3] >= min_visibility))


def extract_features(landmarks):
    """Return 22 features in training order, or None for an unusable pose."""
    if not valid_landmarks(landmarks):
        return None
    p = np.asarray(landmarks, dtype=float)[list(LANDMARK_IDS), :3]
    angles = [calculate_angle(p[a], p[b], p[c]) for a, b, c in ANGLE_TRIPLES]
    if any(a is None for a in angles):
        return None
    distances = [float(np.linalg.norm(p[a] - p[b])) for a, b in DISTANCE_PAIRS]
    scale = next((distances[i] for i in (4, 5, 2, 3) if distances[i] > 1e-10), None)
    if scale is None:
        return None
    return np.asarray(angles + [d / scale for d in distances]
                      + [abs(p[2, 1] - p[0, 1]) / scale,
                         abs(p[3, 1] - p[1, 1]) / scale], dtype=np.float64)
