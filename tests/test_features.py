import numpy as np
import pytest
from gym_tracker.features import calculate_angle, extract_features, valid_landmarks


@pytest.mark.parametrize("a,b,c,expected", [((0, 1), (0, 0), (1, 0), 90),
                                            ((-1, 0), (0, 0), (1, 0), 180),
                                            ((1, 0), (0, 0), (2, 0), 0)])
def test_angle(a, b, c, expected):
    assert calculate_angle(a, b, c) == pytest.approx(expected)


def test_directed_angle():
    assert calculate_angle((0, 1), (0, 0), (1, 0), directed=True) == 270


@pytest.mark.parametrize("point", [None, (float("nan"), 1), (float("inf"), 0), (1,), (0, 0)])
def test_invalid_angle(point):
    assert calculate_angle(point, (0, 0), (1, 1)) is None


@pytest.mark.parametrize("points", [None, [], np.zeros((12, 3)), np.full((33, 4), np.nan), np.zeros((33, 4))])
def test_invalid_landmarks(points):
    assert not valid_landmarks(points)
    assert extract_features(points) is None


def test_feature_contract(landmarks):
    features = extract_features(landmarks)
    assert features.shape == (22,)
    p = landmarks
    assert features[0] == calculate_angle(p[11, :3], p[13, :3], p[15, :3])
    assert features[8] == pytest.approx(np.linalg.norm(p[11, :3] - p[12, :3]) / np.linalg.norm(p[11, :3] - p[23, :3]))
    assert features[20] == pytest.approx(abs(p[13, 1] - p[11, 1]) / np.linalg.norm(p[11, :3] - p[23, :3]))


def test_zero_coordinate_is_valid(landmarks):
    landmarks[11, 2] = 0
    assert extract_features(landmarks) is not None


def test_occluded_and_degenerate(landmarks):
    landmarks[11, 3] = 0.1
    assert extract_features(landmarks) is None
    landmarks[11] = landmarks[13]
    assert extract_features(landmarks) is None
