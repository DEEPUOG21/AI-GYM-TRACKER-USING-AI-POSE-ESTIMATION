import importlib.util
import numpy as np
import pytest

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None,
                                 reason="Full ML runtime not installed")]


def test_bundled_model_inference(landmarks):
    from gym_tracker.classification import get_classifier
    from gym_tracker.features import extract_features
    from gym_tracker.repetitions import EXERCISES
    instance = get_classifier()
    result = instance.predict(np.stack([extract_features(landmarks)] * 30))
    assert result.exercise in EXERCISES
    assert 0 <= result.confidence <= 1
    assert instance is get_classifier()


def test_real_pose_detector():
    from gym_tracker.pose import PoseDetector
    detector = PoseDetector()
    try:
        assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None
    finally:
        detector.close()
        detector.close()
