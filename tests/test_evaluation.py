import numpy as np
import pytest
from gym_tracker.evaluation import classification_metrics
from scripts.evaluate import load_dataset


def test_known_confusion_matrix():
    # Deliberately imperfect synthetic labels test arithmetic, not model quality.
    report = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert report["per_class"]["a"]["precision"] == 1
    assert report["per_class"]["a"]["recall"] == 0.5
    assert report["per_class"]["a"]["f1"] == pytest.approx(2 / 3)
    assert report["macro"]["f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_unknown_empty_and_mismatched_labels():
    for truth, predicted in [([], []), (["x"], ["a"]), (["a"], [])]:
        with pytest.raises(ValueError):
            classification_metrics(truth, predicted, ["a"])


def test_dataset_contract(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, X=np.ones((2, 30, 22)), y=np.array(["squat", "push_up"]))
    x, y = load_dataset(path)
    assert x.shape == (2, 30, 22) and len(y) == 2
    np.savez(path, X=np.ones((2, 30, 22)), y=np.array(["invented", "squat"]))
    with pytest.raises(ValueError):
        load_dataset(path)
