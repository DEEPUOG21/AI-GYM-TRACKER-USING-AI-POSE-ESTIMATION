from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import pytest
from gym_tracker.classification import ExerciseClassifier, LABELS, get_classifier, _cache


class Model:
    input_shape = (None, 30, 22)
    output_shape = (None, 4)

    def __init__(self, output=None):
        self.output = output if output is not None else [[0.1, 0.6, 0.2, 0.1]]

    def __call__(self, x, training=False):
        assert x.shape == (1, 30, 22)
        assert training is False
        return self.output


class Scaler:
    n_features_in_ = 660

    def transform(self, x):
        assert x.shape == (1, 660)
        self.last = x.copy()
        return x


def classifier(output=None):
    return ExerciseClassifier(Model(output), Scaler(), SimpleNamespace(classes_=list(LABELS)))


def test_classification_output_and_order():
    instance = classifier()
    x = np.arange(660).reshape(30, 22)
    result = instance.predict(x)
    assert result.exercise == "squat"
    assert result.confidence == pytest.approx(0.6)
    assert sum(result.probabilities.values()) == pytest.approx(1)
    np.testing.assert_array_equal(instance.scaler.last, x.reshape(1, 660))


@pytest.mark.parametrize("output", [[[0.1, 0.9]], [[0.1, 0.2, 0.3, np.nan]],
                                      [[-0.1, 0.5, 0.3, 0.3]], [[0.2] * 4]])
def test_invalid_output(output):
    with pytest.raises(ValueError):
        classifier(output).predict(np.ones((30, 22)))


@pytest.mark.parametrize("x", [np.ones((29, 22)), np.full((30, 22), np.nan)])
def test_invalid_window(x):
    with pytest.raises(ValueError):
        classifier().predict(x)


def test_atomic_cache_and_retry(monkeypatch, tmp_path):
    import joblib
    calls = []

    def load(path, compile):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("transient load failure")
        return Model()

    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", SimpleNamespace(load_model=load))
    monkeypatch.setattr(joblib, "load", lambda path: Scaler() if path.name == "feature_scaler.pkl"
                        else SimpleNamespace(classes_=list(LABELS)))
    with pytest.raises(OSError):
        get_classifier(tmp_path)
    assert tmp_path not in _cache
    with ThreadPoolExecutor(max_workers=4) as pool:
        instances = list(pool.map(lambda _: get_classifier(tmp_path), range(8)))
    assert all(item is instances[0] for item in instances)
    assert len(calls) == 2
    _cache.pop(tmp_path)
