"""Validated BiLSTM inference with an atomic, thread-safe resource cache."""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import numpy as np

WINDOW_SIZE = 30
FEATURE_COUNT = 22
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
LABELS = {"push-up": "push_up", "squat": "squat",
          "barbell biceps curl": "bicep_curl", "shoulder press": "shoulder_press"}


@dataclass(frozen=True)
class Prediction:
    exercise: str
    confidence: float
    probabilities: dict


class ExerciseClassifier:
    def __init__(self, model, scaler, encoder):
        self.model, self.scaler = model, scaler
        self.labels = tuple(str(x) for x in encoder.classes_)
        self._lock = Lock()
        if len(set(self.labels)) != len(self.labels) or set(self.labels) != set(LABELS):
            raise ValueError(f"Unsupported label encoder classes: {self.labels}")
        if tuple(model.input_shape[1:]) != (WINDOW_SIZE, FEATURE_COUNT):
            raise ValueError(f"Unexpected model input shape: {model.input_shape}")
        if model.output_shape[-1] != len(self.labels):
            raise ValueError("Model and label encoder output sizes disagree")
        if scaler.n_features_in_ != WINDOW_SIZE * FEATURE_COUNT:
            raise ValueError("Scaler must accept 660 features")

    def predict(self, features):
        window = np.asarray(features, dtype=float)
        if window.shape != (WINDOW_SIZE, FEATURE_COUNT) or not np.isfinite(window).all():
            raise ValueError("Expected a finite (30, 22) feature window")
        with self._lock:
            scaled = np.asarray(self.scaler.transform(window.reshape(1, -1)))
            if scaled.shape != (1, 660) or not np.isfinite(scaled).all():
                raise ValueError("Scaler produced invalid features")
            # Direct call avoids creating a tf.data thread pool for every prediction.
            output = np.asarray(self.model(scaled.reshape(1, 30, 22), training=False))
        if (output.shape != (1, len(self.labels)) or not np.isfinite(output).all()
                or np.any(output < 0) or np.any(output > 1)
                or not np.isclose(output.sum(), 1, atol=1e-4)):
            raise ValueError("Classifier must return one normalized probability vector")
        index = int(output[0].argmax())
        return Prediction(LABELS[self.labels[index]], float(output[0, index]),
                          {LABELS[label]: float(output[0, i]) for i, label in enumerate(self.labels)})


_cache = {}
_load_lock = Lock()


def get_classifier(model_dir=MODEL_DIR):
    """Load all artifacts once; a failed load leaves the cache retryable.

    Only load trusted bundled artifacts: pickle/joblib can execute code.
    """
    directory = Path(model_dir).resolve()
    with _load_lock:
        if directory not in _cache:
            import joblib
            from tensorflow.keras.models import load_model
            model = load_model(directory / "exercise_bilstm.h5", compile=False)
            scaler = joblib.load(directory / "feature_scaler.pkl")
            encoder = joblib.load(directory / "label_encoder.pkl")
            _cache[directory] = ExerciseClassifier(model, scaler, encoder)
        return _cache[directory]
