"""Compute classification metrics from real labeled predictions."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, labels):
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if (truth.ndim != 1 or predicted.shape != truth.shape or not len(truth)
            or len(set(labels)) != len(labels) or not labels):
        raise ValueError("Expected nonempty aligned 1D labels and a unique class list")
    if not set(truth).issubset(labels) or not set(predicted).issubset(labels):
        raise ValueError("Unknown labels in evaluation data")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0)
    report = {"samples": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
              "labels": list(labels), "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
              "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                     "f1": float(f1[i]), "support": int(support[i])}
                            for i, label in enumerate(labels)}}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(truth, predicted, labels=labels,
                                                   average=average, zero_division=0)
        report[average] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    return report
