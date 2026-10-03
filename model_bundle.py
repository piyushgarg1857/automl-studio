"""Serializable prediction wrapper that restores original target labels."""
from __future__ import annotations

import numpy as np


class ModelBundle:
    """Wrap a fitted pipeline and optional target LabelEncoder.

    For classification, predict() returns original target labels. For regression,
    predictions pass through unchanged. The wrapper is pickle-friendly.
    """

    def __init__(self, pipeline, label_encoder=None):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    def predict(self, X):
        predictions = np.asarray(self.pipeline.predict(X))
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions.astype(int))
        return predictions

    def predict_proba(self, X):
        """Expose class probabilities when the underlying estimator supports them."""
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self):
        if self.label_encoder is not None:
            return self.label_encoder.classes_
        estimator = self.pipeline.named_steps.get("model")
        return getattr(estimator, "classes_", None)
