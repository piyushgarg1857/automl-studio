import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from evaluation_core import evaluate_models


class AlwaysFailsClassifier(ClassifierMixin, BaseEstimator):
    """A cloneable estimator used to verify per-model failure reporting."""

    def fit(self, X, y):
        raise RuntimeError("intentional test failure")

    def predict(self, X):
        raise AssertionError("predict should not run after fit fails")


class EvaluationRobustnessTests(unittest.TestCase):
    def test_multiclass_classification_preserves_original_labels(self):
        X = pd.DataFrame({
            "signal": np.tile(np.arange(30), 3),
            "group": np.repeat(["g1", "g2", "g3"], 30),
        })
        y = pd.Series(np.repeat(["bronze", "silver", "gold"], 30))
        leaderboard, artifacts = evaluate_models(
            X,
            y,
            {"Logistic Regression": LogisticRegression(max_iter=500)},
            problem_type="classification",
            cv_folds=3,
        )

        artifact = artifacts["Logistic Regression"]
        self.assertNotIn("Error", leaderboard.columns)
        self.assertIsNone(artifact["error"])
        self.assertEqual(set(np.unique(artifact["y_test"])), {"bronze", "silver", "gold"})
        self.assertTrue(set(np.unique(artifact["predictions"])).issubset(
            {"bronze", "silver", "gold"}
        ))

    def test_model_failure_is_captured_without_crashing_evaluation(self):
        X = pd.DataFrame({"feature": np.arange(60, dtype=float)})
        y = pd.Series(["yes", "no"] * 30)
        leaderboard, artifacts = evaluate_models(
            X,
            y,
            {"Broken Model": AlwaysFailsClassifier()},
            problem_type="classification",
            cv_folds=3,
        )

        self.assertIn("Error", leaderboard.columns)
        self.assertIn("RuntimeError", artifacts["Broken Model"]["error"])
        self.assertIsNone(artifacts["Broken Model"]["pipeline"])


if __name__ == "__main__":
    unittest.main()
