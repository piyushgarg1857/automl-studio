import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from evaluation_core import detect_problem_type, evaluate_models


class EvaluationCoreTests(unittest.TestCase):
    def test_detects_numeric_continuous_target_as_regression(self):
        target = pd.Series(np.linspace(0, 10, 30))
        self.assertEqual(detect_problem_type(target), "regression")

    def test_allows_explicit_task_override(self):
        target = pd.Series([0, 1, 2, 3])
        self.assertEqual(detect_problem_type(target, task="regression"), "regression")

    def test_evaluation_handles_mixed_features_and_returns_artifacts(self):
        frame = pd.DataFrame(
            {
                "amount": np.arange(60, dtype=float),
                "segment": ["north", "south", "west"] * 20,
            }
        )
        target = pd.Series(["yes" if i % 2 else "no" for i in range(60)])
        leaderboard, artifacts = evaluate_models(
            frame,
            target,
            {"Logistic Regression": LogisticRegression(max_iter=300)},
            problem_type="classification",
            cv_folds=3,
        )
        self.assertIn("CV F1 Mean", leaderboard.columns)
        self.assertIsNone(artifacts["Logistic Regression"]["error"])
        self.assertEqual(len(artifacts["Logistic Regression"]["predictions"]), 12)


if __name__ == "__main__":
    unittest.main()
