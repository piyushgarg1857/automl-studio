import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from evaluation_core import evaluate_models


class EvaluationEdgeCaseTests(unittest.TestCase):
    def test_regression_metrics_and_artifacts(self):
        X = pd.DataFrame({"x": np.arange(50, dtype=float)})
        y = pd.Series(2.5 * X["x"] + 1.0)
        leaderboard, artifacts = evaluate_models(
            X, y, {"Linear Regression": LinearRegression()},
            problem_type="regression", cv_folds=3,
        )
        row = leaderboard.iloc[0]
        self.assertIn("CV R² Mean", leaderboard.columns)
        self.assertIn("RMSE", leaderboard.columns)
        self.assertTrue(np.isfinite(row["CV R² Mean"]))
        self.assertIsNone(artifacts["Linear Regression"]["error"])
        self.assertEqual(len(artifacts["Linear Regression"]["predictions"]), 10)

    def test_missing_feature_values_are_imputed(self):
        X = pd.DataFrame({
            "number": [0.0, 1.0, np.nan, 3.0, 4.0, np.nan] * 10,
            "kind": ["a", "b", None, "a", "b", "a"] * 10,
        })
        y = pd.Series(["yes", "no"] * 30)
        leaderboard, artifacts = evaluate_models(
            X, y, {"Logistic Regression": LogisticRegression(max_iter=300)},
            problem_type="classification", cv_folds=3,
        )
        self.assertNotIn("Error", leaderboard.columns)
        self.assertIsNone(artifacts["Logistic Regression"]["error"])

    def test_rejects_mismatched_rows(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0])
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            evaluate_models(X, y, {"Linear Regression": LinearRegression()})

if __name__ == "__main__":
    unittest.main()
