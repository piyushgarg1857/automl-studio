import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from evaluation_core import tune_estimator


class TuningCoreTests(unittest.TestCase):
    def test_tuning_uses_pipeline_parameters_and_returns_fitted_search(self):
        X = pd.DataFrame({
            "value": np.arange(60, dtype=float),
            "category": ["a", "b", "c"] * 20,
        })
        y = pd.Series(["class_a" if i % 2 == 0 else "class_b" for i in range(60)])
        search, encoder, task = tune_estimator(
            X,
            y,
            DecisionTreeClassifier(random_state=0),
            {"max_depth": [2, 3]},
            problem_type="classification",
            cv_folds=3,
            n_iter=1,
        )
        self.assertEqual(task, "classification")
        self.assertIsNotNone(encoder)
        self.assertIn("model__max_depth", search.best_params_)
        self.assertIsNotNone(search.best_estimator_.named_steps["preprocess"])
        self.assertTrue(np.isfinite(search.best_score_))


if __name__ == "__main__":
    unittest.main()
