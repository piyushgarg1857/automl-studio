import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_bundle import ModelBundle


class ModelBundleTests(unittest.TestCase):
    def test_predict_decodes_original_labels(self):
        X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
        original_y = np.array(["cat", "dog", "cat", "dog", "cat", "dog"])
        encoder = LabelEncoder()
        y = encoder.fit_transform(original_y)
        pipeline = Pipeline([
            ("preprocess", StandardScaler()),
            ("model", LogisticRegression(random_state=0)),
        ])
        pipeline.fit(X, y)
        bundle = ModelBundle(pipeline, encoder)
        predictions = bundle.predict(X)
        self.assertTrue(set(predictions).issubset({"cat", "dog"}))

    def test_pickle_round_trip_preserves_decoded_predictions(self):
        X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
        encoder = LabelEncoder()
        y = encoder.fit_transform(["low", "high", "low", "high", "low", "high"])
        pipeline = Pipeline([
            ("preprocess", StandardScaler()),
            ("model", LogisticRegression(random_state=0)),
        ]).fit(X, y)
        bundle = ModelBundle(pipeline, encoder)
        restored = pickle.loads(pickle.dumps(bundle))
        np.testing.assert_array_equal(bundle.predict(X), restored.predict(X))


if __name__ == "__main__":
    unittest.main()
