import unittest
from src.data.data_import import load_data
from src.feature.preprocess import preprocess_data
from src.feature.feature_engineering import create_features
from src.model.train import train_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TestFullPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        # Load dataset
        df = load_data()
        self.assertFalse(df.empty)

        # Preprocess and feature engineer
        df_clean = preprocess_data(df)
        df_feat = create_features(df_clean)

        # Split target
        target_col = "Default"  # adjust if your dataset has a different target column
        X = df_feat.drop(target_col, axis=1)
        y = df_feat[target_col]

        # Train model
        model, X_train, X_test, y_train, y_test = train_model(X, y, cv=3)
        self.assertTrue(hasattr(model, "predict"))

if __name__ == "__main__":
    unittest.main()
