import unittest
import pandas as pd
from src.feature.feature_engineering import create_features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "LoanAmount": [1000, 2000, 3000],
            "Income": [5000, 6000, 8000],
            "LoanTerm": [12, 24, 36]
        })

    def test_create_features_adds_new_columns(self):
        result = create_features(self.df.copy())
        self.assertIn("LoanAmount/Income", result.columns)
        self.assertIn("LoanAmount/LoanTerm", result.columns)
        self.assertEqual(len(result), len(self.df))

if __name__ == "__main__":
    unittest.main()
