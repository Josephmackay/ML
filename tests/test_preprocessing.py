import unittest
import pandas as pd
from src.feature.preprocess import preprocess_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "LoanID": [1, 2],
            "HasMortgage": ["Yes", "No"],
            "HasDependents": ["No", "Yes"],
            "HasCoSigner": ["Yes", "No"],
            "Education": ["Graduate", "Undergraduate"],
            "EmploymentType": ["Salaried", "Self-Employed"],
            "MaritalStatus": ["Married", "Single"],
            "LoanPurpose": ["Home", "Car"],
            "LoanAmount": [2000, 3000]
        })

    def test_preprocessing_encodes_and_drops(self):
        processed = preprocess_data(self.df.copy())
        # LoanID should be dropped
        self.assertNotIn("LoanID", processed.columns)
        # Binary encoding should convert to 0/1
        self.assertTrue(all(processed["HasMortgage"].isin([0, 1])))
        # One-hot encoding should add columns
        self.assertTrue(any("Education_" in col for col in processed.columns))

if __name__ == "__main__":
    unittest.main()
