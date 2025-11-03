import unittest
import pandas as pd
from src.data.data_import import load_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TestDataImport(unittest.TestCase):
    def test_load_data_returns_dataframe(self):
        df = load_data()  # Uses default path
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertGreater(len(df.columns), 0)

if __name__ == "__main__":
    unittest.main()
