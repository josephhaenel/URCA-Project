import unittest
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.rf = RandomForest()  # Replace with the actual class name
        self.output_dir = "test_output"

    def tearDown(self):
        if os.path.exists(self.output_dir):
            os.rmdir(self.output_dir)

    def test_compile_and_train(self):
        # Call the method under test
        result = self.rf.compile_and_train(self.output_dir)

        # Assert that the output is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Assert that the output DataFrame has the expected columns
        expected_columns = ['Disease Type', 'Train IoU', 'Validation IoU', 'Train Accuracy', 'Validation Accuracy']
        self.assertListEqual(list(result.columns), expected_columns)

        # Assert that the output DataFrame is saved to the correct file
        expected_file_path = os.path.join(self.output_dir, f"{self.rf.dataset_name}_metrics.xlsx")
        self.assertTrue(os.path.exists(expected_file_path))

        # Assert that the output DataFrame contains the expected number of rows
        expected_num_rows = len(set(self.rf.disease_types)) + 1  # Number of unique disease types + 1 for 'All_Classes'
        self.assertEqual(len(result), expected_num_rows)

        # Assert that the accuracy scores are within the expected range
        for _, row in result.iterrows():
            self.assertGreaterEqual(row['Train Accuracy'], 0.0)
            self.assertLessEqual(row['Train Accuracy'], 1.0)
            self.assertGreaterEqual(row['Validation Accuracy'], 0.0)
            self.assertLessEqual(row['Validation Accuracy'], 1.0)

if __name__ == '__main__':
    unittest.main()