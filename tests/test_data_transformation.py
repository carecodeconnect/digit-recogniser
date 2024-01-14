import unittest
import numpy as np
import sys
sys.path.append('..')
from src.data_transformation import one_hot_encode

class TestUtils(unittest.TestCase):
    """
    class to test data transformations
    """
    def test_one_hot_encode(self):
        # Test the one_hot_enc function with specified case
        # Test: one_hot_enc(y, num_labels=4) with y = [0, 2, 1, 3, 0]
        # should be:   array([[1., 0., 0., 0.],
        #                    [0., 0., 1., 0.],
        #                    [0., 1., 0., 0.],
        #                    [0., 0., 0., 1.],
        #                    [1., 0., 0., 0.]])
        input_labels = [0, 2, 1, 3, 0]
        num_labels = 4
        expected_output = np.array([[1., 0., 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 0., 1.],
                                    [1., 0., 0., 0.]])
        
        output = one_hot_encode(input_labels, num_labels)
        np.testing.assert_array_equal(output, expected_output)

if __name__ == '__main__':
    unittest.main()
