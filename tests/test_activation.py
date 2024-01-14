import unittest
import sys
sys.path.append('..')
from src.activation_functions import sigmoid, softmax
import numpy as np

# class for testing Activation Functions
class ActivationTests(unittest.TestCase):
    def test_sigmoid(self):
        # Call sigmoid from the NeuralNetworkActivation class
        # Test 1: sigmoid(0.5) should be 0.6224593312018546
        result = sigmoid(0.5)
        expected = 0.6224593312018546
        self.assertAlmostEqual(result, expected, places=6)

        # Test 2: sigmoid([-1, 0, 1], derivative=True) should be [0.26894142, 0.5, 0.73105858]
        result_array = sigmoid(np.array([-1, 0, 1]))
        expected_array = np.array([0.26894142, 0.5, 0.73105858])
        np.testing.assert_almost_equal(result_array, expected_array, decimal=6)

    def test_softmax(self):
        # Call softmax from the NeuralNetworkActivation class
        # Test 1: softmax(np.array([[2.0, 1.0, 0.1]])) should be [[0.65900114 0.24243297 0.09856589]]
        result = softmax(np.array([[2.0, 1.0, 0.1]]))
        expected = np.array([[0.65900114, 0.24243297, 0.09856589]])
        np.testing.assert_almost_equal(result, expected, decimal=6)

        # Test 2: softmax(np.array([[4.0, 0.5, 0.1]]) should be [[0.95198267 0.02874739 0.01926995]}
        result_2 = softmax(np.array([[4.0, 0.5, 0.1]]))
        expected_2 = np.array([[0.95198267, 0.02874739, 0.01926995]])
        np.testing.assert_almost_equal(result_2, expected_2, decimal=6)

if __name__ == '__main__':
    unittest.main()
