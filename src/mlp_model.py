import numpy as np
from .activation_functions import sigmoid, softmax

class MLPModel:
    """
    class with MLP architecture, initialisation and training
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.W1, self.W2, self.W3, self.b1, self.b2, self.b3 = self.initialise_wb(input_size, hidden1_size, hidden2_size, output_size)

    def initialise_wb(self, input_size, hidden1_size, hidden2_size, output_size):
        """
        Initialize weights and biases for a neural network.

        Parameters:
        - input_size (int): Number of input features.
        - hidden1_size (int): Number of neurons in the first hidden layer.
        - hidden2_size (int): Number of neurons in the second hidden layer.
        - output_size (int): Number of output neurons (number of classes).

        Returns:
        - W1, W2, W3 (np.ndarray): 2D arrays for the weights with random values from a standard normal
            distribution
        - b1, b2, b3: (np.ndarray): 2D arrays for the bias as zeros
        
        Example usage:
        W1, W2, W3, b1, b2, b3 = initialize_wb(input_size, hidden1_size, hidden2_size, output_size)
        """
        # your code here

        # Initialize weights with random values from a standard normal distribution
        
        # np.random.rand() returns samples from standard normal distributin
        # with mean = 0 and standard deviation = 1
        # .rand() does not allow you to specify mean and standard deviation
        # in this case, np.random.normal the mean is 0 and standard deviation 1
        # so the result is the same
        
        # .normal(mean, standard_deviation)
        #W1 = np.random.normal(0, 1, [input_size, hidden1_size])
        W1 = np.random.randn(input_size, hidden1_size) # size input x hidden1

        #W2 = np.random.normal(0, 1, [hidden1_size, hidden2_size])

        W2 = np.random.randn(hidden1_size, hidden2_size) # size hidden1 x hidden2

        #W3 = np.random.normal(0, 1, [hidden2_size, output_size])

        W3 = np.random.randn(hidden2_size, output_size) # size hidden2 x output
        
        # Initialize biases as zeros
        b1 = np.zeros((1, hidden1_size)) # size 1 x hidden1
        b2 = np.zeros((1, hidden2_size)) # size 1 x hidden2
        b3 = np.zeros((1, output_size)) # size 1 x output
        
        return W1, W2, W3, b1, b2, b3

    def forward_pass(self, X):
        """
        Performs the forward pass using instance's weights and biases.
        """
        z1 = np.dot(X, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = softmax(z3)

        return a1, a2, a3
    
    def backpropagation(self, X, y, a1, a2, a3, W2, W3):
        """
        Performs backpropagation: Computes the gradients of the loss function with respect to the weights
        """
        delta = a3 - y
        dW3 = np.dot(a2.T, delta)
        db3 = np.sum(delta, axis=0, keepdims=True)
        delta = np.dot(delta, W3.T) * (a2 * (1 - a2))
        dW2 = np.dot(a1.T, delta)
        db2 = np.sum(delta, axis=0, keepdims=True)
        delta = np.dot(delta, W2.T) * (a1 * (1 - a1))
        dW1 = np.dot(X.T, delta)
        db1 = np.sum(delta, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        """
        Updates instance's weights and biases based on the computed gradients.
        """
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
    
    def calculate_accuracy(self, X, y):
        """
        Calculates accuracy using the instance's method and weights.
        """
        _, _, a3 = self.forward_pass(X)
        predictions = np.argmax(a3, axis=1)
        accuracy = np.mean(predictions == y)

        return accuracy
    
    def predict(self, X):
        """
        Make predictions for the given input data X.

        Parameters:
        - X (numpy.ndarray): Input data for which predictions are to be made.

        Returns:
        - predictions (numpy.ndarray): The predicted class labels.
        """
        # Conduct the forward pass. We don't need the activations from the first two layers here.
        _, _, a3 = self.forward_pass(X)
        
        # Convert the output layer activations to class label predictions
        predictions = np.argmax(a3, axis=1)
        return predictions

if __name__ == "__main__":
    main()