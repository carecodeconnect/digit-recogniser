import mnist

class DigitDataProcessor:
    @staticmethod
    def load_data():
        """
        Load and return the MNIST dataset for training and testing.

        Returns:
        - X_train (numpy.ndarray): Training set, a 3D array of shape (num_samples, 28, 28),
          where num_samples is the number of training samples.
        - y_train (numpy.ndarray): Training labels, a 1D array of shape (num_samples,) containing
          the corresponding class labels for the training images.
        - X_test (numpy.ndarray): Test set, a 3D array of shape (num_samples, 28, 28),
          where num_samples is the number of test samples.
        - y_test (numpy.ndarray): Test labels, a 1D array of shape (num_samples,) containing
          the corresponding class labels for the test images.

        Example usage:
        >>> X_train, y_train, X_test, y_test = data_loader()
        >>> print("Training set shape:", X_train.shape)
        >>> print("Training labels shape:", y_train.shape)
        >>> print("Test set shape:", X_test.shape)
        >>> print("Test labels shape:", y_test.shape)
        """
        X_train = mnist.train_images()
        y_train = mnist.train_labels()
        
        X_test = mnist.test_images()
        y_test = mnist.test_labels()

        return X_train, y_train, X_test, y_test

    @staticmethod
    def normalise_data(X_train, X_test):
        # Normalisation
        # Divide every single element in X_train and X_test by 255.0

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return X_train, X_test
    
    @staticmethod
    def flatten_images(X_train, X_test):
        # Flatten the shape of X_train and X_test from 3D to 2D by keeping the first dimension unchanged**

        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        return X_train, X_test

    @staticmethod
    def process_data():
        """
        Process the data by performing all necessary steps: loading, normalizing, and flattening.
        
        Returns:
        - X_train (numpy.ndarray): Processed training set.
        - y_train (numpy.ndarray): Training labels.
        - X_test (numpy.ndarray): Processed test set.
        - y_test (numpy.ndarray): Test labels.
        """
        # Load data
        X_train, y_train, X_test, y_test = DigitDataProcessor.load_data()

        # Normalize data
        X_train, X_test = DigitDataProcessor.normalise_data(X_train, X_test)

        # Flatten images
        X_train, X_test = DigitDataProcessor.flatten_images(X_train, X_test)

        return X_train, y_train, X_test, y_test

