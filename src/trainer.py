class Trainer:
    """
    Class to train the model
    """

    def train_model(self, model, X_train, y_train, y_train_onehot, X_test, y_test, learning_rate, epochs, batch_size):
        """
        Trains the model.
        """
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = y_train_onehot[i:i + batch_size]

                # Forward pass
                a1, a2, a3 = model.forward_pass(x_batch)
                
                # Backpropagation
                dW1, db1, dW2, db2, dW3, db3 = model.backpropagation(x_batch, y_batch, a1, a2, a3, model.W2, model.W3)
                
                # Update weights
                model.update_weights(dW1, db1, dW2, db2, dW3, db3, learning_rate)
            
            # Calculate accuracy on the training and test sets
            train_accuracy = model.calculate_accuracy(X_train, y_train)
            test_accuracy = model.calculate_accuracy(X_test, y_test)

            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

