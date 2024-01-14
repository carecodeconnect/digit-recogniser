# import modules
import numpy as np
from src.data_processor import DigitDataProcessor
from src.data_transformation import one_hot_encode, one_hot_decode
from src.visualiser import DataVisualiser
from src.mlp_model import MLPModel
from src.trainer import Trainer

# Process data: load, normalise, and flatten images 
X_train, y_train, X_test, y_test = DigitDataProcessor.process_data()

# Load and visualise data
data_viz = DataVisualiser()
data_viz.plot_random_samples()  # default 4x4 grid

# One-hot encode labels
num_class_labels = 10
y_train_onehot = one_hot_encode(y_train, num_class_labels)

# Define MLP architecture and create an instance of MLPModel
input_size = X_train.shape[1]
hidden1_size = 128
hidden2_size = 64
output_size = num_class_labels
mlp_model = MLPModel(input_size, hidden1_size, hidden2_size, output_size)

# Train the model
learning_rate = 0.1
epochs = 10
batch_size = 64
nn_trainer = Trainer()
nn_trainer.train_model(mlp_model, X_train, y_train, y_train_onehot, X_test, y_test, learning_rate, epochs, batch_size)

# Visualise predictions
data_viz.plot_digit_examples()  # 10 digit examples
data_viz.plot_specific_image(1)  # Display the second image in the dataset

# Make predictions on the test dataset
y_pred = mlp_model.predict(X_test)

# Evaluate accuracy on test
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# If needed, further processing like visualization or comparison can be done here


