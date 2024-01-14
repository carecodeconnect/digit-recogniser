# digit-recogniser: deep learning neural network for digit recognition

# Overview

`digit-recogniser` is a deep learning neural network which recognises handwritten digits. It develops a Multi-Level Perceptron (MLP) model for handwritten digit recognition using the MNIST dataset, implemented entirely using Python's `NumPy` library. The project emphasizes Object-Oriented Programming (OOP) for a modular and clear code structure.

### Multi-Level Perceptron (MLP)
An MLP is a type of artificial neural network featuring multiple layers of nodes in a directed graph. Each node, except for the input nodes, is a neuron typically using a non-linear activation function. MLPs are ideal for complex pattern recognition tasks like image recognition.

### MNIST Dataset
The MNIST dataset, a staple in machine learning, comprises 70,000 images of handwritten digits (0-9), each a 28x28 pixel grayscale image. It's commonly used for training and testing in machine learning.

### NumPy and Neural Networks
The entire neural network in this project is built using `NumPy`, a fundamental package for scientific computing in Python. NumPy's powerful $n$-dimensional array objects and broad collection of mathematical functions allow efficient and effective handling of operations required for neural network computation.

### Activation Functions: Sigmoid and Softmax
- **Sigmoid Function:** Used in hidden layers of the MLP, the sigmoid function maps input values to a range between 0 and 1, providing a smooth gradient and helping with non-linearities in the model.
- **Softmax Function:** Employed in the output layer, the softmax function converts the output logits to probabilities, essential for multi-class classification tasks like digit recognition.

### Object-Oriented Programming (OOP)
OOP is leveraged to structure the code into classes and objects, encapsulating data and functionality. This approach enhances modularity, scalability, and maintainability of the code.

## Implementation Details

1. **Data Processing**
   - `DigitDataProcessor` handles loading, normalizing, and flattening of MNIST images.

2. **Data Visualization**
   - `DataVisualiser` for plotting samples and predictions.

3. **One-Hot Encoding**
   - `one_hot_encode` for converting labels to a format suitable for MLP.

4. **MLP Model**
   - `MLPModel` defines the MLP architecture with sigmoid activation in hidden layers and softmax in the output layer.

5. **Training**
   - `Trainer` manages the training process, leveraging NumPy for efficient computation.

6. **Evaluation**
   - Model's accuracy assessed on the test dataset.

# Usage

Clone the repository: 

```
# with HTTPS
git clone https://github.com/carecodeconnect/digit-recogniser.git
# with SSH 
git clone git@github.com:carecodeconnect/digit-recogniser.git
```

Execute with `python main.py`

# Testing

For unit tests, run `/tests/test.py`.

# Files & Folders
```
├── data
├── docs
│   ├── _build
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   ├── _static
│   └── _templates
├── environment.yml
├── main.py
├── notebooks
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── activation_functions.py
│   ├── data_processor.py
│   ├── data_transformation.py
│   ├── __init__.py
│   ├── mlp_model.py
│   ├── trainer.py
│   └── visualiser.py
├── tests
│   ├── environment.py
│   ├── test_activation.py
│   ├── test_data_transformation.py
│   └── test.py
└── tox.ini

9 directories, 21 files
```