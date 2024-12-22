# Multi-Layer Perceptron (MLP) Implementation
This repository contains an implementation of a simple Multi-Layer Perceptron (MLP) in Python. The MLP is a 
feedforward artificial neural network with a single hidden layer, designed to demonstrate the basics of neural 
network operations, including forward propagation, backpropagation, and weight adjustments during training.

## Features
- Customizable architecture with adjustable input, hidden, and output neurons.
- Implements sigmoid activation function and its derivative.
- Supports backpropagation for training using gradient descent.
- Tracks and visualizes the error across training iterations.
- Random initialization of weights with reproducible results using a random seed.

## Requiremts
- `numpy`: For numerical computations
- `matplotlib`: For plotting error trends
- `scikit-learn`: For utilities such as random state validations

You can install these dependencies using pip:

`pip install numpy matplotlib scikit-learn`

## Usage
The main class in the implementation is `MLP`. It provides methods to define, train, and evaluate the neural network.

### 1. Initialization
You can initialize the MLP with the desired number of neurons in the input, hidden, and output layers. The weights are either provided manually or initialized randomly.

```python
nn = MLP(n_input_neurons=3, n_hidden_neurons=4, n_output_neurons=2, eta=0.03, n_iterations=40000, random_state=42)
```

### 2. Training
The `fit` method trains the network using the provided input (`X`) and output (`Y`) data.

```python
X = np.array([[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 0]])
Y = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

nn.fit(X, Y)
```

### 3. Prediction
Use the `predict` method to get the output for new input data:

```python
predictions = nn.predict([1.0, 0, 1.0])
print(predictions)
```

### 4. Visualisazion
The plot method generates a graph of the error trend over the training iterations:

```python
nn.plot()
```

## Example
Here is a complete example:

```python
import numpy as np
from mlp import MLP

# Define training data
X = np.array([[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 0]])
Y = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

# Initialize the MLP
nn = MLP(n_input_neurons=3, n_hidden_neurons=4, n_output_neurons=2, eta=0.03, n_iterations=40000, random_state=42)

# Train the MLP
nn.fit(X, Y)

# Print network structure and predictions
nn.print()
for x, y in zip(X, Y):
    print(f'Input: {x}, Expected: {y}, Predicted: {nn.predict(x)}')

# Plot error trend
nn.plot()
```

## Code Overview
### `MLP` Class
The `MLP` class is the core of this implementation. Below are its key components:

- #### Initialisazion (`__init__`):
  - Sets up network architecture
  - Initializes weights and biases
  - Configures learning parameters (e.g., learning rate, iterations)

- #### Forward Propagation (`predict`):
  - Computes activations for hidden and output layers using sigmoid activation

- #### Backpropagation (`fit`):
  - Calculates deltas for output and hidden layers
  - Adjusts weights using the gradient descent algorithm

- #### Visualisazion (`plot`):
  - Plots the error across iterations to evaluate training performance

- #### Debugging (`print`):
  - Prints the network's internal state for debugging purposes
 
## Limitations
- The implementation supports only one hidden layer
- Activation functions are limited to sigmoid
- Not optimized for large-scale or complex datasets

## Future Improvements
- Add support for multiple hidden layers.
- Implement additional activation functions (e.g., ReLU, tanh).
- Optimize training with advanced techniques like momentum or Adam optimizer.
- Extend visualization capabilities.

## Licence
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the terms of the license.
