import numpy as np
# Base Layer


class Layer:
    def __init__(self):
        self.input_val = None
        self.output = None

    def forward(self):
        # todo: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # todo: update parameters, return input_val_gradient
        pass

# Dense Layer


class Dense(Layer):
    def __init__(self, input_rows, output_rows):
        super().__init__()
        self.input_rows = input_rows
        self.output_rows = output_rows
        self.weights = np.random.randn(output_rows, input_rows)
        self.bias = np.random.randn(output_rows, 1)

    def forward(self, input_val):
        self.input_val = input_val
        self.output = np.add(
            np.matmul(self.weights, input_val), self.bias)
        return self.output

    def backward(self, output_gradient, learning_rate):
        dE_dW = np.matmul(output_gradient, self.input_val.T)
        dE_dX = np.matmul(self.weights.T, output_gradient)
        self.weights = np.subtract(
            self.weights, np.multiply(learning_rate, dE_dW))
        self.bias -= np.multiply(learning_rate, output_gradient)
        return dE_dX

# Activation Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation_prime = activation_prime
        self.activation = activation

    def forward(self, input_val):
        self.output = self.activation(input_val)
        return self.output

    def backward(self, input_val, output_gradient):
        # todo: return derivative of error w.r.t input_val
        return np.multiply(output_gradient, self.activation_prime(input_val))
