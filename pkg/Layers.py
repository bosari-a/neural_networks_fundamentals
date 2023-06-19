import numpy as np
# Base Layer


class Layer:
    def __init__(self, input_val):
        self.input_val = input_val
        self.output = None

    def forward(self):
        # todo: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # todo: update parameters, return input_val_gradient
        pass

# Dense Layer


class Dense(Layer):
    def __init__(self, input_val, output_size):
        self.output_size = output_size
        super().__init__(input_val)
        self.weights = np.random.randn(output_size, np.shape(input_val)[0])
        self.bias = np.random.randn(output_size, 1)

    def forward(self):
        self.output = np.add(
            np.matmul(self.weights, self.input_val), self.bias)
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
    def __init__(self, input_val, activation, activation_prime, output_gradient):
        super().__init__(input_val)
        self.activation_prime = activation_prime
        self.activation = activation
        self.output_gradient = output_gradient

    def forward(self):
        self.output = self.activation(self.input_val)
        return self.output

    def backward(self):
        # todo: return derivative of error w.r.t input_val
        return np.multiply(self.output_gradient, self.activation_prime(self.input_val))
