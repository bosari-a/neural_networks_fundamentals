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
        self.input_val = input_val.reshape((self.input_rows, 1))
        self.output = np.add(
            np.dot(self.weights, input_val), self.bias)
        self.output = self.output.reshape((self.output_rows, 1))
        return self.output

    def backward(self, output_gradient, learning_rate):
        dE_dW = np.dot(output_gradient, self.input_val.T)
        dE_dW = dE_dW.reshape((self.output_rows, self.input_rows))
        dE_dX = np.dot(self.weights.T, output_gradient)
        dE_dX = dE_dX.reshape((self.input_rows, 1))
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
        self.input_val = input_val
        self.output = self.activation(input_val)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # todo: return derivative of error w.r.t input_val
        output_grad = np.multiply(
            output_gradient, self.activation_prime(self.input_val))
        output_grad = output_grad.reshape((self.input_val.shape[0], 1))
        return output_grad
