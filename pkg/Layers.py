import numpy as np
# Base Layer


class Layer:
    def __init__(self, input):
        self.input = input
        self.output = None

    def forward(self):
        # todo: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # todo: update parameters, return input_gradient
        pass

# Dense Layer


class Dense(Layer):
    def __init__(self, input, output_size):
        self.output_size = output_size
        super().__init__(input)
        self.weights = np.random.randn(output_size, np.shape(input)[0])
        self.bias = np.random.randn(output_size, 1)

    def forward(self):
        return np.add(np.matmul(self.weights, self.input), self.bias)

    def backward(self, output_gradient, learning_rate):
        dE_dW = np.matmul(output_gradient, self.input.T)
        self.weights = np.subtract(
            self.weights, np.multiply(learning_rate, dE_dW))
        self.bias -= np.multiply(learning_rate, output_gradient)
        return np.matmul(self.weights.T, output_gradient)
