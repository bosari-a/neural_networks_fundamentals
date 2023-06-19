import numpy as np
from pkg.Layers import Activation


def tanh_prime(x):
    return 1 - np.tanh(x)**2


class Tanh(Activation):
    def __init__(self, input, output_gradient):
        super().__init__(input, np.tanh, tanh_prime, output_gradient)
