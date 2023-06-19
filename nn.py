from pprint import pprint
import numpy as np
from pkg.Layers import Activation, Dense
import tqdm


def tanh_prime(x):
    return 1 - np.tanh(x)**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, tanh_prime)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X = X.reshape((4, 2, 1))


network = [
    Dense(X.shape[1], 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
