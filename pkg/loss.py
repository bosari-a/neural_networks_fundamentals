import numpy as np


def mse(y_nn, y_true):
    return np.mean(np.power(y_true-y_nn, 2))


def mse_prime(y_nn, y_true):
    return np.array(np.multiply(2, np.mean(y_nn-y_true)))
