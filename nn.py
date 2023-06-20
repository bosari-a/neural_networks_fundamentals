import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pkg.Layers import Activation, Dense
from pkg.loss import mse, mse_prime


def tanh_prime(a):
    return 1 - np.tanh(a)**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, tanh_prime)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X = X.reshape((4, 2, 1))
Y_true = np.array([0, 1, 1, 0])
Y_true = Y_true.reshape((4, 1, 1))

network = [
    Dense(X.shape[1], 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
#
epochs = 1000
learning_rate = 0.1
loss_vs_epoch = []
#
for epoch in tqdm(range(epochs)):
    E = 0
    # forward pass ->
    for x, y in zip(X, Y_true):
        Y_pred = x
        for layer in network:
            Y_pred = layer.forward(Y_pred)

        E += mse(Y_pred, y)
    # backward pass <-
        output_gradient = mse_prime(Y_pred, y)
        for i in range(len(network)-1, -1, -1):
            layer = network[i]
            output_gradient = layer.backward(output_gradient, learning_rate)
    E /= X.shape[0]
    loss_vs_epoch.append(E)

plt.plot(loss_vs_epoch)
plt.xlabel("Error (loss)")
plt.ylabel("Epoch")
plt.show()
