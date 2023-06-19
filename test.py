import numpy as np
from pprint import pprint


class Vehicles:
    def __init__(self, type, brand):
        self.type = type
        self.brand = brand

    def describe(self):
        return f"Type: {self.type}\nBrand: {self.brand}"


class Cars(Vehicles):
    def __init__(self, brand):
        super().__init__("Car", brand)

    def describe(self):
        return f"MODIFIED\nType: {self.type}\nBrand: {self.brand}"


audi = Cars("Audi")

# print(audi.describe())

X = [1, 2, 3]
X = np.array(X)
X = X.reshape((3, 1))

B = [4, 5, 6, 7]
B = np.array(B)
B = B.reshape((4, 1))

W = [
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4]
]
W = np.array(W)
W = W.reshape((4, 3))

Y = np.add(np.matmul(W, X), B)

Z = np.multiply(W, 2)
Z = np.subtract(Z, W)
pprint(1-np.tanh(3)**2)
