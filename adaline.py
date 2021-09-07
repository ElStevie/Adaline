import numpy as np

from perceptron import Perceptron


class Adaline(Perceptron):
    def __init__(self, learning_rate, max_epochs, normalization_range=None, vectors_size=2):
        super().__init__(learning_rate, max_epochs, normalization_range, vectors_size)

    def __activation_function(self, activation_value):
        return 1.0 / (1 + np.exp(activation_value))

    def pw(self, x):
        return self.__activation_function(np.dot(x, self.weights))

    def predict(self, x):
        return 1 if self.pw(x) >= 0 else 0
