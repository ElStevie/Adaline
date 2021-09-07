import numpy as np

from perceptron import Perceptron


class Adaline(Perceptron):
    def __init__(self, learning_rate, max_epochs, normalization_range=None, vectors_size=2):
        super().__init__(learning_rate, max_epochs, normalization_range, vectors_size)

    def activation_function(self, activation_value):
        return 1.0 / (1.0 + np.exp(-activation_value))

    def fw(self, x):
        return self.activation_function(np.dot(x, self.weights))

