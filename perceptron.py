import random
import numpy as np


class Perceptron:

    def __init__(self, learning_rate, max_epochs, normalization_range=None, vectors_size=2):
        if normalization_range is None:
            normalization_range = [-1.0, 1.0]
        self.normalization_range = normalization_range
        self.learning_rate = learning_rate
        self.vectors_size = vectors_size
        self.max_epochs = max_epochs
        self.weights = []

    def init_weights(self):
        self.weights = []
        start = self.normalization_range[0]
        end = self.normalization_range[1]
        self.weights.append(random.uniform(start, end))  # Theta
        for _ in range(self.vectors_size):               # Other weights
            self.weights.append(random.uniform(start, end))
        self.weights = np.array(self.weights)

    def pw(self, x):
        return 1 if np.dot(x, self.weights) >= 0 else 0

