#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from model import NeuralNetwork


def main():

    # ----DATA PREPARATION---- #
    x, y = make_moons(1000, noise=0.20)

    plt.scatter(x[:, 0], x[:, 1], s=10, c=y, cmap=plt.cm.Spectral)

    np.random.seed(0)
    np.random.shuffle(y)

    np.random.seed(0)
    np.random.shuffle(x)

    # ----MODEL TRAINING---- #
    settings = {
        'layer_sizes': [x.shape[1], 3, len(np.unique(y))],
        'epochs': 100,
        'alpha': 0.01,
        'lmbda': 0.001
    }

    nn = NeuralNetwork(**settings)

    nn.train(x, y)


if __name__ == "__main__":
    main()
