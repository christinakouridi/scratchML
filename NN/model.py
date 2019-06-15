#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Python implementation of a Neural Network from scratch.
    It can be used for any number of hidden layers.

    Input:
    ------
    layer_sizes: number of neurons in each layer of neural network (list)
    epochs: number of training iterations (int)
    alpha: learning rate (float)
    lmbda: regularisation term (float)
    """

    def __init__(self, layer_sizes, epochs, alpha, lmbda):
        self.n = layer_sizes  # stores number of neurons in each layer
        self.L = len(layer_sizes) - 1  # number of neurons in output layer

        self.epochs = epochs  # number of training iterations
        self.alpha = alpha  # learning rate
        self.lmbda = lmbda  # regularisaiton factor

        self.parameters = {}  # dictionary to values of weights and parameters
        self.derivatives = {}

        for i in range(1, self.L + 1):
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            self.parameters['b' + str(i)] = np.ones((self.n[i], 1))  # keep bias terms separately
            self.parameters['z' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['a' + str(i)] = np.ones((self.n[i], 1))

        self.parameters['a0'] = np.ones((self.n[0], 1))
        self.parameters['J'] = 0

    def _formaty(self, y):
        """"
        Transforms training labels into a binary matrix
        """
        ymat = np.zeros((len(y), len(np.unique(y))))
        for i in range(len(y)):
            ymat[i][y[i] - 1] = 1
        return ymat

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_gradient(self, z):
        a = self._sigmoid(z)
        return a * (1 - a)

    def _forward_propagate(self, x):
        """
        Forward propagation through Neural Network.
        Calculates activations for every layer L.

        Input:
        ------
        X: one training example
        ------
        """
        self.parameters['a0'] = x

        for l in range(1, self.L + 1):
            self.parameters['z' + str(l)] = np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)]) + \
                                            self.parameters['b' + str(l)]
            self.parameters['a' + str(l)] = self._sigmoid(self.parameters['z' + str(l)])
        return

    def _calculate_cost(self, y):
        """"
        Calculates training loss for each training example.
        The regularisation term is added later on the summed loss of all training examples for a given epoch
        """
        self.parameters['J'] = -(1 / self.m) * (np.sum(y * np.log(self.parameters['a' + str(self.L)])) + \
                                                np.sum((1 - y) * np.log(1 - self.parameters['a' + str(self.L)])))
        return

    def _calculate_gradients(self, y):
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str(self.L)] - y
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)],
                                                      (self.parameters['a' + str(self.L - 1)]).T) + self.lmbda * \
                                               self.parameters['W' + str(self.L)]
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        for l in range(self.L - 1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot((self.parameters['W' + str(l + 1)]).T,
                                                     self.derivatives['dz' + str(l + 1)]) * self._sigmoid_gradient(
                self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(self.derivatives['dz' + str(l)],
                                                     (self.parameters['a' + str(l - 1)]).T) + self.lmbda * \
                                              self.parameters['W' + str(l)]
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
        return

    def _update_parameters(self):
        """"
        Updates weights with backpropagated loss
        """
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= self.alpha * self.derivatives['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.alpha * self.derivatives['db' + str(l)]
        return

    def train(self, x, y):
        """"
        Main body of the NeuralNetwork class

        Input:
        -----
        x: input array
        y: output array

        Output:
        ------
        plot of training losses vs training iterations
        """

        self.m = x.shape[0]
        y = self._formaty(y)
        Js = []

        for epoch in range(self.epochs):
            J = 0  # cost
            J_reg = 0  # regularisation cost-term
            n_c = 0  # number of correct predictions

            for i in range(self.m):
                x_i = x[i, :]
                y_i = y[i, :]

                x_i = x_i.reshape((len(x_i), 1))
                y_i = y_i.reshape((len(y_i), 1))

                self._forward_propagate(x_i)
                self._calculate_cost(y_i)
                self._calculate_gradients(y_i)
                self._update_parameters()

                J += self.parameters['J']

                # accuracy
                ypred = np.round(self.parameters['a' + str(self.L)])
                if ypred.argmax() == y_i.argmax():
                    n_c += 1

            # add regularisation cost term
            for l in range(1, self.L + 1):
                J_reg += (self.lmbda / (2 * self.m)) * (np.sum(self.parameters['W' + str(l)] ** 2))
            J += J_reg

            # keep track of costs by saving to a list
            Js.append(J)

            if epoch % 10 == 0:
                print("Iteration:", epoch, "Cost:", round(J, 4), "Accuracy:", round(n_c * 100 / self.m, 3))

        plt.plot([i for i in range(self.epochs)], Js)
        plt.show()
        return