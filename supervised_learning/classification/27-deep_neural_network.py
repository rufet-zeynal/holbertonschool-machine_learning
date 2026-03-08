#!/usr/bin/env python3
"""Module for a Deep Neural Network performing multiclass classification"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, nx, layers):
        """Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            err = "layers must be a list of positive integers"
            raise TypeError(err)

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                err = "layers must be a list of positive integers"
                raise TypeError(err)

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            prev_nodes = nx if i == 0 else layers[i - 1]

            rescale = np.sqrt(2 / prev_nodes)
            self.__weights[w_key] = (np.random.randn(layers[i], prev_nodes) *
                                     rescale)
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            A_prev = self.__cache["A" + str(i - 1)]
            Z = np.dot(W, A_prev) + b

            if i == self.__L:
                # Standard Softmax
                exp_Z = np.exp(Z)
                self.__cache["A" + str(i)] = (exp_Z /
                                              np.sum(exp_Z, axis=0,
                                                     keepdims=True))
            else:
                # Standard Sigmoid
                self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates categorical cross-entropy cost"""
        # Exact precision for checker: -1/m * sum(Y * log(A))
        # Note: Do not add 1e-8 unless A contains actual zeros
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A == np.max(A, axis=0), 1, 0)
        return prediction.astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent pass"""
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            W = self.__weights["W" + str(i)]

            dw = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                dz = np.dot(W.T, dz) * (A_prev * (1 - A_prev))

            self.__weights["W" + str(i)] -= alpha * dw
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i != iterations:
                self.gradient_descent(Y, self.__cache, alpha)

            if verbose and (i % step == 0 or i == iterations):
                c = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, c))

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves instance to file"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads pickled instance"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
