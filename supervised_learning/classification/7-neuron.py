#!/usr/bin/env python3
"""
Module defining a single neuron with full training, verbosity,
and graphing capabilities.
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights getter"""
        return self.__W

    @property
    def b(self):
        """Bias getter"""
        return self.__b

    @property
    def A(self):
        """Activation getter"""
        return self.__A

    def forward_prop(self, X):
        """Calculates forward propagation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent"""
        m = Y.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron, prints progress, and plots the cost curve.
        """
        # 1. Base Validations
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # 2. Step Validations
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                err = "step must be positive and <= iterations"
                raise ValueError(err)

        costs = []
        steps = []

        # 3. Training Loop
        for i in range(iterations + 1):
            self.forward_prop(X)

            # Check step or if it's the final iteration
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__A)

                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

                if graph:
                    costs.append(cost)
                    steps.append(i)

            # 4. Perform Gradient Descent (skip on final loop)
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)

        # 5. Graphing
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # 6. Return Final Evaluation
        return self.evaluate(X, Y)
