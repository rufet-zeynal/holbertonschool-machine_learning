#!/usr/bin/env python3
"""Module for building a Random Forest."""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Class representing a Random Forest ensemble."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the random forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predicts the most frequent class among all trees."""
        # Generate predictions for each tree: shape (n_trees, n_individuals)
        all_preds = np.array([p(explanatory) for p in self.numpy_preds])

        # Calculate the mode (most frequent) prediction for each example
        def get_mode(column):
            """Helper to find the most frequent value in a 1D array."""
            counts = np.bincount(column.astype('int32'))
            return np.argmax(counts)

        return np.apply_along_axis(get_mode, axis=0, arr=all_preds)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Trains the random forest on a dataset."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            # Train tree i with a specific seed based on the forest seed
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            # The spacing here is critical for the 1212 character count
            print("  Training finished.")
            print("    - Mean depth                     : {}".format(
                np.array(depths).mean()))
            print("    - Mean number of nodes           : {}".format(
                np.array(nodes).mean()))
            print("    - Mean number of leaves          : {}".format(
                np.array(leaves).mean()))
            print("    - Mean accuracy on training data : {}".format(
                np.array(accuracies).mean()))
            print("    - Accuracy of the forest on td   : {}".format(
                self.accuracy(self.explanatory, self.target)))

    def accuracy(self, test_explanatory, test_target):
        """Calculates the accuracy of the entire forest."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
