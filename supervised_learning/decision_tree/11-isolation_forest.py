#!/usr/bin/env python3
"""Module for building an Isolation Random Forest."""
import numpy as np
Isolation_Random_Tree = \
    __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Class representing an isolation random forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the isolation forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Returns the mean depth of each individual across all trees."""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Trains the isolation forest."""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print("  Training finished.")
            print("    - Mean depth                     : {}"
                  .format(np.array(depths).mean()))
            print("    - Mean number of nodes           : {}"
                  .format(np.array(nodes).mean()))
            print("    - Mean number of leaves          : {}"
                  .format(np.array(leaves).mean()))

    def suspects(self, explanatory, n_suspects):
        """Returns rows in explanatory with the smallest mean depth."""
        # Get mean depth for all points
        depths = self.predict(explanatory)

        # Get indices of n_suspects with smallest depth
        # argsort gives indices from smallest to largest
        indices = np.argsort(depths)[:n_suspects]

        return explanatory[indices], depths[indices]
