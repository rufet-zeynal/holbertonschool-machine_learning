#!/usr/bin/env python3
"""Module for building an Isolation Random Tree."""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Class representing a random tree for isolation."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initializes the isolation tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Returns string representation."""
        return self.root.__str__()

    def depth(self):
        """Returns max depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts nodes."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Updates node bounds."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Returns all leaves."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Sets predict function based on leaf depths."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def pred_func(A):
            return np.sum(np.array([leaf.value * leaf.indicator(A)
                                    for leaf in leaves]), axis=0)
        self.predict = pred_func

    def np_extrema(self, arr):
        """Helper to find min and max."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Finds a random feature and threshold."""
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            pop = node.sub_population
            f_min, f_max = self.np_extrema(self.explanatory[pop, feat])
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feat, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf whose value is its depth."""
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively splits nodes."""
        node.feature, node.threshold = \
            self.random_split_criterion(node)

        val = self.explanatory[:, node.feature]
        mask = val > node.threshold
        left_pop = np.logical_and(node.sub_population, mask)
        right_pop = np.logical_and(node.sub_population, ~mask)

        def is_it_a_leaf(pop, depth):
            return depth >= self.max_depth or np.sum(pop) <= 1

        if is_it_a_leaf(left_pop, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if is_it_a_leaf(right_pop, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Trains the isolation tree."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        n_obs = explanatory.shape[0]
        self.root.sub_population = np.ones(n_obs, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print("    - Depth                     : {}"
                  .format(self.depth()))
            print("    - Number of nodes           : {}"
                  .format(self.count_nodes()))
            print("    - Number of leaves          : {}"
                  .format(self.count_nodes(only_leaves=True)))
