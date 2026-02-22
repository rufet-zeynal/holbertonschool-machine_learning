#!/usr/bin/env python3
"""Module for building a Random Forest."""
import numpy as np


class Node:
    """Represents an internal decision node."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None
        self.indicator = None

    def max_depth_below(self):
        """Returns the maximum depth of the subtree."""
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes or leaves in the subtree."""
        if self.is_leaf:
            return 1
        l_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        r_count = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return l_count + r_count
        return 1 + l_count + r_count

    def get_leaves_below(self):
        """Returns a list of all leaves in the subtree."""
        if self.is_leaf:
            return [self]
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively computes feature bounds for each node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Computes the indicator function for the node."""
        def is_large_enough(x):
            return np.all(np.array([np.greater(x[:, k], self.lower[k])
                                    for k in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([np.less_equal(x[:, k], self.upper[k])
                                    for k in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Recursive prediction."""
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Terminal leaf node."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        pass


class Decision_Tree():
    """Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_predict(self):
        self.root.update_bounds_below()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(np.array(
            [leaf.value * leaf.indicator(A) for leaf in leaves]), axis=0)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            f_min, f_max = self.np_extrema(
                self.explanatory[:, feat][node.sub_population])
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feat, threshold

    def fit(self, explanatory, target):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()

    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)
        mask = self.explanatory[:, node.feature] > node.threshold
        left_pop = np.logical_and(node.sub_population, mask)
        right_pop = np.logical_and(node.sub_population, ~mask)

        def is_leaf(pop, depth):
            if not np.any(pop) or depth >= self.max_depth:
                return True
            if np.sum(pop) < self.min_pop:
                return True
            return np.unique(self.target[pop]).size == 1

        if is_leaf(left_pop, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if is_leaf(right_pop, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        target_subset = self.target[sub_population]
        value = np.argmax(np.bincount(target_subset))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size


class Random_Forest():
    """Ensemble of Decision Trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.split_criterion = split_criterion
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def fit(self, explanatory, target, verbose=0):
        """Fits the forest by training multiple trees on bagged data."""
        for _ in range(self.n_trees):
            # Bagging: Sample with replacement
            indices = self.rng.integers(0, target.size, target.size)
            # Create a new seed for each tree from the forest's RNG
            tree_seed = self.rng.integers(0, 2**32)
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=tree_seed,
                              split_criterion=self.split_criterion)
            T.fit(explanatory[indices], target[indices])
            self.trees.append(T)

        if verbose == 1:
            print("  Training finished.")
            print("    - Mean depth                     : {}".format(
                np.mean([t.depth() for t in self.trees])))
            print("    - Mean number of nodes           : {}".format(
                np.mean([t.count_nodes() for t in self.trees])))
            print("    - Mean number of leaves          : {}".format(
                np.mean([t.count_nodes(only_leaves=True) for t in self.trees])))
            print("    - Mean accuracy on training data : {}".format(
                np.mean([t.accuracy(explanatory, target) for t in self.trees])))
            print("    - Accuracy of the forest on td : {}".format(
                self.accuracy(explanatory, target)))

    def predict(self, explanatory):
        """Predicts by taking the majority vote of all trees."""
        all_preds = np.array([t.predict(explanatory) for t in self.trees])
        # Majority vote across trees
        def majority(col):
            return np.argmax(np.bincount(col.astype('int32')))
        return np.apply_along_axis(majority, axis=0, arr=all_preds)

    def accuracy(self, test_explanatory, test_target):
        """Calculates accuracy of the forest ensemble."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
