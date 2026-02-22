#!/usr/bin/env python3
"""Module for building a Decision Tree with Gini splitting criterion."""
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
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

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
        """Recursive prediction for a single individual."""
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def left_child_add_prefix(self, text):
        """Adds formatting for left child branches."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds formatting for right child branches."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns string representation of the node."""
        if self.is_leaf:
            return "-> leaf [value={}]".format(self.value)
        if self.is_root:
            out = "root [feature={}, threshold={}]\n".format(
                self.feature, self.threshold)
        else:
            out = "node [feature={}, threshold={}]\n".format(
                self.feature, self.threshold)
        out += self.left_child_add_prefix(self.left_child.__str__())
        out += self.right_child_add_prefix(self.right_child.__str__())
        return out


class Leaf(Node):
    """Represents a terminal leaf node."""

    def __init__(self, value, depth=None):
        """Initializes a Leaf node."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """Leaves have no children."""
        pass


class Decision_Tree():
    """Main Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns max depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns count of nodes or leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Returns all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Starts recursive bound update from root."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Computes the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(np.array(
            [leaf.value * leaf.indicator(A) for leaf in leaves]), axis=0)

    def pred(self, x):
        """Returns prediction for a single individual."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Returns min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Returns a random feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feat = self.rng.integers(0, self.explanatory.shape[1])
            f_min, f_max = self.np_extrema(
                self.explanatory[:, feat][node.sub_population])
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feat, threshold

    def possible_thresholds(self, node, feature):
        """Returns list of midpoints between unique feature values."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Calculates best threshold for one feature using Gini impurity."""
        pop = node.sub_population
        feat_vals = self.explanatory[pop, feature]
        targets = self.target[pop]
        t_holds = self.possible_thresholds(node, feature)
        classes = np.unique(targets)

        # Reshape for broadcasting into (individuals, thresholds, classes)
        # targets_mask: True if individual i belongs to class k
        targets_mask = (targets[:, None] == classes[None, :])
        # feat_mask: True if individual i feature > threshold j
        feat_mask = (feat_vals[:, None] > t_holds[None, :])

        # Left_F[i, j, k]: individual i is class k AND feature > threshold j
        left_f = np.logical_and(targets_mask[:, None, :],
                                feat_mask[:, :, None])
        # Right_F[i, j, k]: individual i is class k AND feature <= threshold j
        right_f = np.logical_and(targets_mask[:, None, :],
                                 ~feat_mask[:, :, None])

        # Sum over individuals (axis 0) to get card(class k in child j)
        left_counts = np.sum(left_f, axis=0)
        right_counts = np.sum(right_f, axis=0)

        # card(P) for each child j
        left_pop_size = np.sum(left_counts, axis=1)
        right_pop_size = np.sum(right_counts, axis=1)

        # Gini = 1 - sum((class_count / pop_size)**2)
        # Use np.divide and where to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            gini_l = 1 - np.sum(np.square(np.divide(
                left_counts, left_pop_size[:, None])), axis=1)
            gini_r = 1 - np.sum(np.square(np.divide(
                right_counts, right_pop_size[:, None])), axis=1)

        gini_l = np.nan_to_num(gini_l)
        gini_r = np.nan_to_num(gini_r)

        # Weighted Gini average
        n = targets.size
        gini_avg = (left_pop_size / n) * gini_l + (right_pop_size / n) * gini_r

        best_idx = np.argmin(gini_avg)
        return t_holds[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        """Returns the best feature and threshold across all features."""
        res = np.array([self.Gini_split_criterion_one_feature(node, i)
                        for i in range(self.explanatory.shape[1])])
        best_feature = np.argmin(res[:, 1])
        return best_feature, res[best_feature, 0]

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print("    - Depth                     : {}".format(
                self.depth()))
            print("    - Number of nodes           : {}".format(
                self.count_nodes()))
            print("    - Number of leaves          : {}".format(
                self.count_nodes(only_leaves=True)))
            print("    - Accuracy on training data : {}".format(
                self.accuracy(self.explanatory, self.target)))

    def fit_node(self, node):
        """Recursively trains a node."""
        node.feature, node.threshold = self.split_criterion(node)
        mask_l = self.explanatory[:, node.feature] > node.threshold
        left_pop = np.logical_and(node.sub_population, mask_l)
        right_pop = np.logical_and(node.sub_population, ~mask_l)

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
        """Creates a leaf child."""
        target_subset = self.target[sub_population]
        value = np.argmax(np.bincount(target_subset))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates a node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates accuracy."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
