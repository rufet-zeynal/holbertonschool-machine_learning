#!/usr/bin/env python3
"""
Decision Tree construction - counting nodes and leaves
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None,
                 left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes the node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the max depth below this node.
        """
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below this node.
        """
        left_count =\
            self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count =\
            self.right_child.count_nodes_below(only_leaves=only_leaves)

        if only_leaves:
            return left_count + right_count
        else:
            return 1 + left_count + right_count


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        """Initializes the leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the leaf's depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 because a leaf is always one node/leaf."""
        return 1


class Decision_Tree():
    """
    The main Decision Tree class.
    """
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
        """Returns the tree depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the total number of nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)
