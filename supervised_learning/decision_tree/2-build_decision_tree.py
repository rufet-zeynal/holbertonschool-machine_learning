#!/usr/bin/env python3
"""
Module to build and visualize a Decision Tree structure.
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes the node with its properties and children."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth of the tree below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes or leaves in the subtree."""
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Adds formatting prefixes for the left child's string representation."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds formatting prefixes for the right child's string representation."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns the string representation of the node and its children."""
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
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        """Initializes the leaf with a value and its depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 as the leaf is a single node/leaf."""
        return 1

    def __str__(self):
        """Returns the string representation of the leaf."""
        return "-> leaf [value={}]".format(self.value)


class Decision_Tree():
    """
    Main class for the Decision Tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the tree parameters."""
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
        """Returns the total depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the total count of nodes or leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns the string representation of the entire tree."""
        return self.root.__str__()
