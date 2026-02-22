#!/usr/bin/env python3
"""
Decision Tree construction - printing the tree
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |      " + line + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "           " + line + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = f"node [feature={self.feature}, threshold={self.threshold}]"

        left_text = str(self.left_child)
        right_text = str(self.right_child)

        text += "\n" + self.left_child_add_prefix(left_text)
        text += self.right_child_add_prefix(right_text)

        return text.rstrip()


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """
    The main Decision Tree class.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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

    def __str__(self):
        return str(self.root)