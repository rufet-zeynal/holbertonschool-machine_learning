#!/usr/bin/env python3
"""Module to load Gymnasium's FrozenLake environment."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLake environment with text-based rendering."""
    if desc is None and map_name is None:
        map_name = '8x8'

    return gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
