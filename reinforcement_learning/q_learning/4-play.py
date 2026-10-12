#!/usr/bin/env python3
"""Play an episode with a trained agent"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode, always exploiting the Q-table.

    env: the FrozenLakeEnv instance
    Q: numpy.ndarray Q-table
    max_steps: maximum number of steps in the episode

    Returns: total rewards for the episode, and a list of rendered
             outputs representing the board state at each step
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
