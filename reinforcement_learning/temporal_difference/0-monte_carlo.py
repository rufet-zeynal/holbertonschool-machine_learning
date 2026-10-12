#!/usr/bin/env python3
"""Estimate state values using Monte Carlo prediction."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform first-visit Monte Carlo prediction.

    Args:
        env: Environment instance.
        V: numpy.ndarray of shape (s,) containing value estimates.
        policy: Function that takes a state and returns an action.
        episodes: Number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        V: Updated value estimates.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        first_visit = {}

        for step in range(max_steps):
            first_visit.setdefault(state, step)
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        total_return = 0.0

        for step in reversed(range(len(episode))):
            state, reward = episode[step]
            total_return = reward + gamma * total_return

            if first_visit[state] == step:
                V[state] += alpha * (total_return - V[state])

    return V
