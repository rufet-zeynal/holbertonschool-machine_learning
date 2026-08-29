#!/usr/bin/env python3
"""
Monte Carlo algorithm for Value Estimation
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: The Gymnasium environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns the next action.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        # Calculate returns and update value estimates
        G = 0
        # Keep track of states for the first-visit check
        states_in_episode = [step[0] for step in episode]

        for i in range(len(episode) - 1, -1, -1):
            s, r = episode[i]
            G = gamma * G + r

            # First-visit MC: only update if it's the first time we visited
            # this state in the current episode
            if s not in states_in_episode[:i]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
