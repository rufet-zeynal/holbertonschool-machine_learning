#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the first-visit Monte Carlo algorithm.

    env: environment instance
    V: numpy.ndarray of shape (s,) with the value estimate
    policy: function that takes a state and returns the next action
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = new_state

        states = [s for s, _ in episode]
        G = 0
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = r + gamma * G
            # first-visit: only update if s does not appear earlier
            if s not in states[:t]:
                V[s] += alpha * (G - V[s])

    return V
