#!/usr/bin/env python3
"""Monte Carlo module for Reinforcement Learning."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm for value estimation.

    Parameters:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode = []
        for step in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        states = [step[0] for step in episode]
        for i, (s, r) in enumerate(reversed(episode)):
            G = gamma * G + r
            step_idx = len(episode) - 1 - i
            if s not in states[:step_idx]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
