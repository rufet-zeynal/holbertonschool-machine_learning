#!/usr/bin/env python3
"""Estimate state values using TD(lambda)."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform TD(lambda) with accumulating eligibility traces.

    Uses the initialized terminal-state values for bootstrapping,
    following the convention in the provided example.

    Returns:
        V: The updated state-value estimates.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V, dtype=float)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V
