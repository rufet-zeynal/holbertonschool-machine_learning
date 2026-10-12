#!/usr/bin/env python3
"""Estimate action values using SARSA(lambda)."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Perform SARSA(lambda) with accumulating eligibility traces.

    Uses initialized terminal-state action values for bootstrapping,
    following the convention in the provided example.

    Returns:
        Q: The updated Q table.
    """
    initial_epsilon = epsilon

    def epsilon_greedy(state):
        """Choose an action using the current epsilon."""
        if np.random.uniform() > epsilon:
            return np.argmax(Q[state])
        return np.random.randint(Q.shape[1])

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state)
        eligibility = np.zeros_like(Q, dtype=float)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(next_state)

            delta = (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon
        ) * np.exp(-epsilon_decay * episode)

    return Q
