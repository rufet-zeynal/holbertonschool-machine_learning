#!/usr/bin/env python3
"""SARSA(lambda) algorithm."""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm."""
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(Q)

        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                td_error = reward - Q[state, action]
            else:
                if np.random.uniform() < epsilon:
                    next_action = np.random.randint(Q.shape[1])
                else:
                    next_action = np.argmax(Q[next_state])

                td_error = (
                    reward
                    + gamma * Q[next_state, next_action]
                    - Q[state, action]
                )

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1

            Q += alpha * td_error * eligibility

            state = next_state

            if terminated or truncated:
                break

            action = next_action

        epsilon = max(epsilon - epsilon_decay, min_epsilon)

    return Q
