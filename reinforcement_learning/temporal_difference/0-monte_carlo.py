#!/usr/bin/env python3
"""Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    for _ in range(episodes):
        state, _ = env.reset()

        states = []
        rewards = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)

            state = next_state

            if terminated or truncated:
                break

        G = 0
        visited = set()

        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G

            state = states[t]

            if state in visited:
                continue

            visited.add(state)

            V[state] += alpha * (G - V[state])

    return V
