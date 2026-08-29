#!/usr/bin/env python3
"""Monte Carlo algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Performs the Monte Carlo algorithm.

    env: environment instance
    V: ndarray, shape (s,) - value estimates
    policy: function(state) -> action
    episodes: number of episodes to train over
    max_steps: max steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        # generate one full episode under the policy
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        episode = np.array(episode, dtype=int)
        G = 0
        visited = set()

        # walk the episode backward, update on first visit only
        for state, reward in episode[::-1]:
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
