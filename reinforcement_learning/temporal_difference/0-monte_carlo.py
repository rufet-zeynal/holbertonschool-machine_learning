#!/usr/bin/env python3
"""Monte Carlo algorithm."""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])

    return V
