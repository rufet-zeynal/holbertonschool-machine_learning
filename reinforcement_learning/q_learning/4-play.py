#!/usr/bin/env python3
"""Play a FrozenLake episode using a trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Play an episode by always choosing the highest-valued action.

    Args:
        env: FrozenLake environment with render_mode="ansi".
        Q: Q-table containing action values for each state.
        max_steps: Maximum number of steps in the episode.

    Returns:
        total_rewards: Total reward earned during the episode.
        rendered_outputs: Board frames, including initial and final states.
    """
    state, _ = env.reset()
    total_rewards = 0.0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
