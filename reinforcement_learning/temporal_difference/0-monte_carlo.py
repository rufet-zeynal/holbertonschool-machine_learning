import numpy as np


def value_iteration(env, gamma=0.9, theta=1e-5):
    """Performs value iteration on a given Gym/Farama environment."""
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            action_values = []

            for a in env.P[s]:
                v_a = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    v_a += prob * (reward + gamma * V[next_state])
                action_values.append(v_a)

            V[s] = max(action_values) if action_values else 0
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V
