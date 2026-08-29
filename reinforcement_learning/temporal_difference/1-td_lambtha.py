#!/usr/bin/env python3
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """ performs the TD(λ) algorithm

        env: is the openAI environment instance
        V: is a numpy.ndarray of shape (s,) containing the
          value estimate
        policy: is a function that takes in a state and returns
          the next action to take
        lambtha: is the eligibility trace factor
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
        Returns: V, the updated value estimate
    """
    n = env.observation_space.n
    Et = [0 for i in range(n)]
    for _ in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1.0
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            if env.desc.reshape(n)[new_state] == b'G':
                reward = 1
            if env.desc.reshape(n)[new_state] == b'H':
                reward = -1
            # deltat = (R(t + 1) + gamma * V(St + 1) - V(St))
            deltat = reward + gamma * V[new_state] - V[state]
            # V(St) = V(St) + alpha * deltat * Et[state]
            V[state] = V[state] + alpha * deltat * Et[state]
            if done:
                break
            state = new_state
    return V
