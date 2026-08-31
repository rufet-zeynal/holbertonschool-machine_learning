#!/usr/bin/env python3
"""Training loop for policy gradients"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train a policy on env using Monte-Carlo policy gradient"""
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        rewards = []
        gradients = []
        done = False

        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            gradients.append(grad)
            rewards.append(reward)

        score = sum(rewards)
        scores.append(score)

        for t, grad in enumerate(gradients):
            Gt = sum(R * gamma ** i for i, R in enumerate(rewards[t:]))
            weight += alpha * grad * Gt

        print("Episode: {} Score: {}".format(episode, score))

    return scores
