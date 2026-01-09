#!/usr/bin/env python3
"""
Python plotting
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Bar chart plotting
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    fruit_name = ['apples', 'bananas', 'oranges', 'peaches']
    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    bottom = np.zeros(3)
    for i in range(4):
        plt.bar(names, fruit[i], bottom=bottom, color=colors[i],
                label=fruit_name[i], width=0.5)
        bottom += fruit[i]
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
