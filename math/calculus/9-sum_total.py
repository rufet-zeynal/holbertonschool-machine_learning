#!/usr/bin/env python3


def summation_i_squared(n):
    total = 0
    for i in range(1, n+1):
        total += i**2
    return total

n = int(input("n = "))
print(summation_i_squared(n))