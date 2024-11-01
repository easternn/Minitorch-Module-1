"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

def mul(a, b):
    return a * b


def id(a):
    return a


def add(a, b):
    return a + b


def neg(a):
    return -a


def lt(a, b):
    return a < b


def eq(a, b):
    return a == b


def max(a, b):
    if a > b:
        return a
    return b


def is_close(a, b):
    return abs(a - b) < 1e-2


def sigmoid(a):
    if a >= 0:
        return 1 / (1 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))


def relu(a):
    if a <= 0:
        return 0.0
    return a


def log(a):
    return math.log(a)


def exp(a):
    return math.exp(a)


def inv(a):
    return 1 / a


def log_back(a, b):
    return b / a


def inv_back(a, b):
    return (-1 / a**2) * b


def relu_back(a, b):
    if a > 0:
        return b
    return 0

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

def map(iterable, func):
    a = []
    for item in iterable:
        a.append(func(item))
    return a


def zipWith(iterable_1, iterable_2, func):
    a = []
    iterable_2_iterator = iterable_2.__iter__()
    for item1 in iterable_1:
        item2 = next(iterable_2_iterator)
        a.append(func(item1, item2))
    return a


def reduce(a, func):
    current = None
    for item in a:
        if current is None:
            current = item
        else:
            current = func(current, item)
    if current is None:
        return 0
    return current


def negList(a):
    return map(a, neg)


def addLists(a, b):
    return zipWith(a, b, add)


def sum(a):
    return reduce(a, add)


def prod(a):
    return reduce(a, mul)

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
