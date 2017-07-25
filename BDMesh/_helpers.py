from __future__ import division, print_function
import math as m


def check_if_integer(x, threshold=1e-10):
    lower = m.floor(x)
    upper = m.ceil(x)
    closest = lower if abs(lower - x) < abs(upper - x) else upper
    if abs(closest - x) < threshold:
        return True
    else:
        return False
