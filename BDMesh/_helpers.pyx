from __future__ import division, print_function
from libc.math cimport floor, ceil


cdef bint check_if_integer_c(double x, double *threshold):
    lower = floor(x)
    upper = ceil(x)
    closest = lower if abs(lower - x) < abs(upper - x) else upper
    if abs(closest - x) < threshold[0]:
        return True
    else:
        return False

def check_if_integer(double x, double threshold=1.0e-10):
    return check_if_integer_c(x, &threshold)
