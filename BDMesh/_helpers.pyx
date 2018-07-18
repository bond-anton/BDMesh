from __future__ import division, print_function
import numpy as np
from libc.math cimport floor, ceil
from cython cimport boundscheck, wraparound


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

@boundscheck(False)
@wraparound(False)
cdef double[:] interp_1d(double[:] x_new, double[:] x, double[:] y):
    cdef:
        int n = x_new.shape[0], m = x.shape[0]
        int i, j = 1
        double[:] y_new = np.zeros(n, dtype=np.double)
    for i in range(n):
        while x_new[i] > x[j] and j < m - 1:
            j += 1
        y_new[i] = y[j-1] + (x_new[i] - x[j-1]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
    return y_new

@boundscheck(False)
@wraparound(False)
cdef double trapz_1d(double[:] y, double[:] x):
    cdef:
        int nx = x.shape[0], ny = y.shape[0], i
        double res = 0.0
    for i in range(nx - 1):
        res += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2
    return res
