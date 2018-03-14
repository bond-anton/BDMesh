from __future__ import division, print_function
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


cdef double trapz_1d(double[:] y, double[:] x):
    cdef:
        int nx = x.shape[0], ny = y.shape[0], i
        double res = 0.0
    if nx == ny:
        with boundscheck(False), wraparound(False):
            for i in range(nx - 1):
                res += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2
        return res
    else:
        raise ValueError('x and y must be the same size')
