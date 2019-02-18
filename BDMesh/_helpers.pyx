from libc.math cimport floor, ceil, fabs
from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone


cdef bint check_if_integer_c(double x, double *threshold) nogil:
    lower = floor(x)
    upper = ceil(x)
    closest = lower if fabs(lower - x) < fabs(upper - x) else upper
    if fabs(closest - x) < threshold[0]:
        return True
    else:
        return False


cpdef bint check_if_integer(double x, double threshold=1.0e-10) nogil:
    return check_if_integer_c(x, &threshold)


@boundscheck(False)
@wraparound(False)
cdef double[:] interp_1d(double[:] x_new, double[:] x, double[:] y):
    cdef:
        int n = x_new.shape[0], m = x.shape[0]
        int i, j = 1
        array[double] y_new, template = array('d')
    y_new = clone(template, n, zero=False)
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


@boundscheck(False)
@wraparound(False)
cdef double[:] linspace(double start, double stop, int num):
    cdef:
        int i
        double step = (stop - start) / (num - 1)
        array[double] result, template = array('d')
    result = clone(template, num, zero=False)
    for i in range(num - 1):
        result[i] = start + i * step
    result[num - 1] = stop
    return result


cdef bint allclose(double[:] x, double[:] y, double threshold=1e-14):
    cdef:
        int i, s1 = x.shape[0], s2 = y.shape[0]
    if s1 != s2:
        return False
    for i in range(s1):
        if fabs(x[i] - y[i]) > threshold:
            return False
    return True
