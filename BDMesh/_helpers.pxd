cdef bint check_if_integer_c(double x, double *threshold) nogil
cpdef bint check_if_integer(double x, double threshold=*) nogil

cdef double[:] interp_1d(double[:] x_new, double[:] x, double[:] y)

cdef double trapz_1d(double[:] x, double[:] y)

cdef double[:] linspace(double start, double stop, int num)

cdef bint allclose(double[:] x, double[:] y, double threshold=*)
