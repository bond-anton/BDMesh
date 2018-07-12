from .Mesh1D cimport Mesh1D

cdef class Mesh1DUniform(Mesh1D):

    cdef:
        int __num
        int[2] __crop

    cdef double __calc_local_step(self)
    cdef double __calc_physical_step(self)

    cpdef trim(self)
