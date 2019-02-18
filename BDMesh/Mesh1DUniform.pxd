from .Mesh1D cimport Mesh1D

cdef class Mesh1DUniform(Mesh1D):

    cdef:
        int __num
        int[2] __crop

    cdef double __calc_local_step(self) nogil
    cdef double __calc_physical_step(self) nogil

    cpdef void trim(self)
    cpdef inner_mesh_indices(self, Mesh1D mesh)
    cpdef bint is_aligned_with(self, Mesh1DUniform mesh)
