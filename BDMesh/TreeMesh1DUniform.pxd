from .TreeMesh1D cimport TreeMesh1D

cdef class TreeMesh1DUniform(TreeMesh1D):

    cdef:
        int __refinement_coefficient
        bint __aligned
        int[2] __crop

    cpdef void trim(self)