cdef class Mesh1D:

    cdef:
        double __physical_boundary_1, __physical_boundary_2
        double __boundary_condition_1, __boundary_condition_2
        double[:] __local_nodes
        double[:] __solution
        double[:] __residual

    cdef double j(self)
    cdef double[:] to_physical(self, double[:] x)
    cdef double[:] to_local(self, double[:] x)
    cdef double int_res(self)
    cpdef bint is_inside_of(self, Mesh1D mesh)
    cpdef bint overlap_with(self, Mesh1D mesh)
    cpdef void merge_with(self, Mesh1D other, double threshold=*, bint self_priority=*)
