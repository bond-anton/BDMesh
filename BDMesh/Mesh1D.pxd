cdef class Mesh1D:

    cdef:
        double __physical_boundary_1, __physical_boundary_2
        double __boundary_condition_1, __boundary_condition_2
        double[:] __local_nodes
        double[:] __solution
        double[:] __residual

    cdef double j(self) nogil
    cpdef double[:] to_physical_coordinate(self, double[:] x)
    cpdef double[:] to_local_coordinate(self, double[:] x)
    cdef double int_res(self)
    cpdef bint is_inside_of(self, Mesh1D mesh)
    cpdef bint overlap_with(self, Mesh1D mesh)
    cpdef bint merge_with(self, Mesh1D other, double threshold=*, bint self_priority=*)
    cpdef double[:] interpolate_solution(self, double[:] phys_nodes)
    cpdef double[:] interpolate_residual(self, double[:] phys_nodes)
