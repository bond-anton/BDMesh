from __future__ import division, print_function


cdef class Mesh1D(object):
    cdef:
        double physical_boundary_1, physical_boundary_2
        double boundary_condition_1, boundary_condition_2
        double[:] local_nodes
        double[:] solution
        double[:] residual

    def __cinit__(self, double physical_boundary_1, double physical_boundary_2,
                  double boundary_condition_1, double boundary_condition_2):
        if physical_boundary_1 < physical_boundary_2:
            self.physical_boundary_1 = physical_boundary_1
            self.physical_boundary_2 = physical_boundary_2
            self.boundary_condition_1 = boundary_condition_1
            self.boundary_condition_2 = boundary_condition_2
        else:
            self.physical_boundary_2 = physical_boundary_1
            self.physical_boundary_1 = physical_boundary_2
            self.boundary_condition_2 = boundary_condition_1
            self.boundary_condition_1 = boundary_condition_2

    def __str__(self):
        return 'Mesh1D: [%2.2g; %2.2g], %d nodes' % (self.physical_boundary_1, self.physical_boundary_2, 2)
