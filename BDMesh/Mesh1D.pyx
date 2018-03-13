from __future__ import division, print_function


cdef class Mesh1D(object):
    cdef:
        double __physical_boundary_1, __physical_boundary_2
        double __boundary_condition_1, __boundary_condition_2
        double[:] __local_nodes
        double[:] __solution
        double[:] __residual

    def __cinit__(self, double physical_boundary_1, double physical_boundary_2,
                  double boundary_condition_1, double boundary_condition_2):
        if physical_boundary_1 < physical_boundary_2:
            self.__physical_boundary_1 = physical_boundary_1
            self.__physical_boundary_2 = physical_boundary_2
            self.__boundary_condition_1 = boundary_condition_1
            self.__boundary_condition_2 = boundary_condition_2
        else:
            self.__physical_boundary_2 = physical_boundary_1
            self.__physical_boundary_1 = physical_boundary_2
            self.__boundary_condition_2 = boundary_condition_1
            self.__boundary_condition_1 = boundary_condition_2

    @property
    def physical_boundary_1(self):
        return self.__physical_boundary_1

    @physical_boundary_1.setter
    def physical_boundary_1(self, physical_boundary_1):
        if self.__physical_boundary_2 is None or self.__physical_boundary_2 > float(physical_boundary_1):
            self.__physical_boundary_1 = float(physical_boundary_1)
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    def __str__(self):
        return 'Mesh1D: [%2.2g; %2.2g], %d nodes' % (self.__physical_boundary_1, self.__physical_boundary_2, 2)
