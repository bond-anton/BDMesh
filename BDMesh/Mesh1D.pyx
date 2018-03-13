from __future__ import division, print_function
import numpy as np
from cpython.object cimport Py_EQ
from cython import boundscheck, wraparound

from ._helpers import trapz_1d


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
        self.__local_nodes = np.array([0.0, 1.0], dtype=np.double)
        self.__solution = np.array([0.0, 0.0], dtype=np.double)
        self.__residual = np.array([0.0, 0.0], dtype=np.double)

    def __str__(self):
        return 'Mesh1D: [%2.2g; %2.2g], %d nodes' % (self.__physical_boundary_1, self.__physical_boundary_2,
                                                     len(self.__local_nodes))

    def __richcmp__(x, y, int op):
        if op == Py_EQ:
            if isinstance(x, Mesh1D) and isinstance(y, Mesh1D):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if np.allclose(x.local_nodes, y.local_nodes):
                                return True
            return False
        else:
            return False

    @property
    def local_nodes(self):
        return np.asarray(self.__local_nodes)

    @local_nodes.setter
    def local_nodes(self, double[:] local_nodes):
        n = local_nodes.shape[0]
        if n < 2:
            raise ValueError('Mesh must have at least two nodes')
        if local_nodes[0] == 0.0 and local_nodes[n-1] == 1.0:
            #if self.__local_nodes is None:
            self.__local_nodes = local_nodes
            #     self.solution = np.zeros(self.num)
            #     self.residual = np.zeros(self.num)
            # else:
            #     physical_nodes_old = self.physical_nodes
            #     self.__local_nodes = np.array(local_nodes).astype(np.float)
            #     self.solution = np.interp(self.physical_nodes, physical_nodes_old, self.solution)
            #     self.residual = np.interp(self.physical_nodes, physical_nodes_old, self.residual)
        else:
            raise ValueError('Local mesh nodes must start with 0.0 and end with 1.0')

    @property
    def physical_boundary_1(self):
        return self.__physical_boundary_1

    @physical_boundary_1.setter
    def physical_boundary_1(self, double physical_boundary_1):
        if self.__physical_boundary_2 > <double>physical_boundary_1:
            self.__physical_boundary_1 = <double>physical_boundary_1
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    @property
    def physical_boundary_2(self):
        return self.__physical_boundary_2

    @physical_boundary_2.setter
    def physical_boundary_2(self, double physical_boundary_2):
        if self.__physical_boundary_1 < <double>physical_boundary_2:
            self.__physical_boundary_2 = <double>physical_boundary_2
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    cdef double jacobian(self):
        return self.__physical_boundary_2 - self.__physical_boundary_1

    cdef double[:] to_physical(self, double[:] x):
        cdef:
            int n = x.shape[0], i
            double[:] res = np.zeros(n)
        with boundscheck(False), wraparound(False):
            for i in range(n):
                res[i] = self.__physical_boundary_1 + self.jacobian() * x[i]
        return res

    def to_physical_coordinate(self, double[:] x):
        return np.asarray(self.to_physical(x))

    cdef double[:] to_local(self, double[:] x):
        cdef:
            int n = x.shape[0], i
            double[:] res = np.zeros(n)
        with boundscheck(False), wraparound(False):
            for i in range(n):
                res[i] = (x[i] - self.__physical_boundary_1) / self.jacobian()
        return res

    def to_local_coordinate(self, double[:] x):
        return np.asarray(self.to_local(x))

    @property
    def physical_nodes(self):
        return np.asarray(self.to_physical(self.__local_nodes))

    @property
    def num(self):
        return self.__local_nodes.shape[0]

    @property
    def boundary_condition_1(self):
        return self.__boundary_condition_1

    @boundary_condition_1.setter
    def boundary_condition_1(self, double boundary_condition_1):
        self.__boundary_condition_1 = <double>boundary_condition_1

    @property
    def boundary_condition_2(self):
        return self.__boundary_condition_2

    @boundary_condition_2.setter
    def boundary_condition_2(self, double boundary_condition_2):
        self.__boundary_condition_2 = <double>boundary_condition_2

    @property
    def solution(self):
        return np.asarray(self.__solution)

    @solution.setter
    def solution(self, solution):
        if solution.shape[0] == self.__local_nodes.shape[0]:
            self.__solution = solution
        else:
            raise ValueError('Length of solution must match number of mesh nodes')

    @property
    def residual(self):
        return np.asarray(self.__residual)

    @residual.setter
    def residual(self, residual):
        if residual.shape[0] == self.__local_nodes.shape[0]:
            self.__residual = residual
        else:
            raise ValueError('Length of residual must match number of mesh nodes')

    cdef double int_res(self):
        return trapz_1d(self.__residual, self.to_physical(self.__local_nodes))

    @property
    def integrational_residual(self):
        return self.int_res()

    cpdef bint is_inside_of(self, Mesh1D mesh):
        if mesh.__physical_boundary_1 <= self.__physical_boundary_1:
            if mesh.__physical_boundary_2 >= self.__physical_boundary_2:
                return True
            else:
                return False
        else:
            return False

    cpdef bint overlap_with(self, Mesh1D mesh):
        if self.is_inside_of(mesh) or mesh.is_inside_of(self):
            return True
        elif mesh.__physical_boundary_1 <= self.__physical_boundary_1 <= mesh.__physical_boundary_2:
            return True
        elif mesh.__physical_boundary_2 >= self.__physical_boundary_2 >= mesh.__physical_boundary_1:
            return True
        else:
            return False
