from __future__ import division, print_function
import numpy as np
from cpython.object cimport Py_EQ, Py_NE
from cython import boundscheck, wraparound

from ._helpers cimport trapz_1d, interp_1d


cdef class Mesh1D(object):

    def __init__(self, double physical_boundary_1, double physical_boundary_2,
                 double boundary_condition_1=0.0, double boundary_condition_2=0.0):
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
        elif op == Py_NE:
            if isinstance(x, Mesh1D) and isinstance(y, Mesh1D):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if np.allclose(x.local_nodes, y.local_nodes):
                                return False
            return True
        else:
            return False

    @property
    def local_nodes(self):
        return np.asarray(self.__local_nodes)

    @local_nodes.setter
    def local_nodes(self, double[:] local_nodes):
        cdef:
            int n = local_nodes.shape[0]
            double[:] physical_nodes_old
        if n < 2:
            raise ValueError('Mesh must have at least two nodes')
        if local_nodes[0] == 0.0 and local_nodes[n-1] == 1.0:
            physical_nodes_old = self.to_physical(self.__local_nodes)
            self.__local_nodes = local_nodes
            self.__solution = interp_1d(self.to_physical(self.__local_nodes), physical_nodes_old, self.__solution)
            self.__residual = interp_1d(self.to_physical(self.__local_nodes), physical_nodes_old, self.__residual)
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

    cdef double j(self):
        return self.__physical_boundary_2 - self.__physical_boundary_1

    @property
    def jacobian(self):
        return self.j()

    cdef double[:] to_physical(self, double[:] x):
        cdef:
            int n = x.shape[0], i
            double[:] res = np.zeros(n)
        with boundscheck(False), wraparound(False):
            for i in range(n):
                res[i] = self.__physical_boundary_1 + self.j() * x[i]
        return res

    def to_physical_coordinate(self, double[:] x):
        return np.asarray(self.to_physical(x))

    cdef double[:] to_local(self, double[:] x):
        cdef:
            int n = x.shape[0], i
            double[:] res = np.zeros(n)
        with boundscheck(False), wraparound(False):
            for i in range(n):
                res[i] = (x[i] - self.__physical_boundary_1) / self.j()
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
    def solution(self, double[:] solution):
        if solution.shape[0] == self.__local_nodes.shape[0]:
            self.__solution = solution
        else:
            raise ValueError('Length of solution must match number of mesh nodes')

    @property
    def residual(self):
        return np.asarray(self.__residual)

    @residual.setter
    def residual(self, double[:] residual):
        if residual.shape[0] == self.__local_nodes.shape[0]:
            self.__residual = residual
        else:
            raise ValueError('Length of residual must match number of mesh nodes')

    cdef double int_res(self):
        return trapz_1d(self.__residual, self.to_physical(self.__local_nodes))

    @property
    def integrational_residual(self):
        return self.int_res()

    def local_f(self, f, args=None):
        """
        return function equivalent to f on local nodes
        :param f: callable with first argument x - coordinate in physical space
        :param args: possible additional arguments of f
        :return: function equivalent to f on local nodes
        """
        assert callable(f)

        def f_local(x, arguments=args):
            if arguments is not None:
                return f(self.to_physical_coordinate(x), arguments)
            else:
                return f(self.to_physical_coordinate(x))

        return f_local

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

    cpdef bint merge_with(self, Mesh1D other, double threshold=1e-10, bint self_priority=True):
        """
        Merge mesh with another mesh
        :param other: Mesh1D to merge with
        :param threshold: threshold for nodes matching
        :param self_priority: which solution and residual values are in priority ('self' or 'other')
        :return:
        """
        cdef:
            int n = self.__local_nodes.shape[0]
            int m = other.__local_nodes.shape[0]
            int i = 0, j = 0, k = 0
            double bc_1, bc_2
            double[:] phys = np.zeros(n + m)
            double[:] sol = np.zeros(n + m)
            double[:] res = np.zeros(n + m)
            double[:] phys_self, phys_other
        if not self.overlap_with(other):
            return False
        if self.__physical_boundary_1 <= other.__physical_boundary_1:
            bc_1 = self.__boundary_condition_1
        else:
            bc_1 = other.__boundary_condition_1
        if self.__physical_boundary_2 >= other.__physical_boundary_2:
            bc_2 = self.__boundary_condition_2
        else:
            bc_2 = other.__boundary_condition_2
        phys_self = self.to_physical(self.__local_nodes)
        phys_other = other.to_physical(other.__local_nodes)
        with boundscheck(False), wraparound(False):
            while i < n or j < m:
                if i < n:
                    if j < m:
                        if abs(phys_self[i] - phys_other[j]) < threshold:
                            if self_priority:
                                phys[k] = phys_self[i]
                                sol[k] = self.__solution[i]
                                res[k] = self.__residual[i]
                            else:
                                phys[k] = phys_other[j]
                                sol[k] = other.__solution[j]
                                res[k] = other.__residual[j]
                            i += 1
                            j += 1
                        elif phys_self[i] < phys_other[j]:
                            phys[k] = phys_self[i]
                            sol[k] = self.__solution[i]
                            res[k] = self.__residual[i]
                            i += 1
                        else:
                            phys[k] = phys_other[j]
                            sol[k] = other.__solution[j]
                            res[k] = other.__residual[j]
                            j += 1
                    else:
                        phys[k] = phys_self[i]
                        sol[k] = self.__solution[i]
                        res[k] = self.__residual[i]
                        i += 1
                else:
                    if j < m:
                        phys[k] = phys_other[j]
                        sol[k] = other.__solution[j]
                        res[k] = other.__residual[j]
                        j += 1
                k += 1
            self.__physical_boundary_1 = phys[0]
            self.__physical_boundary_2 = phys[k - 1]
            self.__boundary_condition_1 = bc_1
            self.__boundary_condition_2 = bc_2
            self.__local_nodes = self.to_local(phys[:k])
            self.__solution = sol[:k]
            self.__residual = res[:k]
            return True
