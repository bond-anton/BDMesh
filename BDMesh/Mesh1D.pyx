from cython import boundscheck, wraparound

from cpython.object cimport Py_EQ, Py_NE
from cpython.array cimport array, clone
from libc.math cimport fabs

from ._helpers cimport trapz_1d, interp_1d, allclose


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
        self.__local_nodes = clone(array('d'), 2, zero=False)
        self.__local_nodes[0] = 0.0
        self.__local_nodes[1] = 1.0
        self.__solution = clone(array('d'), 2, zero=True)
        self.__residual = clone(array('d'), 2, zero=True)

    def __str__(self):
        return 'Mesh1D: [%2.2g; %2.2g], %d nodes' % (self.__physical_boundary_1, self.__physical_boundary_2,
                                                     len(self.__local_nodes))

    def __richcmp__(x, y, int op):
        if op == Py_EQ:
            if isinstance(x, Mesh1D) and isinstance(y, Mesh1D):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if allclose(x.local_nodes, y.local_nodes):
                                return True
            return False
        elif op == Py_NE:
            if isinstance(x, Mesh1D) and isinstance(y, Mesh1D):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if allclose(x.local_nodes, y.local_nodes):
                                return False
            return True
        else:
            return False

    @property
    def local_nodes(self):
        return self.__local_nodes

    @local_nodes.setter
    def local_nodes(self, double[:] local_nodes):
        cdef:
            int n = local_nodes.shape[0]
            double[:] physical_nodes_old
        if n < 2:
            raise ValueError('Mesh must have at least two nodes')
        if local_nodes[0] == 0.0 and local_nodes[n-1] == 1.0:
            physical_nodes_old = self.to_physical_coordinate(self.__local_nodes)
            self.__local_nodes = local_nodes
            self.__solution = interp_1d(self.to_physical_coordinate(self.__local_nodes),
                                        physical_nodes_old, self.__solution)
            self.__residual = interp_1d(self.to_physical_coordinate(self.__local_nodes),
                                        physical_nodes_old, self.__residual)
        else:
            raise ValueError('Local mesh nodes must start with 0.0 and end with 1.0')

    @property
    def physical_boundary_1(self):
        return self.__physical_boundary_1

    @physical_boundary_1.setter
    def physical_boundary_1(self, double physical_boundary_1):
        if self.__physical_boundary_2 > physical_boundary_1:
            self.__physical_boundary_1 = physical_boundary_1
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    @property
    def physical_boundary_2(self):
        return self.__physical_boundary_2

    @physical_boundary_2.setter
    def physical_boundary_2(self, double physical_boundary_2):
        if self.__physical_boundary_1 < physical_boundary_2:
            self.__physical_boundary_2 = physical_boundary_2
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    cdef double j(self) nogil:
        return self.__physical_boundary_2 - self.__physical_boundary_1

    @property
    def jacobian(self):
        return self.j()

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] to_physical_coordinate(self, double[:] x):
        cdef:
            int s = x.shape[0], i
            array[double] result, template = array('d')
        result = clone(template, s, zero=False)
        for i in range(s):
            result[i] = self.__physical_boundary_1 + self.j() * x[i]
        return result

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] to_local_coordinate(self, double[:] x):
        cdef:
            int s = x.shape[0], i
            array[double] result, template = array('d')
        result = clone(template, s, zero=False)
        for i in range(s):
            result[i] = (x[i] - self.__physical_boundary_1) / self.j()
        return result

    @property
    def physical_nodes(self):
        return self.to_physical_coordinate(self.__local_nodes)

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
        return self.__solution

    @solution.setter
    def solution(self, double[:] solution):
        if solution.shape[0] == self.__local_nodes.shape[0]:
            self.__solution = solution
        else:
            raise ValueError('Length of solution must match number of mesh nodes')

    @property
    def residual(self):
        return self.__residual

    @residual.setter
    def residual(self, double[:] residual):
        if residual.shape[0] == self.__local_nodes.shape[0]:
            self.__residual = residual
        else:
            raise ValueError('Length of residual must match number of mesh nodes')

    cdef double int_res(self):
        return trapz_1d(self.__residual, self.to_physical_coordinate(self.__local_nodes))

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

    @boundscheck(False)
    @wraparound(False)
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
            double[:] phys_self, phys_other
            array[double] phys, sol, res, template = array('d')
        phys = clone(template, n + m, zero=False)
        sol = clone(template, n + m, zero=False)
        res = clone(template, n + m, zero=False)
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
        phys_self = self.to_physical_coordinate(self.__local_nodes)
        phys_other = other.to_physical_coordinate(other.__local_nodes)
        while i < n or j < m:
            if i < n:
                if j < m:
                    if fabs(phys_self[i] - phys_other[j]) < threshold:
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
        self.__local_nodes = self.to_local_coordinate(phys[:k])
        self.__solution = sol[:k]
        self.__residual = res[:k]
        return True

    cpdef double[:] interpolate_solution(self, double[:] phys_nodes):
        return interp_1d(phys_nodes, self.to_physical_coordinate(self.__local_nodes), self.__solution)

    cpdef double[:] interpolate_residual(self, double[:] phys_nodes):
        return interp_1d(phys_nodes, self.to_physical_coordinate(self.__local_nodes), self.__residual)
