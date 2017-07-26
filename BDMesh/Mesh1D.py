from __future__ import division, print_function
import numpy as np
from numbers import Number


class Mesh1D(object):
    """
    One dimensional mesh for boundary problems
    """

    def __init__(self, physical_boundary_1, physical_boundary_2,
                 boundary_condition_1=None, boundary_condition_2=None):
        """
        UniformMesh1D constructor
        :param physical_boundary_1: float value of left physical boundary position
        :param physical_boundary_2: float value of right physical boundary position
        :param boundary_condition_1: float value of boundary condition at left physical boundary
        :param boundary_condition_2: float value of boundary condition at right physical boundary
        """
        assert isinstance(physical_boundary_1, Number)
        assert isinstance(physical_boundary_2, Number)
        self.__physical_boundary_1 = None
        self.__physical_boundary_2 = None
        self.__boundary_condition_1 = None
        self.__boundary_condition_2 = None
        self.__local_nodes = None
        self.__solution = None
        self.__residual = None
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

        self.local_nodes = np.array([0.0, 1.0])
        self.solution = np.zeros(self.num)
        self.residual = np.zeros(self.num)

    def __str__(self):
        return 'Mesh1D: [%2.2g; %2.2g], %d nodes' % (self.physical_boundary_1, self.physical_boundary_2, self.num)

    def __eq__(self, other):
        assert isinstance(other, Mesh1D)
        if self.physical_boundary_1 == other.physical_boundary_1:
            if self.physical_boundary_2 == other.physical_boundary_2:
                if self.local_nodes.size == other.local_nodes.size:
                    if np.allclose(self.local_nodes, other.local_nodes):
                        return True
        return False

    @property
    def physical_boundary_1(self):
        return self.__physical_boundary_1

    @physical_boundary_1.setter
    def physical_boundary_1(self, physical_boundary_1):
        assert isinstance(physical_boundary_1, Number)
        if self.__physical_boundary_2 is None or self.__physical_boundary_2 > float(physical_boundary_1):
            self.__physical_boundary_1 = float(physical_boundary_1)
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    @property
    def physical_boundary_2(self):
        return self.__physical_boundary_2

    @physical_boundary_2.setter
    def physical_boundary_2(self, physical_boundary_2):
        assert isinstance(physical_boundary_2, Number)
        if self.__physical_boundary_1 is None or self.__physical_boundary_1 < float(physical_boundary_2):
            self.__physical_boundary_2 = float(physical_boundary_2)
        else:
            raise ValueError('physical boundary 2 must be greater than physical boundary 1')

    @property
    def boundary_condition_1(self):
        return self.__boundary_condition_1

    @boundary_condition_1.setter
    def boundary_condition_1(self, boundary_condition_1):
        if boundary_condition_1 is None or isinstance(boundary_condition_1, Number):
            self.__boundary_condition_1 = boundary_condition_1
        else:
            raise ValueError('Boundary condition must be either number or None')

    @property
    def boundary_condition_2(self):
        return self.__boundary_condition_2

    @boundary_condition_2.setter
    def boundary_condition_2(self, boundary_condition_2):
        if boundary_condition_2 is None or isinstance(boundary_condition_2, Number):
            self.__boundary_condition_2 = boundary_condition_2
        else:
            raise ValueError('Boundary condition must be either number or None')

    @property
    def local_nodes(self):
        return self.__local_nodes

    @local_nodes.setter
    def local_nodes(self, local_nodes):
        try:
            _ = iter(local_nodes)
        except TypeError:
            raise TypeError(local_nodes, 'is not iterable')
        if len(local_nodes) < 2:
            raise ValueError('Mesh must have at least two nodes')
        if (np.array(local_nodes).astype(np.float)[[0, -1]] == np.array([0.0, 1.0])).all():
            self.__local_nodes = np.array(local_nodes).astype(np.float)
        else:
            raise ValueError('Local mesh nodes must start with 0.0 and end with 1.0')

    @property
    def jacobian(self):
        return self.physical_boundary_2 - self.physical_boundary_1

    def to_physical_coordinate(self, x):
        return self.physical_boundary_1 + self.jacobian * x

    def to_local_coordinate(self, x):
        return (x - self.physical_boundary_1) / self.jacobian

    @property
    def physical_nodes(self):
        return self.to_physical_coordinate(self.local_nodes)

    @property
    def num(self):
        return len(self.local_nodes)

    @num.setter
    def num(self, num):
        raise AttributeError("can't set attribute")

    @property
    def solution(self):
        return self.__solution

    @solution.setter
    def solution(self, solution):
        try:
            _ = iter(solution)
        except TypeError:
            raise TypeError(solution, 'is not iterable')
        if len(solution) == self.num:
            self.__solution = np.array(solution)
        else:
            raise ValueError('Length of solution must match number of mesh nodes')

    @property
    def residual(self):
        return self.__residual

    @residual.setter
    def residual(self, residual):
        try:
            _ = iter(residual)
        except TypeError:
            raise TypeError(residual, 'is not iterable')
        if len(residual) == self.num:
            self.__residual = np.array(residual)
        else:
            raise ValueError('Length of residual must match number of mesh nodes')

    @property
    def integrational_residual(self):
        return np.trapz(self.residual, self.physical_nodes)

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

    def is_inside_of(self, mesh):
        assert isinstance(mesh, Mesh1D)
        if mesh.physical_boundary_1 <= self.physical_boundary_1:
            if mesh.physical_boundary_2 >= self.physical_boundary_2:
                return True
            else:
                return False
        else:
            return False

    def overlap_with(self, mesh):
        assert isinstance(mesh, Mesh1D)
        if self.is_inside_of(mesh) or mesh.is_inside_of(self):
            return True
        elif mesh.physical_boundary_1 <= self.physical_boundary_1 <= mesh.physical_boundary_2:
            return True
        elif mesh.physical_boundary_2 >= self.physical_boundary_2 >= mesh.physical_boundary_1:
            return True
        else:
            return False

    def merge_with(self, mesh):
        assert isinstance(mesh, Mesh1D)
        if self.overlap_with(mesh):
            merged_physical_nodes = np.unique(np.concatenate((self.physical_nodes, mesh.physical_nodes)).round(12))
            if self.physical_boundary_1 > mesh.physical_boundary_1:
                self.boundary_condition_1 = mesh.boundary_condition_1
                self.physical_boundary_1 = mesh.physical_boundary_1
            if self.physical_boundary_2 < mesh.physical_boundary_2:
                self.boundary_condition_2 = mesh.boundary_condition_2
                self.physical_boundary_2 = mesh.physical_boundary_2
            self.local_nodes = np.concatenate(([0.0], self.to_local_coordinate(merged_physical_nodes[1:-1]), [1.0]))
            # TODO: solution/residual merging
            self.solution = np.zeros(self.num)
            self.residual = np.zeros(self.num)

    # TODO: add/remove nodes routines
