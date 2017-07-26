from __future__ import division, print_function
import math as m
import numpy as np
from numbers import Number

from BDMesh import Mesh1D
from ._helpers import check_if_integer


class Mesh1DUniform(Mesh1D):
    """
    One dimensional uniform mesh for boundary problems
    """

    def __init__(self, physical_boundary_1, physical_boundary_2,
                 boundary_condition_1=None, boundary_condition_2=None,
                 physical_step=None, num=None, crop=None):
        """
        UniformMesh1D constructor
        :param physical_boundary_1: float value of left physical boundary position.
        :param physical_boundary_2: float value of right physical boundary position.
        :param boundary_condition_1: float value of boundary condition at left physical boundary.
        :param boundary_condition_2: float value of boundary condition at right physical boundary.
        :param physical_step: float value of desired mesh physical step size. Mesh will use closest possible value.
        :param num: number of mesh nodes.
        :param crop: iterable of two integers specifying number of nodes to crop from each side of mesh.
        """
        super(Mesh1DUniform, self).__init__(physical_boundary_1, physical_boundary_2,
                                            boundary_condition_1=boundary_condition_1,
                                            boundary_condition_2=boundary_condition_2)
        self.__physical_step = None
        self.__local_step = None
        self.__crop = None
        if physical_step is None:
            self.num = num
        else:
            self.physical_step = physical_step
        self.crop = crop

    def __str__(self):
        return 'Mesh1DUniform: [%2.2g; %2.2g], %2.2g step, %d nodes' % (self.physical_boundary_1,
                                                                        self.physical_boundary_2,
                                                                        self.physical_step,
                                                                        self.num)

    @property
    def physical_step(self):
        return self.__physical_step

    @physical_step.setter
    def physical_step(self, physical_step):
        assert isinstance(physical_step, Number)
        physical_step = float(abs(physical_step))
        if np.allclose([physical_step], [0.0]):
            raise ValueError('step can not be zero!')
        elif physical_step > self.jacobian:
            physical_step = self.jacobian
        num_points = int(np.ceil(self.jacobian / physical_step) + 1)
        if self.physical_boundary_1 + (num_points - 1) * physical_step > self.physical_boundary_2:
            num_points -= 1
        self.local_nodes = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
        self.__physical_step = self.local_step * self.jacobian

    @property
    def num(self):
        return len(self.local_nodes)

    @num.setter
    def num(self, num):
        if num is None:
            num_points = 2
        elif not isinstance(num, Number):
            raise ValueError('number of nodes must be integer')
        elif not check_if_integer(num):
            raise ValueError('number of nodes must be integer')
        elif int(num) < 2:
            raise ValueError('number of nodes must be greater or equal to two')
        else:
            num_points = int(num)
        self.local_nodes = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
        self.__physical_step = self.local_step * self.jacobian

    @property
    def local_step(self):
        return self.local_nodes[-1] / (self.num - 1)

    @local_step.setter
    def local_step(self, local_step):
        assert isinstance(local_step, Number)
        local_step = float(abs(local_step))
        if local_step > 1:
            local_step = 1
        elif np.allclose([local_step], [0.0]):
            raise ValueError('step can not be zero!')
        self.physical_step = local_step * self.jacobian

    @property
    def crop(self):
        return self.__crop

    @crop.setter
    def crop(self, crop):
        if crop is None:
            self.__crop = np.array([0, 0], dtype=np.int)
        else:
            try:
                _ = iter(crop)
            except TypeError:
                raise TypeError(crop, 'is not iterable')
            if len(crop) != 2:
                raise ValueError('crop must be iterable of size 2')
            else:
                crop = np.array(crop).astype(np.int)
                if np.sum(crop) > self.num - 2:
                    raise ValueError('At least two nodes must remain after trimming')
                if (crop < 0).any():
                    raise ValueError('crop positions must be greater than zero')
                self.__crop = crop

    def trim(self):
        solution = self.solution[self.crop[0]:-self.crop[1]]
        residual = self.residual[self.crop[0]:-self.crop[1]]
        self.physical_boundary_1 += self.crop[0] * self.physical_step
        self.physical_boundary_2 -= self.crop[1] * self.physical_step
        num_points = int(np.ceil(self.num - np.sum(self.crop)))
        self.local_nodes = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
        self.solution = solution
        self.residual = residual
        self.boundary_condition_1 = self.solution[0]
        self.boundary_condition_2 = self.solution[-1]
        self.crop = np.array([0, 0])

    def inner_mesh_indices(self, mesh):
        assert isinstance(mesh, Mesh1D)
        if mesh.is_inside_of(self):
            local_start = self.to_local_coordinate(mesh.physical_boundary_1)
            local_stop = self.to_local_coordinate(mesh.physical_boundary_2)
            idx1 = np.where(abs(self.local_nodes - local_start) <= self.local_step / 2)[0][0]
            idx2 = np.where(abs(self.local_nodes - local_stop) <= self.local_step / 2)[0][0]
            return [idx1, idx2]
        else:
            return [None, None]

    def is_aligned_with(self, mesh):
        assert isinstance(mesh, Mesh1DUniform)
        min_step = min(mesh.physical_step, self.physical_step)
        max_step = max(mesh.physical_step, self.physical_step)
        step_ratio = max_step / min_step
        if check_if_integer(step_ratio, 1e-8):
            shift = abs(self.physical_boundary_1 - mesh.physical_boundary_1) / min_step
            if check_if_integer(shift, 1e-8):
                return True
            return False
        else:
            return False

    def merge_with(self, mesh):
        assert isinstance(mesh, Mesh1DUniform)
        if self.overlap_with(mesh) and abs(self.physical_step - mesh.physical_step) < 0.1 * self.physical_step:
            if self.is_aligned_with(mesh):
                if self.physical_boundary_1 > mesh.physical_boundary_1:
                    self.boundary_condition_1 = mesh.boundary_condition_1
                    self.crop[0] = mesh.crop[0]
                    self.physical_boundary_1 = mesh.physical_boundary_1
                if self.physical_boundary_2 < mesh.physical_boundary_2:
                    self.boundary_condition_2 = mesh.boundary_condition_2
                    self.crop[1] = mesh.crop[1]
                    self.physical_boundary_2 = mesh.physical_boundary_2
                self.physical_step = min(self.physical_step, mesh.physical_step)
                # TODO: solution/residual merging
                self.solution = np.zeros(self.num)
                self.residual = np.zeros(self.num)
            else:
                print('meshes are not aligned and could not be merged')
        else:
            print(abs(self.physical_step - mesh.physical_step), self.physical_step)
            print('meshes do not overlap or have different step size')
