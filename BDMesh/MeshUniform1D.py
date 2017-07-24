from __future__ import division, print_function
import math as m
import numpy as np
from numbers import Number

from BDMesh import Mesh1D
from ._helpers import check_if_integer


class MeshUniform1D(Mesh1D):
    """
    One dimensional uniform mesh for boundary problems
    """

    def __init__(self, physical_boundary_1, physical_boundary_2, physical_step,
                 boundary_condition_1, boundary_condition_2, crop=None):
        """
        UniformMesh1D constructor
        :param physical_boundary_1: float value of left physical boundary position
        :param physical_boundary_2: float value of right physical boundary position
        :param physical_step: float value of mesh physical step size
        :param boundary_condition_1: float value of boundary condition at left physical boundary
        :param boundary_condition_2: float value of boundary condition at right physical boundary
        :param crop: iterable of two integers specifying number of nodes to crop from each side of mesh
        """
        super(MeshUniform1D, self).__init__(physical_boundary_1, physical_boundary_2,
                                            boundary_condition_1, boundary_condition_2)
        self.__physical_step = None
        self.__local_step = None
        self.__crop = None
        self.physical_step = physical_step
        self.crop = crop

    @property
    def physical_step(self):
        return self.__physical_step

    @physical_step.setter
    def physical_step(self, physical_step):
        assert isinstance(physical_step, Number)
        self.__physical_step = float(abs(physical_step))
        num_points = int(np.ceil((self.physical_boundary_2 - self.physical_boundary_1) / self.__physical_step) + 1)
        if self.physical_boundary_1 + (num_points - 1) * self.__physical_step > self.physical_boundary_2:
            num_points -= 1
        self.local_nodes = np.linspace(0.0, 1.0, num=num_points, endpoint=True)

    @property
    def local_step(self):
        return self.local_nodes[-1] / (self.num - 1)

    @local_step.setter
    def local_step(self, local_step):
        assert isinstance(local_step, Number)
        self.physical_step = float(abs(local_step)) * self.jacobian

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
                if check_if_integer(crop[0]) and check_if_integer(crop[1]):
                    self.__crop = np.array([int(crop[0]), int(crop[1])], dtype=np.int)
                else:
                    raise ValueError('crop must be two integers')

    def trim(self, debug=False):
        if debug:
            print('Cropping', np.sum(self.crop), 'elements')
            print('Physical boundary 1:', self.physical_boundary_1)
            print('Physical boundary 2:', self.physical_boundary_2)
        self.physical_boundary_1 += self.crop[0] * self.physical_step
        self.physical_boundary_2 -= self.crop[1] * self.physical_step
        if debug:
            print(' -> Physical boundary 1:', self.physical_boundary_1)
            print(' -> Physical boundary 2:', self.physical_boundary_2)
        num_points = int(np.ceil(self.num - np.sum(self.crop)))
        self.local_nodes = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
        self.solution = self.solution[self.crop[0]:self.solution.size - self.crop[1]]
        self.residual = self.residual[self.crop[0]:self.residual.size - self.crop[1]]
        self.boundary_condition_1 = self.solution[0]
        self.boundary_condition_2 = self.solution[-1]
        self.crop = np.array([0, 0])

    def inner_mesh_indexes(self, mesh):
        assert isinstance(mesh, MeshUniform1D)
        if mesh.is_inside_of(self):
            local_start = self.to_local_coordinate(mesh.physical_boundary_1)
            local_stop = self.to_local_coordinate(mesh.physical_boundary_2)
            idx1 = np.where(abs(self.local_nodes - local_start) < self.local_step / 2)[0][0]
            idx2 = np.where(abs(self.local_nodes - local_stop) < self.local_step / 2)[0][0]
            return [idx1, idx2]
        else:
            return [None, None]

    def is_aligned_with(self, mesh):
        assert isinstance(mesh, MeshUniform1D)
        if mesh.num < self.num:
            big_mesh = self
            small_mesh = mesh
        else:
            big_mesh = mesh
            small_mesh = self
        min_step = min(mesh.physical_step, self.physical_step)
        max_step = max(mesh.physical_step, self.physical_step)
        step_ratio = max_step / min_step
        if check_if_integer(step_ratio, 1e-8):
            shift = abs(big_mesh.physical_boundary_1 - small_mesh.physical_boundary_1) / min_step
            if check_if_integer(shift, 1e-6):
                return True
            shift = min(
                min(abs(big_mesh.physical_nodes - small_mesh.physical_boundary_1) / min_step),
                min(abs(big_mesh.physical_nodes - small_mesh.physical_boundary_2) / min_step),
                min(abs(small_mesh.physical_nodes - big_mesh.physical_boundary_1) / min_step),
                min(abs(small_mesh.physical_nodes - big_mesh.physical_boundary_2) / min_step)
            )
            if check_if_integer(shift, 1e-6):
                return True
            shifts = []
            for i in range(small_mesh.num):
                shift = min(abs(big_mesh.physical_nodes - small_mesh.physical_nodes[i]) / min_step)
                shifts.append(shift)
                if check_if_integer(shift, 1e-6):
                    return True
            shift = min(shifts)
            print('SHIFT', shift)
            if abs(round(shift)-shift) / shift < 1e-3:
                return True
            return False
        else:
            print(abs(m.floor(step_ratio) - step_ratio))
            return False

    def merge_with(self, mesh):
        assert isinstance(mesh, MeshUniform1D)
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
