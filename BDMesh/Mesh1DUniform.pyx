from __future__ import division, print_function
import numpy as np

from libc.math cimport floor, ceil
from .Mesh1D cimport Mesh1D
# from ._helpers import check_if_integer


cdef class Mesh1DUniform(Mesh1D):
    """
    One dimensional uniform mesh for boundary problems
    """

    def __init__(self, double physical_boundary_1, double physical_boundary_2,
                 double boundary_condition_1=0.0, double boundary_condition_2=0.0,
                 double physical_step=0.0, int num=2, crop=[0, 0]):
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
        if physical_step <= 0.0:
            if num <= 2:
                self.__num = 2
            else:
                self.__num = num
        else:
            self.__num = int(floor(abs(physical_boundary_2 - physical_boundary_1) / physical_step)) + 1
        self.__local_nodes = np.linspace(0.0, 1.0, num=self.__num, endpoint=True)
        if crop[0] <= 0:
            self.__crop[0] = 0
        elif crop[0] >= self.__num:
            self.__crop[0] = self.__num - 2
        else:
            self.__crop[0] = int(crop[0])
        if crop[1] <= 0:
            self.__crop[1] = 0
        elif crop[1] >= self.__num - self.__crop[0]:
            self.__crop[1] = self.__num - self.__crop[0] - 2
        else:
            self.__crop[1] = int(crop[1])

    def __str__(self):
        return 'Mesh1DUniform: [%2.2g; %2.2g], %2.2g step, %d nodes' % (self.physical_boundary_1,
                                                                        self.physical_boundary_2,
                                                                        self.physical_step,
                                                                        self.num)

    @property
    def num(self):
        return self.__num

    @num.setter
    def num(self, int num):
        if num < 2:
            self.__num = 2
        else:
            self.__num = num
        self.__local_nodes = np.linspace(0.0, 1.0, num=self.__num, endpoint=True)

    @property
    def local_step(self):
        return 1.0 / (self.__num - 1)

    @local_step.setter
    def local_step(self, double local_step):
        if local_step > 1:
            self.num = 2
        elif local_step <= 0:
            self.num = 2
        else:
            self.num = int(1.0 / local_step) + 1

    @property
    def physical_step(self):
        return self.local_step * self.jacobian

    @physical_step.setter
    def physical_step(self, double physical_step):
        if physical_step > self.jacobian:
            self.num = 2
        elif physical_step <= 0.0:
            self.num = 2
        else:
            self.num = int(ceil(self.jacobian / physical_step)) + 1

    @property
    def crop(self):
        return self.__crop

    @crop.setter
    def crop(self, crop):
        if crop[0] <= 0:
            self.__crop[0] = 0
        elif crop[0] >= self.__num:
            self.__crop[0] = self.__num - 2
        else:
            self.__crop[0] = int(crop[0])
        if crop[1] <= 0:
            self.__crop[1] = 0
        elif crop[1] >= self.__num - self.__crop[0]:
            self.__crop[1] = self.__num - self.__crop[0] - 2
        else:
            self.__crop[1] = int(crop[1])

    # def trim(self):
    #     solution = self.solution[self.crop[0]:self.num-self.crop[1]]
    #     residual = self.residual[self.crop[0]:self.num-self.crop[1]]
    #     self.physical_boundary_1 += self.crop[0] * self.physical_step
    #     self.physical_boundary_2 -= self.crop[1] * self.physical_step
    #     num_points = int(np.ceil(self.num - np.sum(self.crop)))
    #     self.local_nodes = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
    #     self.solution = solution
    #     self.residual = residual
    #     self.boundary_condition_1 = self.solution[0]
    #     self.boundary_condition_2 = self.solution[-1]
    #     self.crop = np.array([0, 0])
    #
    # def inner_mesh_indices(self, mesh):
    #     # assert isinstance(mesh, Mesh1D)
    #     if mesh.is_inside_of(self):
    #         local_start = self.to_local_coordinate(mesh.physical_boundary_1)
    #         local_stop = self.to_local_coordinate(mesh.physical_boundary_2)
    #         idx1 = np.where(abs(self.local_nodes - local_start) <= self.local_step / 2)[0][0]
    #         idx2 = np.where(abs(self.local_nodes - local_stop) <= self.local_step / 2)[0][0]
    #         return [idx1, idx2]
    #     else:
    #         return [None, None]
    #
    # def is_aligned_with(self, mesh):
    #     assert isinstance(mesh, Mesh1DUniform)
    #     min_step = min(mesh.physical_step, self.physical_step)
    #     max_step = max(mesh.physical_step, self.physical_step)
    #     step_ratio = max_step / min_step
    #     if check_if_integer(step_ratio, 1e-8):
    #         shift = abs(self.physical_boundary_1 - mesh.physical_boundary_1) / min_step
    #         if check_if_integer(shift, 1e-8):
    #             return True
    #         return False
    #     else:
    #         return False
    #
    # def merge_with(self, other, priority='self'):
    #     assert isinstance(other, Mesh1DUniform)
    #     if self.overlap_with(other):
    #         if self.is_aligned_with(other):
    #             if priority == 'self':
    #                 tmp_mesh_1 = self.copy()
    #                 tmp_mesh_2 = other.copy()
    #             elif priority == 'other':
    #                 tmp_mesh_1 = other.copy()
    #                 tmp_mesh_2 = self.copy()
    #             else:
    #                 raise ValueError('Priority must be either "self" or "other"')
    #             tmp_mesh_1.physical_step = self.physical_step
    #             tmp_mesh_2.physical_step = self.physical_step
    #             merged_physical_nodes, indices = np.unique(np.concatenate((tmp_mesh_1.physical_nodes,
    #                                                                        tmp_mesh_2.physical_nodes)).round(12),
    #                                                        return_index=True)
    #             idx_1 = indices[np.where(indices < tmp_mesh_1.num)]
    #             idx_2 = indices[np.where(indices >= tmp_mesh_1.num)] - tmp_mesh_1.num
    #             solution = np.zeros(merged_physical_nodes.size)
    #             solution[np.where(indices < tmp_mesh_1.num)] = tmp_mesh_1.solution[idx_1]
    #             solution[np.where(indices >= tmp_mesh_1.num)] = tmp_mesh_2.solution[idx_2]
    #             residual = np.zeros(merged_physical_nodes.size)
    #             residual[np.where(indices < tmp_mesh_1.num)] = tmp_mesh_1.residual[idx_1]
    #             residual[np.where(indices >= tmp_mesh_1.num)] = tmp_mesh_2.residual[idx_2]
    #
    #             if self.physical_boundary_1 + self.crop[0] > other.physical_boundary_1 + other.crop[0]:
    #                 self.boundary_condition_1 = other.boundary_condition_1
    #                 self.crop[0] = other.crop[0]
    #                 self.physical_boundary_1 = other.physical_boundary_1
    #             if self.physical_boundary_2 - self.crop[1] < other.physical_boundary_2 - other.crop[1]:
    #                 self.boundary_condition_2 = other.boundary_condition_2
    #                 self.crop[1] = other.crop[1]
    #                 self.physical_boundary_2 = other.physical_boundary_2
    #             self.physical_step = min(self.physical_step, other.physical_step)
    #             self.solution = np.interp(self.physical_nodes, merged_physical_nodes, solution)
    #             self.residual = np.interp(self.physical_nodes, merged_physical_nodes, residual)
    #         else:
    #             raise ValueError('meshes are not aligned')
    #     else:
    #         raise ValueError('meshes do not overlap')
