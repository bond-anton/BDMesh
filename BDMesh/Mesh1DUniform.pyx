from __future__ import division, print_function
import numpy as np

from cpython.object cimport Py_EQ, Py_NE

from libc.math cimport floor, ceil
from .Mesh1D cimport Mesh1D
from ._helpers cimport check_if_integer_c


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
        self.__solution = np.zeros(self.__num, dtype=np.double)
        self.__residual = np.zeros(self.__num, dtype=np.double)

    def __str__(self):
        return 'Mesh1DUniform: [%2.2g; %2.2g], %2.2g step, %d nodes' % (self.physical_boundary_1,
                                                                        self.physical_boundary_2,
                                                                        self.physical_step,
                                                                        self.num)

    def __richcmp__(x, y, int op):
        if op == Py_EQ:
            if isinstance(x, Mesh1DUniform) and isinstance(y, Mesh1DUniform):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if np.allclose(x.local_nodes, y.local_nodes):
                                return True
            return False
        elif op == Py_NE:
            if isinstance(x, Mesh1DUniform) and isinstance(y, Mesh1DUniform):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if np.allclose(x.local_nodes, y.local_nodes):
                                return False
            return True
        else:
            return False

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

    cdef double __calc_local_step(self):
        return 1.0 / (self.__num - 1)

    @property
    def local_step(self):
        return self.__calc_local_step()

    @local_step.setter
    def local_step(self, double local_step):
        if local_step > 1:
            self.num = 2
        elif local_step <= 0:
            self.num = 2
        else:
            self.num = int(1.0 / local_step) + 1

    cdef double __calc_physical_step(self):
        return self.__calc_local_step() * self.j()

    @property
    def physical_step(self):
        return self.__calc_physical_step()

    @physical_step.setter
    def physical_step(self, double physical_step):
        if physical_step > self.j():
            self.num = 2
        elif physical_step <= 0.0:
            self.num = 2
        else:
            self.num = int(ceil(self.j() / physical_step)) + 1

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
        elif crop[1] >= self.__num - self.__crop[0] - 2:
            self.__crop[1] = self.__num - self.__crop[0] - 2
        else:
            self.__crop[1] = int(crop[1])

    cpdef trim(self):
        cdef:
            double step = self.__calc_physical_step()
        self.__solution = self.__solution[self.__crop[0]:self.__num - self.__crop[1]]
        self.__residual = self.__residual[self.__crop[0]:self.__num - self.__crop[1]]
        self.__physical_boundary_1 += self.__crop[0] * step
        self.__physical_boundary_2 -= self.__crop[1] * step
        self.__num = int(np.ceil(self.__num - self.__crop[0] - self.__crop[1]))
        self.__local_nodes = np.linspace(0.0, 1.0, num=self.__num, endpoint=True)
        self.__boundary_condition_1 = self.__solution[0]
        self.__boundary_condition_2 = self.__solution[-1]
        self.__crop = np.array([0, 0])
    
    cpdef inner_mesh_indices(self, Mesh1D mesh):
        cdef:
            int idx1 = -1
            int idx2 = -1
            double local_start, local_stop, local_step
        if mesh.is_inside_of(self):
            local_start = self.to_local_coordinate(np.array([mesh.physical_boundary_1]))[0]
            local_stop = self.to_local_coordinate(np.array([mesh.physical_boundary_2]))[0]
            local_step = self.__calc_local_step()
            idx1 = np.where(abs(np.asarray(self.__local_nodes) - local_start) <= local_step / 2)[0][0]
            idx2 = np.where(abs(np.asarray(self.__local_nodes) - local_stop) <= local_step / 2)[0][0]
        return idx1, idx2

    cpdef bint is_aligned_with(self,  Mesh1DUniform mesh):
        cdef:
            double min_step, max_step, step_ratio, shift
            double threshold = 1.0e-8
        min_step = min(mesh.physical_step, self.physical_step)
        max_step = max(mesh.physical_step, self.physical_step)
        step_ratio = max_step / min_step
        if check_if_integer_c(step_ratio, &threshold):
            shift = abs(self.physical_boundary_1 - mesh.physical_boundary_1) / min_step
            if check_if_integer_c(shift, &threshold):
                return True
            return False
        else:
            return False

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
