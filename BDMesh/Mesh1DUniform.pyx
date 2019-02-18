from cython import boundscheck, wraparound

from cpython.object cimport Py_EQ, Py_NE
from cpython.array cimport array, clone

from libc.math cimport ceil, round, fabs, fmin, fmax
from .Mesh1D cimport Mesh1D
from ._helpers cimport check_if_integer_c, interp_1d, linspace, allclose


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
            self.__num = int(round(abs(physical_boundary_2 - physical_boundary_1) / physical_step)) + 1
        self.__local_nodes = linspace(0.0, 1.0, num=self.__num)
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
        self.__solution = clone(array('d'), self.__num, zero=True)
        self.__residual = clone(array('d'), self.__num, zero=True)

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
                            if allclose(x.local_nodes, y.local_nodes):
                                return True
            return False
        elif op == Py_NE:
            if isinstance(x, Mesh1DUniform) and isinstance(y, Mesh1DUniform):
                if x.physical_boundary_1 == y.physical_boundary_1:
                    if x.physical_boundary_2 == y.physical_boundary_2:
                        if x.local_nodes.size == y.local_nodes.size:
                            if allclose(x.local_nodes, y.local_nodes):
                                return False
            return True
        else:
            return False

    @property
    def num(self):
        return self.__num

    @num.setter
    def num(self, int num):
        cdef:
            double[:] old_nodes = self.to_physical_coordinate(self.__local_nodes)
            double[:] new_nodes
        if num < 2:
            self.__num = 2
        else:
            self.__num = num
        self.__local_nodes = linspace(0.0, 1.0, num=self.__num)
        new_nodes = self.to_physical_coordinate(self.__local_nodes)
        self.__solution = interp_1d(new_nodes, old_nodes, self.__solution)
        self.__residual = interp_1d(new_nodes, old_nodes, self.__residual)

    cdef double __calc_local_step(self) nogil:
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

    cdef double __calc_physical_step(self) nogil:
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
            self.num = int(round(self.j() / physical_step)) + 1

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

    @boundscheck(False)
    @wraparound(False)
    cpdef void trim(self):
        cdef:
            double step = self.__calc_physical_step()
        self.__solution = self.__solution[self.__crop[0]:self.__num - self.__crop[1]]
        self.__residual = self.__residual[self.__crop[0]:self.__num - self.__crop[1]]
        self.__physical_boundary_1 += self.__crop[0] * step
        self.__physical_boundary_2 -= self.__crop[1] * step
        self.__num = int(ceil(self.__num - self.__crop[0] - self.__crop[1]))
        self.__local_nodes = linspace(0.0, 1.0, num=self.__num)
        self.__boundary_condition_1 = self.__solution[0]
        self.__boundary_condition_2 = self.__solution[self.__num - 1]
        self.__crop[0] = 0
        self.__crop[1] = 0

    @boundscheck(False)
    @wraparound(False)
    cpdef inner_mesh_indices(self, Mesh1D mesh):
        cdef:
            int i, idx1 = -1, idx2 = -1
            double local_start, local_stop, local_step
        if mesh.is_inside_of(self):
            local_start = (mesh.__physical_boundary_1 - self.__physical_boundary_1) / self.j()
            local_stop = (mesh.__physical_boundary_2 - self.__physical_boundary_1) / self.j()
            local_step = self.__calc_local_step() / 2
            for i in range(self.__local_nodes.shape[0]):
                if fabs(self.__local_nodes[i] - local_start) <= local_step:
                    idx1 = i
                if fabs(self.__local_nodes[i] - local_stop) <= local_step:
                    idx2 = i
                if idx1 > -1 and idx2 > -1:
                    break
        return idx1, idx2

    cpdef bint is_aligned_with(self,  Mesh1DUniform mesh):
        cdef:
            double min_step, max_step, step_ratio, shift
            double threshold = 1.0e-8
        min_step = fmin(mesh.physical_step, self.physical_step)
        max_step = fmax(mesh.physical_step, self.physical_step)
        step_ratio = max_step / min_step
        if check_if_integer_c(step_ratio, &threshold):
            shift = fabs(self.physical_boundary_1 - mesh.physical_boundary_1) / min_step
            if check_if_integer_c(shift, &threshold):
                return True
            return False
        else:
            return False

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
            double inner_pb1, inner_pb2, new_pb1, new_pb2, new_bc1, new_bc2
            double physical_step = self.physical_step
            double[:] new_sol, new_res
            int new_num, id1_1, id1_2, id2_1, id2_2, new_id1, new_id2
            int[2] new_crop = [0, 0]
            array[double] template = array('d')
        if not self.overlap_with(other):
            return False
        if not self.is_aligned_with(other):
            return False
        if abs(physical_step - other.physical_step) > threshold:
            return False
        if self.__physical_boundary_1 < other.__physical_boundary_1:
            inner_pb1 = other.__physical_boundary_1
            new_pb1 = self.__physical_boundary_1
            new_bc1 = self.__boundary_condition_1
            new_crop[0] = self.__crop[0]
            id2_1 = 0
            id1_1 = int(round((inner_pb1 - new_pb1) / physical_step))
            new_id1 = id1_1
        else:
            inner_pb1 = self.__physical_boundary_1
            new_pb1 = other.__physical_boundary_1
            new_bc1 = other.__boundary_condition_1
            new_crop[0] = other.crop[0]
            id1_1 = 0
            id2_1 = int(round((inner_pb1 - new_pb1) / physical_step))
            new_id1 = id2_1
        if self.__physical_boundary_1 == other.__physical_boundary_1:
            new_crop[0] = min(self.__crop[0], other.crop[0])
        if self.__physical_boundary_2 > other.__physical_boundary_2:
            inner_pb2 = other.__physical_boundary_2
            new_pb2 = self.__physical_boundary_2
            new_bc2 = self.__boundary_condition_2
            new_crop[1] = self.__crop[1]
            id2_2 = other.num - 1
            id1_2 = self.num - int(round((new_pb2 - inner_pb2) / physical_step)) - 1
        else:
            inner_pb2 = self.__physical_boundary_2
            new_pb2 = other.__physical_boundary_2
            new_bc2 = other.__boundary_condition_2
            new_crop[1] = other.crop[1]
            id1_2 = self.num - 1
            id2_2 = other.num - int(round((new_pb2 - inner_pb2) / physical_step)) - 1
        if self.__physical_boundary_2 == other.__physical_boundary_2:
            new_crop[1] = min(self.__crop[1], other.crop[1])
        new_id2 = new_id1 + int(round((inner_pb2 - inner_pb1) / physical_step))
        new_num = int(round((new_pb2 - new_pb1) / physical_step)) + 1
        new_sol = clone(template, new_num, zero=False)
        new_res = clone(template, new_num, zero=False)
        if self_priority:
            new_sol[new_id1:new_id2] = self.__solution[id1_1:id1_2]
            new_res[new_id1:new_id2] = self.__residual[id1_1:id1_2]
            if new_id1 > 0:
                if id1_1 > 0:
                    new_sol[0:new_id1] = self.__solution[0:id1_1]
                    new_res[0:new_id1] = self.__residual[0:id1_1]
                else:
                    new_sol[0:new_id1] = other.__solution[0:id2_1]
                    new_res[0:new_id1] = other.__residual[0:id2_1]
            if new_id2 < new_num - 1:
                if id1_2 < self.__num - 1:
                    new_sol[new_id2:] = self.__solution[id1_2:]
                    new_res[new_id2:] = self.__residual[id1_2:]
                else:
                    new_sol[new_id2:] = other.__solution[id2_2:]
                    new_res[new_id2:] = other.__residual[id2_2:]
        else:
            new_sol[new_id1:new_id2] = other.__solution[id2_1:id2_2]
            new_res[new_id1:new_id2] = other.__residual[id2_1:id2_2]
            if new_id1 > 0:
                if id2_1 > 0:
                    new_sol[0:new_id1] = other.__solution[0:id2_1]
                    new_res[0:new_id1] = other.__residual[0:id2_1]
                else:
                    new_sol[0:new_id1] = self.__solution[0:id1_1]
                    new_res[0:new_id1] = self.__residual[0:id1_1]
            if new_id2 < new_num - 1:
                if id2_2 < self.__num - 1:
                    new_sol[new_id2:] = other.__solution[id2_2:]
                    new_res[new_id2:] = other.__residual[id2_2:]
                else:
                    new_sol[new_id2:] = self.__solution[id1_2:]
                    new_res[new_id2:] = self.__residual[id1_2:]
        self.__num = new_num
        self.__local_nodes = linspace(0.0, 1.0, new_num)
        self.__physical_boundary_1 = new_pb1
        self.__physical_boundary_2 = new_pb2
        self.__boundary_condition_1 = new_bc1
        self.__boundary_condition_2 = new_bc2
        self.__solution = new_sol
        self.__residual = new_res
        self.__crop = new_crop
        return True
