from __future__ import division, print_function
import numpy as np

from cython import boundscheck, wraparound
from libc.math cimport ceil, round, log

from .Mesh1D cimport Mesh1D
from .Mesh1DUniform cimport Mesh1DUniform
from .TreeMesh1D import TreeMesh1D
from ._helpers cimport check_if_integer_c


cdef class TreeMesh1DUniform(TreeMesh1D):
    """
    Manages the tree of uniform meshes
    """

    def __init__(self, Mesh1DUniform root_mesh, int refinement_coefficient=2, bint aligned=True, crop=[0, 0]):
        """
        Constructor method
        :param root_mesh: uniform mesh which is a root of the tree
        :param refinement_coefficient: coefficient of nested meshes step refinement
        :param aligned: set to True (default) if you want nodes of nested meshes to be aligned with parent mesh
        :param crop: iterable of two integers specifying number of root_mesh nodes to crop from both side of meshes tree
        """
        super(TreeMesh1DUniform, self).__init__(root_mesh)
        self.__refinement_coefficient = refinement_coefficient
        self.__aligned = aligned
        self.__crop[0] = crop[0]
        self.__crop[1] = crop[1]

    @property
    def refinement_coefficient(self):
        return self.__refinement_coefficient

    @refinement_coefficient.setter
    def refinement_coefficient(self, int refinement_coefficient):
        cdef:
            double ratio
            int level
            Mesh1DUniform mesh
        ratio = self.refinement_coefficient / refinement_coefficient
        with boundscheck(False), wraparound(False):
            for level in self.levels[1:]:
                for mesh in self.tree[level]:
                    mesh.physical_step *= ratio
        self.__refinement_coefficient = refinement_coefficient
        self.cleanup()

    @property
    def aligned(self):
        return self.__aligned

    @aligned.setter
    def aligned(self, bint aligned):
        cdef:
            bint tree_is_aligned
            Mesh1DUniform parent, mesh
            int level
        if not aligned:
            self.__aligned = aligned
        else:
            tree_is_aligned = True
            parent = self.root_mesh
            with boundscheck(False), wraparound(False):
                for level in self.levels[1:]:
                    for mesh in self.tree[level]:
                        if not parent.is_aligned_with(mesh):
                            tree_is_aligned = False
                            break
                    if not tree_is_aligned:
                        break
            if not tree_is_aligned:
                # TODO: add some smart alignment procedure
                self.__aligned = False
            else:
                self.__aligned = aligned

    @property
    def crop(self):
        return self.__crop

    @crop.setter
    def crop(self, crop):
        if crop[0] <= 0:
            self.__crop[0] = 0
        elif crop[0] >= self.root_mesh.num:
            self.__crop[0] = self.root_mesh.num - 2
        else:
            self.__crop[0] = int(crop[0])

        if crop[1] <= 0:
            self.__crop[1] = 0
        elif crop[1] >= self.root_mesh.num - self.__crop[0] - 2:
            self.__crop[1] = self.root_mesh.num - self.__crop[0] - 2
        else:
            self.__crop[1] = int(crop[1])

    @boundscheck(False)
    @wraparound(False)
    cpdef bint add_mesh(self, Mesh1D mesh, int level=-1):
        cdef:
            double log_level, threshold = 1e-6
            int calc_level
        assert isinstance(mesh, Mesh1DUniform)
        log_level = log(self.root_mesh.physical_step / mesh.physical_step) / log(self.refinement_coefficient)
        if not check_if_integer_c(log_level, &threshold):
            return False
        calc_level = int(round(log_level))
        if self.aligned and not self.tree[calc_level - 1][0].is_aligned_with(mesh):
            return False
        return super(TreeMesh1DUniform, self).add_mesh(mesh, calc_level)

    @boundscheck(False)
    @wraparound(False)
    cpdef void trim(self):
        cdef:
            int level = 1
            bint trimmed
            list meshes_for_deletion
            Mesh1DUniform mesh
            double left_offset, right_offset
            int[2] crop
        self.root_mesh.crop = self.crop
        self.root_mesh.trim()
        trimmed = True if level > self.levels[len(self.levels) - 1] else False
        while not trimmed:
            meshes_for_deletion = []
            for mesh in self.tree[level]:
                mesh.trim()
                crop = [0, 0]
                left_offset = (self.root_mesh.physical_boundary_1 - mesh.physical_boundary_1) / mesh.physical_step
                right_offset = (mesh.physical_boundary_2 - self.root_mesh.physical_boundary_2) / mesh.physical_step
                crop[0] = int(ceil(left_offset)) if left_offset > 0 else 0
                crop[1] = int(ceil(right_offset)) if right_offset > 0 else 0
                if crop[0] + crop[1] >= mesh.num - 2:
                        meshes_for_deletion.append(mesh)
                        continue
                mesh.crop = np.array(crop)
                mesh.trim()
            for mesh in meshes_for_deletion:
                self.del_mesh(mesh)
            level += 1
            if level > self.levels[len(self.levels) - 1]:
                trimmed = True
        self.crop = [0, 0]
