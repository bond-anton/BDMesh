from __future__ import division, print_function
import math as m
import numpy as np

from BDMesh import TreeMesh1D, Mesh1DUniform
from ._helpers import check_if_integer


class TreeMesh1DUniform(TreeMesh1D):
    """
    Manages the tree of uniform meshes
    """

    def __init__(self, root_mesh, refinement_coefficient=2, aligned=True, crop=None):
        """
        Constructor method
        :param root_mesh: uniform mesh which is a root of the tree
        :param refinement_coefficient: coefficient of nested meshes step refinement
        :param aligned: set to True (default) if you want nodes of nested meshes to be aligned with parent mesh
        :param crop: iterable of two integers specifying number of root_mesh nodes to crop from both side of meshes tree
        """
        assert isinstance(root_mesh, Mesh1DUniform)
        super(TreeMesh1DUniform, self).__init__(root_mesh)
        self.__refinement_coefficient = None
        self.__aligned = None
        self.__crop = None
        self.refinement_coefficient = refinement_coefficient
        self.aligned = aligned
        self.crop = crop

    @property
    def refinement_coefficient(self):
        return self.__refinement_coefficient

    @refinement_coefficient.setter
    def refinement_coefficient(self, refinement_coefficient):
        assert isinstance(refinement_coefficient, (float, int))
        # TODO: recalculate all meshes if refinement coefficient changed. For now just forbid to change.
        if self.refinement_coefficient is None:
            self.__refinement_coefficient = refinement_coefficient
        else:
            raise NotImplementedError('Change of refinement coefficient is not implemented yet.')

    @property
    def aligned(self):
        return self.__aligned

    @aligned.setter
    def aligned(self, aligned):
        assert isinstance(aligned, bool)
        # TODO: if True, add check whether meshes are actually aligned. For now forbid to change to True
        if self.aligned is None or not aligned:
            self.__aligned = aligned
        else:
            raise NotImplementedError('Change of aligned flag to True is not implemented yet.')

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
                if np.sum(crop) > self.root_mesh.num - 2:
                    raise ValueError('At least two nodes must remain after trimming')
                if (crop < 0).any():
                    raise ValueError('crop positions must be greater than zero')
                self.__crop = np.array([int(crop[0]), int(crop[1])], dtype=np.int)

    def add_mesh(self, mesh, **kwargs):
        assert isinstance(mesh, Mesh1DUniform)
        level = m.log(self.root_mesh.physical_step / mesh.physical_step, self.refinement_coefficient)
        if not check_if_integer(level, threshold=1e-10):
            raise ValueError('refinement coefficient rule is violated')
        level = int(level)
        if self.aligned and not self.tree[level-1][0].is_aligned_with(mesh):
            raise ValueError('all child meshes must be aligned with the root mesh')
        super(TreeMesh1DUniform, self).add_mesh(mesh, level)

    def trim(self):
        self.root_mesh.crop = self.crop
        self.root_mesh.trim()
        level = 1
        trimmed = True if level > self.levels[-1] else False
        while not trimmed:
            meshes_for_deletion = []
            for mesh in self.tree[level]:
                mesh.trim()
                crop = [0, 0]
                left_offset = (self.root_mesh.physical_boundary_1 - mesh.physical_boundary_1) / mesh.physical_step
                right_offset = (mesh.physical_boundary_2 - self.root_mesh.physical_boundary_2) / mesh.physical_step
                crop[0] = int(m.ceil(left_offset)) if left_offset > 0 else 0
                crop[1] = int(m.ceil(right_offset)) if right_offset > 0 else 0
                if crop[0] + crop[1] >= mesh.num - 2:
                        meshes_for_deletion.append(mesh)
                        continue
                mesh.crop = np.array(crop)
                mesh.trim()
            for mesh in meshes_for_deletion:
                self.del_mesh(mesh)
            level += 1
            if level > self.levels[-1]:
                trimmed = True
        self.crop = [0, 0]
