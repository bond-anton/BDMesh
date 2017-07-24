from __future__ import division, print_function
import math as m
import numpy as np

from BDMesh import TreeMesh1D, MeshUniform1D
from ._helpers import check_if_integer


class TreeMeshUniform1D(TreeMesh1D):
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
        assert isinstance(root_mesh, MeshUniform1D)
        self.__refinement_coefficient = None
        self.__aligned = None
        self.__crop = None
        super(TreeMeshUniform1D, self).__init__(root_mesh)
        self.refinement_coefficient = refinement_coefficient
        self.aligned = aligned
        self.crop = crop

    @property
    def refinement_coefficient(self):
        return self.__refinement_coefficient

    @refinement_coefficient.setter
    def refinement_coefficient(self, refinement_coefficient):
        assert isinstance(refinement_coefficient, (float, int))
        self.__refinement_coefficient = refinement_coefficient

    @property
    def aligned(self):
        return self.__aligned

    @aligned.setter
    def aligned(self, aligned):
        assert isinstance(aligned, bool)
        self.__aligned = aligned

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

    def add_mesh(self, mesh, **kwargs):
        assert isinstance(mesh, MeshUniform1D)
        level = m.log(self.root_mesh.physical_step / mesh.physical_step, self.refinement_coefficient)
        lower_estimation = m.floor(level)
        upper_estimation = m.ceil(level)
        if abs(lower_estimation - level) < abs(upper_estimation - level):
            level = int(lower_estimation)
        else:
            level = int(upper_estimation)
        if self.aligned and not self.tree[level-1][0].is_aligned_with(mesh):
            raise Exception('all child meshes must be aligned with the root mesh')
        super(TreeMeshUniform1D, self).add_mesh(mesh, level)

    def trim(self, debug=False):
        if debug:
            print('Cropping', np.sum(self.crop), 'elements (of root_mesh):', self.crop)
        self.root_mesh.crop = self.crop
        self.root_mesh.trim()
        level = 1
        trimmed = True if level > self.levels[-1] else False
        while not trimmed:
            if debug:
                print('trimming level', level)
            meshes_for_delete = []
            for mesh in self.tree[level]:
                mesh.trim()
                crop = [0, 0]
                left_offset = (self.root_mesh.physical_boundary_1 - mesh.physical_boundary_1) / mesh.physical_step
                right_offset = (mesh.physical_boundary_2 - self.root_mesh.physical_boundary_2) / mesh.physical_step
                crop[0] = int(m.ceil(left_offset)) if left_offset > 0 else 0
                crop[1] = int(m.ceil(right_offset)) if right_offset > 0 else 0
                if crop[0] == 0 and crop[1] > 0:
                    if crop[1] >= mesh.num:
                        if debug:
                            print('Deleting mesh')
                        meshes_for_delete.append(mesh)
                        continue
                elif crop[1] == 0 and crop[0] > 0:
                    if crop[0] >= mesh.num:
                        if debug:
                            print('Deleting mesh')
                        meshes_for_delete.append(mesh)
                        continue
                if debug:
                    print('Cropping mesh by', crop)
                mesh.crop = np.array(crop)
                mesh.trim()
            for mesh in meshes_for_delete:
                self.del_mesh(mesh, del_children=True)
            level += 1
            if level > self.levels[-1]:
                trimmed = True
        self.crop = [0, 0]

    def flatten(self, debug=False):
        flat_grid = self.root_mesh.physical_nodes
        flat_sol = self.root_mesh.solution
        flat_res = self.root_mesh.residual
        if debug:
            print('root_mesh is from', flat_grid[0], 'to', flat_grid[-1])
        for level in self.levels[1:]:
            if debug:
                print('working with level', level)
            for mesh in self.tree[level]:
                if debug:
                    print('flat_grid is from', flat_grid[0], 'to', flat_grid[-1])
                    print('merging mesh from', mesh.physical_boundary_1,)
                    print('to', mesh.physical_boundary_2, mesh.physical_step)
                ins_idx1 = np.where(flat_grid <= mesh.physical_boundary_1 + mesh.physical_step / 10)[0][-1]
                ins_idx2 = np.where(flat_grid >= mesh.physical_boundary_2 - mesh.physical_step / 10)[0][0]
                if ins_idx2 == flat_grid.size:
                    flat_grid = np.hstack((flat_grid[0:ins_idx1], mesh.physical_nodes))
                    flat_sol = np.hstack((flat_sol[0:ins_idx1], mesh.solution))
                    flat_res = np.hstack((flat_res[0:ins_idx1], mesh.residual))
                else:
                    if ins_idx2 < flat_grid.size - 1:
                        ins_idx2 += 1
                    if len(flat_grid[ins_idx2:]) == 1 and flat_grid[ins_idx2] == mesh.physical_nodes[-1]:
                        flat_grid = np.hstack((flat_grid[0:ins_idx1], mesh.physical_nodes))
                        flat_sol = np.hstack((flat_sol[0:ins_idx1], mesh.solution))
                        flat_res = np.hstack((flat_res[0:ins_idx1], mesh.residual))
                    else:
                        flat_grid = np.hstack((flat_grid[0:ins_idx1], mesh.physical_nodes, flat_grid[ins_idx2:]))
                        flat_sol = np.hstack((flat_sol[0:ins_idx1], mesh.solution, flat_sol[ins_idx2:]))
                        flat_res = np.hstack((flat_res[0:ins_idx1], mesh.residual, flat_res[ins_idx2:]))
                if debug:
                    print('flat_grid is from', flat_grid[0], 'to', flat_grid[-1])
        return flat_grid, flat_sol, flat_res
