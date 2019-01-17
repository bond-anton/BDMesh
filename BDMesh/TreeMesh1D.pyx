from __future__ import division, print_function
import numpy as np

from cython import boundscheck, wraparound

from .Mesh1D cimport Mesh1D
from ._helpers cimport interp_1d


cdef class TreeMesh1D(object):
    """
    Manages the tree of meshes
    """

    def __init__(self, Mesh1D root_mesh):
        """
        Tree constructor
        :param root_mesh: this mesh is a root of the tree
        """
        self.__tree = {0: [root_mesh]}

    @property
    def tree(self):
        return self.__tree

    @property
    def levels(self):
        return list(self.__tree.keys())

    @property
    def root_mesh(self):
        return self.__tree[0][0]

    @wraparound(False)
    cpdef bint add_mesh(self, Mesh1D mesh, int level=0):
        cdef:
            list levels = list(self.__tree.keys())
        if level in levels:
            self.__tree[level].append(mesh)
        else:
            self.__tree[level] = [mesh]
        self.cleanup()
        return True

    @boundscheck(False)
    @wraparound(False)
    cpdef int get_mesh_level(self, Mesh1D mesh):
        cdef:
            int level
            Mesh1D tree_mesh
        for level in self.__tree.keys():
            for tree_mesh in self.__tree[level]:
                if tree_mesh == mesh:
                    return level
        return -1

    @boundscheck(False)
    @wraparound(False)
    cpdef dict get_children(self, Mesh1D mesh):
        cdef:
            dict children = {}
            int level = self.get_mesh_level(mesh)
            long[:] upper_levels
            int upper_level
            Mesh1D tree_mesh
            list levels
        upper_levels = np.array(self.levels)[np.where(np.array(self.levels) > level)]
        for upper_level in upper_levels:
            for tree_mesh in self.tree[upper_level]:
                if tree_mesh.is_inside_of(mesh):
                    if upper_level in list(children.keys()):
                        children[upper_level].append(tree_mesh)
                    else:
                        children[upper_level] = [tree_mesh]
        return children

    @boundscheck(False)
    @wraparound(False)
    cpdef bint del_mesh(self, Mesh1D mesh):
        cdef:
            dict children
            int level = self.get_mesh_level(mesh)
            int child_level
            Mesh1D child
        if level > 0:
            children = self.get_children(mesh)
            if children == {}:
                self.__tree[level].remove(mesh)
            else:
                for child_level in sorted(children.keys(), reverse=True):
                    for child in children[child_level]:
                        self.del_mesh(child)
                self.del_mesh(mesh)
            self.cleanup()
            return True
        elif level == 0:
            return False
        else:
            return False

    @boundscheck(False)
    @wraparound(False)
    cpdef void remove_coarse_duplicates(self):
        cdef:
            int level, upper_level
            long[:] upper_levels
            Mesh1D mesh, tree_mesh
            bint mesh_removed
        for level in self.levels:
            for mesh in self.__tree[level]:
                mesh_removed = False
                upper_levels = np.array(self.levels)[np.where(np.array(self.levels) > level)]
                for upper_level in upper_levels:
                    for tree_mesh in self.__tree[upper_level]:
                        if mesh.is_inside_of(tree_mesh):
                            self.__tree[level].remove(mesh)
                            mesh_removed = True
                            break
                    if mesh_removed:
                        break
        self.recalculate_levels()

    @boundscheck(False)
    @wraparound(False)
    cpdef void recalculate_levels(self):
        cdef:
            int level, offset
        for level in self.levels:
            if not self.tree[level]:
                self.tree.pop(level)
        offset = min(self.levels)
        if offset != 0:
            for level in self.levels:
                self.tree[level - offset] = self.tree.pop(level)

    @boundscheck(False)
    @wraparound(False)
    cpdef void merge_overlaps_at_level(self, int level):
        cdef:
            bint overlap_found = True
            list i_list
        while overlap_found:
            overlap_found = False
            i_list = []
            for i in range(len(self.tree[level])):
                i_list.append(i)
                for j in range(len(self.tree[level])):
                    if j not in i_list:
                        if self.__tree[level][i].overlap_with(self.tree[level][j]):
                            overlap_found = True
                            self.tree[level][i].merge_with(self.tree[level][j])
                            self.tree[level].pop(j)
                            break
                if overlap_found:
                    break

    @boundscheck(False)
    @wraparound(False)
    cpdef void merge_overlaps(self):
        cdef:
            int level
        for level in self.levels:
            self.merge_overlaps_at_level(level)

    cpdef void cleanup(self):
        self.merge_overlaps()
        self.recalculate_levels()

    @boundscheck(False)
    @wraparound(False)
    cpdef Mesh1D flatten(self):
        cdef:
            Mesh1D flattened_mesh, mesh
            int level
        flattened_mesh = Mesh1D(self.root_mesh.physical_boundary_1, self.root_mesh.physical_boundary_2,
                                boundary_condition_1=self.root_mesh.boundary_condition_1,
                                boundary_condition_2=self.root_mesh.boundary_condition_2)
        flattened_mesh.local_nodes = self.root_mesh.local_nodes
        flattened_mesh.solution = self.root_mesh.solution
        flattened_mesh.residual = self.root_mesh.residual
        for level in self.levels[1:]:
            for mesh in self.__tree[level]:
                flattened_mesh.merge_with(mesh, threshold=1.0e-10, self_priority=False)
        return flattened_mesh

    cpdef double[:] interpolate_solution(self, double[:] phys_nodes):
        cdef:
            Mesh1D flattened = self.flatten()
        return interp_1d(phys_nodes, flattened.to_physical(flattened.__local_nodes), flattened.__solution)

    cpdef double[:] interpolate_residual(self, double[:] phys_nodes):
        cdef:
            Mesh1D flattened = self.flatten()
        return interp_1d(phys_nodes, flattened.to_physical(flattened.__local_nodes), flattened.__residual)
