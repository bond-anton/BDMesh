from __future__ import division, print_function
import numpy as np

from BDMesh import Mesh1D
from ._helpers import check_if_integer


class TreeMesh1D(object):
    """
    Manages the tree of meshes
    """

    def __init__(self, root_mesh):
        """
        Tree constructor
        :param root_mesh: this mesh is a root of the tree
        """
        assert isinstance(root_mesh, Mesh1D)
        self.__tree = {0: [root_mesh]}

    @property
    def tree(self):
        return self.__tree

    @property
    def levels(self):
        return list(self.tree.keys())

    @property
    def root_mesh(self):
        return self.tree[0][0]

    def add_mesh(self, mesh, level):
        assert isinstance(mesh, Mesh1D)
        assert isinstance(level, (float, int))
        if not check_if_integer(level, 1e-10):
            raise ValueError('level must be integer')
        try:
            self.tree[int(level)].append(mesh)
        except KeyError:
            self.tree[int(level)] = [mesh]
        self.cleanup()

    def get_mesh_level(self, mesh):
        for level in self.tree.keys():
            for tree_mesh in self.tree[level]:
                if tree_mesh == mesh:
                    return level
        print('mesh not found in a tree')
        return -1

    def get_children(self, mesh):
        children = {}
        level = self.get_mesh_level(mesh)
        upper_levels = np.array(self.tree.keys())
        upper_levels = upper_levels[np.where(upper_levels > level)]
        for upper_level in upper_levels:
            for tree_mesh in self.tree[upper_level]:
                if tree_mesh.is_inside_of(mesh):
                    try:
                        children[upper_level].append(tree_mesh)
                    except KeyError:
                        children[upper_level] = [tree_mesh]
        return children

    def del_mesh(self, mesh, del_children=True):
        level = self.get_mesh_level(mesh)
        if level > 0:
            children = self.get_children(mesh)
            if children == {}:
                self.tree[level].remove(mesh)
            elif del_children:
                for child_level in sorted(children.keys(), reverse=True):
                    print('deleting children at level', child_level)
                    for child in children[child_level]:
                        self.del_mesh(child, del_children=False)
                self.del_mesh(mesh, del_children=False)
            else:
                print('mesh has children, use del_children=True flag')
        elif level == 0:
            print('Can not delete root mesh')
        else:
            print('mesh not found in a tree')
        self.cleanup()

    def cleanup(self):
        self.merge_overlaps()
        # self.remove_coarse_duplicates()
        self.recalculate_levels()

    def remove_coarse_duplicates(self):
        for level in self.tree.keys():
            for mesh in self.tree[level]:
                upper_levels = np.array(self.tree.keys())
                upper_levels = upper_levels[np.where(upper_levels > level)]
                for upper_level in upper_levels:
                    for tree_mesh in self.tree[upper_level]:
                        if mesh.is_inside_of(tree_mesh):
                            self.tree[level].remove(mesh)
                            print('mesh overlaps with coarse mesh. DELETING COARSE.')

    def recalculate_levels(self):
        tidy = False
        while not tidy:
            for level in self.tree.keys():
                if not self.tree[level]:
                    self.tree.pop(level)
                    break
            tidy = True
        offset = min(self.tree.keys())
        if offset != 0:
            for level in self.tree.keys():
                self.tree[level - offset] = self.tree.pop(level)

    def merge_overlaps(self):
        for level in self.levels:
            overlap_found = True
            while overlap_found:
                overlap_found = False
                i_list = []
                for i in range(len(self.tree[level])):
                    i_list.append(i)
                    for j in range(len(self.tree[level])):
                        if j not in i_list:
                            if self.tree[level][i].overlap_with(self.tree[level][j]):
                                overlap_found = True
                                self.tree[level][i].merge_with(self.tree[level][j])
                                self.tree[level].remove(self.tree[level][j])
                                break
                    if overlap_found:
                        break
