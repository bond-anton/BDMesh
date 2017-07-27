from __future__ import division, print_function
from copy import copy, deepcopy
import math as m
import numpy as np
import unittest

from BDMesh import Mesh1D, TreeMesh1D


class TestTreeMesh1D(unittest.TestCase):

    def setUp(self):
        self.root_mesh = Mesh1D(0.0, 10.0)
        self.tree = TreeMesh1D(self.root_mesh)

    def test_constructor(self):
        with self.assertRaises(AssertionError):
            TreeMesh1D(1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh]})
        self.assertEqual(self.tree.levels, [0])
        self.assertEqual(self.tree.root_mesh, self.root_mesh)

    def test_add_mesh(self):
        # adding not overlapping meshes
        mesh1 = Mesh1D(5, 15)
        self.tree.add_mesh(mesh=mesh1, level=1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh1]})
        self.assertEqual(self.tree.levels, [0, 1])
        mesh2 = Mesh1D(1, 4)
        self.tree.add_mesh(mesh=mesh2, level=1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh1, mesh2]})
        self.assertEqual(self.tree.levels, [0, 1])
        mesh3 = Mesh1D(6, 9)
        self.tree.add_mesh(mesh=mesh3, level=5)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh1, mesh2], 5:[mesh3]})
        self.assertEqual(self.tree.levels, [0, 1, 5])
        # adding overlapping meshes
        mesh4 = Mesh1D(12, 17)
        self.tree.add_mesh(mesh=mesh4, level=1)
        mesh4.merge_with(mesh1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh4, mesh2], 5: [mesh3]})
        mesh4 = Mesh1D(3, 6)
        self.tree.add_mesh(mesh=mesh4, level=1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh1], 5: [mesh3]})
        self.assertEqual(self.tree.levels, [0, 1, 5])
        self.assertEqual(mesh1.physical_boundary_1, 1)
        self.assertEqual(mesh1.physical_boundary_2, 17)
        # testing exceptions
        with self.assertRaises(ValueError):
            self.tree.add_mesh(mesh=mesh1, level=1.5)
        with self.assertRaises(AssertionError):
            self.tree.add_mesh(mesh=mesh1, level='1')
        with self.assertRaises(AssertionError):
            self.tree.add_mesh(mesh='a', level=1)
        # test mesh level search
        self.assertEqual(self.tree.get_mesh_level(self.root_mesh), 0)
        self.assertEqual(self.tree.get_mesh_level(mesh1), 1)
        self.assertEqual(self.tree.get_mesh_level(mesh4), -1)
        with self.assertRaises(AssertionError):
            self.tree.get_mesh_level(2)

    def test_get_children(self):
        mesh = Mesh1D(1, 7)
        self.tree.add_mesh(mesh=mesh, level=1)
        self.tree.add_mesh(mesh=Mesh1D(1, 6), level=2)
        self.tree.add_mesh(mesh=Mesh1D(2, 3), level=3)
        self.tree.add_mesh(mesh=Mesh1D(4, 5), level=3)
        children = self.tree.get_children(mesh)
        self.assertEqual(children, {2: [Mesh1D(1, 6)], 3: [Mesh1D(2, 3), Mesh1D(4, 5)]})
        self.tree.add_mesh(mesh=Mesh1D(4.5, 7.5), level=3)
        children = self.tree.get_children(mesh)
        self.assertEqual(children, {2: [Mesh1D(1, 6)], 3: [Mesh1D(2, 3)]})
        # testing exceptions
        with self.assertRaises(AssertionError):
            self.tree.get_children(mesh='a')

    def test_delete(self):
        mesh = Mesh1D(1, 7)
        self.tree.add_mesh(mesh=mesh, level=1)
        self.tree.add_mesh(mesh=Mesh1D(1, 6), level=2)
        self.tree.add_mesh(mesh=Mesh1D(2, 3), level=3)
        self.tree.add_mesh(mesh=Mesh1D(4, 5), level=3)
        children = self.tree.get_children(mesh)
        self.assertEqual(children, {2: [Mesh1D(1, 6)], 3: [Mesh1D(2, 3), Mesh1D(4, 5)]})
        self.tree.del_mesh(Mesh1D(4, 5))
        children = self.tree.get_children(mesh)
        self.assertEqual(children, {2: [Mesh1D(1, 6)], 3: [Mesh1D(2, 3)]})
        self.tree.add_mesh(mesh=Mesh1D(4, 5), level=3)
        children = self.tree.get_children(mesh)
        self.assertEqual(children, {2: [Mesh1D(1, 6)], 3: [Mesh1D(2, 3), Mesh1D(4, 5)]})
        self.tree.del_mesh(Mesh1D(1, 6))
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [Mesh1D(1, 7)]})
        # check if exceptions are raised
        with self.assertRaises(ValueError):
            self.tree.del_mesh(self.root_mesh)
        with self.assertRaises(ValueError):
            self.tree.del_mesh(Mesh1D(100, 110))
        with self.assertRaises(AssertionError):
            self.tree.del_mesh(1)

    def test_remove_coarse_duplicates(self):
        mesh = Mesh1D(1, 9)
        self.tree.add_mesh(mesh=mesh, level=1)
        self.tree.add_mesh(mesh=Mesh1D(1, 7), level=2)
        self.tree.add_mesh(mesh=Mesh1D(1, 7), level=3)
        self.tree.add_mesh(mesh=Mesh1D(8, 9), level=3)
        children = self.tree.get_children(mesh)
        self.assertEqual(children, {2: [Mesh1D(1, 7)], 3: [Mesh1D(1, 7), Mesh1D(8, 9)]})
        self.tree.remove_coarse_duplicates()
        children = self.tree.get_children(mesh)
        print(self.tree.tree)
        self.assertEqual(children, {3: [Mesh1D(1, 7), Mesh1D(8, 9)]})
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh], 3: [Mesh1D(1, 7), Mesh1D(8, 9)]})
