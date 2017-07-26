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
        with self.assertRaises(ValueError):
            self.tree.add_mesh(mesh=mesh1, level=1.5)
        with self.assertRaises(AssertionError):
            self.tree.add_mesh(mesh=mesh1, level='1')
        with self.assertRaises(AssertionError):
            self.tree.add_mesh(mesh='a', level=1)
