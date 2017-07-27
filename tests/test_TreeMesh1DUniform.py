from __future__ import division, print_function
from copy import copy, deepcopy
import math as m
import numpy as np
import unittest

from BDMesh import Mesh1DUniform, TreeMesh1DUniform


class TestTreeMesh1DUniform(unittest.TestCase):

    def setUp(self):
        self.root_mesh = Mesh1DUniform(0.0, 10.0, physical_step=1.0)
        self.tree = TreeMesh1DUniform(self.root_mesh, refinement_coefficient=2, aligned=True, crop=None)

    def test_constructor(self):
        with self.assertRaises(AssertionError):
            TreeMesh1DUniform(1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh]})
        self.assertEqual(self.tree.levels, [0])
        self.assertEqual(self.tree.root_mesh, self.root_mesh)
        self.assertEqual(self.tree.refinement_coefficient, 2)
        self.assertTrue(self.tree.aligned)
        np.testing.assert_equal(self.tree.crop, np.array([0, 0]))

    def test_refinement_coefficient(self):
        self.assertEqual(self.tree.refinement_coefficient, 2)
        with self.assertRaises(NotImplementedError):
            self.tree.refinement_coefficient = 4
        with self.assertRaises(AssertionError):
            self.tree.refinement_coefficient = 'a'

    def test_aligned(self):
        self.assertTrue(self.tree.aligned)
        self.tree.aligned = False
        self.assertFalse(self.tree.aligned)
        with self.assertRaises(NotImplementedError):
            self.tree.aligned = True
        self.assertFalse(self.tree.aligned)
        with self.assertRaises(AssertionError):
            self.tree.aligned = 'a'

    def test_crop(self):
        np.testing.assert_equal(self.tree.crop, np.array([0, 0]))
        self.tree.crop = [3.0, 2.0]
        np.testing.assert_equal(self.tree.crop, np.array([3, 2]))
        with self.assertRaises(TypeError):
            self.tree.crop = 1
        with self.assertRaises(ValueError):
            self.tree.crop = 'ab'
        with self.assertRaises(ValueError):
            self.tree.crop = 'abc'
        with self.assertRaises(ValueError):
            self.tree.crop = [1, 2, 3]
        with self.assertRaises(ValueError):
            self.tree.crop = [5, 5]
        with self.assertRaises(ValueError):
            self.tree.crop = [-1, 1]
        with self.assertRaises(ValueError):
            self.tree.crop = [1, -1]
        self.tree.crop = (4.0, 1)
        np.testing.assert_equal(self.tree.crop, np.array([4, 1]))
        self.tree.crop = np.array([3.0, 3])
        np.testing.assert_equal(self.tree.crop, np.array([3, 3]))

    def test_add_mesh(self):
        # adding not overlapping meshes
        mesh = Mesh1DUniform(1, 8, physical_step=0.5)
        self.tree.add_mesh(mesh=mesh)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh]})
        mesh1 = Mesh1DUniform(2, 10, physical_step=0.25)
        self.tree.add_mesh(mesh=mesh1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh], 2:[mesh1]})
        # testing exceptions
        with self.assertRaises(ValueError):
            self.tree.add_mesh(mesh=Mesh1DUniform(1.3, 8, physical_step=0.5))
        with self.assertRaises(ValueError):
            self.tree.add_mesh(mesh=Mesh1DUniform(1.3, 8, physical_step=0.33))
        with self.assertRaises(AssertionError):
            self.tree.add_mesh(mesh='a', level=1)
