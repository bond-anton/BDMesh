from __future__ import division, print_function
import math as m
import numpy as np
import unittest

from BDMesh import Mesh1D


class TestMesh1D(unittest.TestCase):

    def setUp(self):
        self.mesh = Mesh1D(m.pi, 2*m.pi)

    def test_equality(self):
        other_mesh = Mesh1D(m.pi, 2 * m.pi)
        self.assertEqual(self.mesh, other_mesh)
        other_mesh = Mesh1D(2 * m.pi, m.pi)
        self.assertEqual(self.mesh, other_mesh)
        other_mesh = Mesh1D(m.pi, 3 * m.pi)
        self.assertNotEqual(self.mesh, other_mesh)
        other_mesh = Mesh1D(3 * m.pi, m.pi)
        self.assertNotEqual(self.mesh, other_mesh)

    def test_physical_boundaries(self):
        self.assertEqual(self.mesh.physical_boundary_1, m.pi)
        self.assertEqual(self.mesh.physical_boundary_2, 2 * m.pi)
        self.mesh.physical_boundary_1 = 0.1
        self.assertEqual(self.mesh.physical_boundary_1, 0.1)
        self.mesh.physical_boundary_2 = 2.1
        self.assertEqual(self.mesh.physical_boundary_2, 2.1)
        with self.assertRaises(ValueError):
            self.mesh.physical_boundary_1 = 3.1
        with self.assertRaises(ValueError):
            self.mesh.physical_boundary_2 = -3.1

    def test_local_nodes(self):
        np.testing.assert_equal(self.mesh.local_nodes, np.array([0.0, 1.0]))
