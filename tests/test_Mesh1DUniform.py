from __future__ import division, print_function
import math as m
import numpy as np
import unittest

from BDMesh import Mesh1DUniform


class TestMesh1DUniform(unittest.TestCase):

    def setUp(self):
        self.mesh = Mesh1DUniform(m.pi, 2*m.pi)

    def test_equality(self):
        other_mesh = Mesh1DUniform(m.pi, 2 * m.pi)
        self.assertEqual(self.mesh, other_mesh)
        other_mesh = Mesh1DUniform(2 * m.pi, m.pi, boundary_condition_1=1, boundary_condition_2=3)
        self.assertEqual(self.mesh, other_mesh)
        other_mesh = Mesh1DUniform(m.pi, 3 * m.pi)
        self.assertNotEqual(self.mesh, other_mesh)
        other_mesh = Mesh1DUniform(3 * m.pi, m.pi)
        self.assertNotEqual(self.mesh, other_mesh)
        with self.assertRaises(AssertionError):
            self.mesh == 'a'
        self.assertEqual(str(self.mesh),
                         'Mesh1DUniform: [%2.2g; %2.2g], %2.2g step, %d nodes' % (self.mesh.physical_boundary_1,
                                                                                  self.mesh.physical_boundary_2,
                                                                                  self.mesh.physical_step,
                                                                                  self.mesh.num))
        self.mesh = Mesh1DUniform(-10, 10, physical_step=1.0)
        other_mesh = Mesh1DUniform(-10, 10, num=21)
        self.assertEqual(self.mesh, other_mesh)

    def test_physical_step(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0, num=100)
        self.assertEqual(self.mesh.physical_step, 1.0)
        with self.assertRaises(AssertionError):
            self.mesh.physical_step = 'a'
        self.mesh.physical_step = 1.1
        self.assertNotEqual(self.mesh.physical_step, 1.1)
        with self.assertRaises(ValueError):
            self.mesh.physical_step = 0
        self.mesh.physical_step = self.mesh.jacobian
        self.assertEqual(self.mesh.physical_step, self.mesh.jacobian)
        self.mesh.physical_step = 1.1 * self.mesh.jacobian
        self.assertEqual(self.mesh.physical_step, self.mesh.jacobian)
        self.mesh.physical_step = -1.0
        self.assertEqual(self.mesh.physical_step, 1.0)


    def test_local_step(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        self.assertEqual(self.mesh.local_step, 0.1)
        self.mesh.local_step = 0.05
        self.assertEqual(self.mesh.local_step, 0.05)
        self.mesh.local_step = 0.053
        self.assertNotEqual(self.mesh.local_step, 0.053)
        with self.assertRaises(AssertionError):
            self.mesh.local_step = 'a'
        self.mesh.local_step = 1
        self.assertEqual(self.mesh.local_step, 1)
        with self.assertRaises(ValueError):
            self.mesh.local_step = 0
        self.mesh.local_step = 2
        self.assertEqual(self.mesh.local_step, 1)
        self.mesh.local_step = -2
        self.assertEqual(self.mesh.local_step, 1)
        self.mesh.local_step = -0.5
        self.assertEqual(self.mesh.local_step, 0.5)
