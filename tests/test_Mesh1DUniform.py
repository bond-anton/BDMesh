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