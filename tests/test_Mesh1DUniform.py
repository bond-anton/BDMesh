from __future__ import division, print_function
import random
import math as m
import numpy as np
import unittest

from BDMesh import Mesh1DUniform, Mesh1D


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
        with self.assertRaises(TypeError):
            self.mesh.physical_step = 'a'
        self.mesh.physical_step = 1.1
        self.assertNotEqual(self.mesh.physical_step, 1.1)
        self.mesh.physical_step = self.mesh.jacobian
        self.assertEqual(self.mesh.physical_step, self.mesh.jacobian)
        self.mesh.physical_step = 1.1 * self.mesh.jacobian
        self.assertEqual(self.mesh.physical_step, self.mesh.jacobian)
        self.mesh.physical_step = -1.0
        self.assertEqual(self.mesh.physical_step, 10.0)

    def test_local_step(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        self.assertEqual(self.mesh.local_step, 0.1)
        self.mesh.local_step = 0.05
        self.assertEqual(self.mesh.local_step, 0.05)
        self.mesh.local_step = 0.053
        self.assertNotEqual(self.mesh.local_step, 0.053)
        with self.assertRaises(TypeError):
            self.mesh.local_step = 'a'
        self.mesh.local_step = 1
        self.assertEqual(self.mesh.local_step, 1)
        self.mesh.local_step = 2
        self.assertEqual(self.mesh.local_step, 1)
        self.mesh.local_step = -2
        self.assertEqual(self.mesh.local_step, 1)
        self.mesh.local_step = -0.5
        self.assertEqual(self.mesh.local_step, 1.0)

    def test_num(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        self.assertEqual(self.mesh.num, 11)
        self.mesh.num = 12
        self.assertEqual(self.mesh.num, 12)
        with self.assertRaises(TypeError):
            self.mesh.num = 'a'
        with self.assertRaises(TypeError):
            self.mesh.num = None
        self.assertEqual(self.mesh.num, 12)
        self.mesh.num = 2 + 1e-11
        self.assertEqual(self.mesh.num, 2)
        self.mesh.num = 2.8
        self.assertEqual(self.mesh.num, 2)
        self.mesh.num = 2
        self.assertEqual(self.mesh.num, 2)

    def test_crop(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        self.mesh.crop = [3, 2]
        np.testing.assert_equal(self.mesh.crop, np.array([3, 2]))
        self.mesh.crop = [0, 0]
        np.testing.assert_equal(self.mesh.crop, np.array([0, 0]))
        self.mesh.crop = [3, 2]
        np.testing.assert_equal(self.mesh.crop, np.array([3, 2]))
        with self.assertRaises(TypeError):
            self.mesh.crop = 3
        with self.assertRaises(TypeError):
            self.mesh.crop = 'a'
        with self.assertRaises(TypeError):
            self.mesh.crop = 'ab'
        self.mesh.crop = [-3, 2]
        np.testing.assert_equal(self.mesh.crop, np.array([0, 2]))
        self.mesh.crop = [3, -2]
        np.testing.assert_equal(self.mesh.crop, np.array([3, 0]))
        self.mesh.crop = [3, 2, 1]
        np.testing.assert_equal(self.mesh.crop, np.array([3, 2]))
        self.mesh.crop = [5, 5]
        np.testing.assert_equal(self.mesh.crop, np.array([5, 4]))

    def test_trim(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        self.mesh.crop = [3, 2]
        self.mesh.trim()
        trimmed = Mesh1DUniform(3, 8, physical_step=1.0)
        self.assertEqual(self.mesh, trimmed)

    def test_inner_mesh_indices(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        inner = Mesh1DUniform(3, 7, physical_step=1.0)
        indices = self.mesh.inner_mesh_indices(inner)
        self.assertEqual(indices, (3, 7))
        inner = Mesh1D(3, 7)
        indices = self.mesh.inner_mesh_indices(inner)
        self.assertEqual(indices, (3, 7))
        with self.assertRaises(AttributeError):
            self.mesh.inner_mesh_indices(1)
        inner = Mesh1DUniform(3, 17, physical_step=1.0)
        indices = self.mesh.inner_mesh_indices(inner)
        self.assertEqual(indices, (-1, -1))
        inner = Mesh1DUniform(-3, 17, physical_step=1.0)
        indices = self.mesh.inner_mesh_indices(inner)
        self.assertEqual(indices, (-1, -1))
        inner = Mesh1DUniform(0.55, 9.55, physical_step=1.0)
        indices = self.mesh.inner_mesh_indices(inner)
        self.assertEqual(indices, (1, 10))

    """
    def test_aligned(self):
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        # check if aligned with equal mesh
        other = Mesh1DUniform(0, 10, physical_step=1.0)
        self.assertTrue(self.mesh.is_aligned_with(other))
        # check if aligned with integer node mesh
        other = Mesh1DUniform(100, 110, physical_step=1.0)
        self.assertTrue(self.mesh.is_aligned_with(other))
        # check if aligned with half-step node mesh
        other = Mesh1DUniform(100, 110, physical_step=0.5)
        self.assertTrue(self.mesh.is_aligned_with(other))
        # check if aligned with floating point step mesh
        num = 899
        self.mesh = Mesh1DUniform(0, 10, num=num + 1)
        start = 100 * self.mesh.physical_step
        for i in range(1, 10):
            other = Mesh1DUniform(start, start + 10, num=2 * num + 1)
            self.assertTrue(self.mesh.is_aligned_with(other))
            num = other.num - 1
            start += other.physical_step * 7
        # check AssertionError
        with self.assertRaises(AssertionError):
            self.mesh.is_aligned_with(1)
        # check if not aligned with mesh of same step but shifted by some offset value
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        offset = 0.33
        other = Mesh1DUniform(100 + offset, 110 + offset, physical_step=1.0)
        self.assertFalse(self.mesh.is_aligned_with(other))
        # check if not aligned with mesh of non-integer step coefficient
        coeff = 1.33
        other = Mesh1DUniform(100, 110, physical_step=1.0 * coeff)
        self.assertFalse(self.mesh.is_aligned_with(other))

    def test_merge(self):
        # check merging with equal mesh
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        other = Mesh1DUniform(0, 10, physical_step=1.0)
        self.mesh.merge_with(other)
        self.assertEqual(self.mesh, other)
        # check mearging with floating point step mesh
        for step_coeff in range(1, 5):
            self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
            num = self.mesh.num - 1
            start = -5
            for i in range(1, 5):
                other = Mesh1DUniform(start, start + 10, num=step_coeff * num + 1)
                self.mesh.merge_with(other)
                merged = Mesh1DUniform(min(self.mesh.physical_boundary_1, start),
                                       max(self.mesh.physical_boundary_2, start + 10),
                                       physical_step=other.physical_step)
                self.assertEqual(self.mesh, merged)
                num = other.num - 1
                start += 1 + other.physical_step * random.choice([-1, 1])
                self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        # check merging with not overlapping mesh
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        with self.assertRaises(ValueError):
            self.mesh.merge_with(Mesh1DUniform(11, 21, physical_step=1.0))
        # check merging with not aligned mesh
        self.mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        with self.assertRaises(ValueError):
            self.mesh.merge_with(Mesh1DUniform(5, 15, physical_step=0.6))
        # check AssertionError
        with self.assertRaises(AssertionError):
            self.mesh.merge_with(1)
        # test priority of meshes
        self.mesh = Mesh1DUniform(0, 10, physical_step=0.1)
        other = Mesh1DUniform(5, 15, physical_step=0.1)
        self.mesh.merge_with(other, priority='self')
        merged = Mesh1DUniform(0, 15, physical_step=0.1)
        self.assertEqual(self.mesh, merged)
        self.mesh = Mesh1DUniform(0, 10, physical_step=0.1)
        other = Mesh1DUniform(5, 15, physical_step=0.1)
        self.mesh.merge_with(other, priority='other')
        merged = Mesh1DUniform(0, 15, physical_step=0.1)
        self.assertEqual(self.mesh, merged)
        self.mesh = Mesh1DUniform(0, 10, physical_step=0.1)
        other = Mesh1DUniform(5, 15, physical_step=0.1)
        with self.assertRaises(ValueError):
            self.mesh.merge_with(other, priority='xxx')
"""
