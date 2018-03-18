from __future__ import division, print_function
from copy import copy, deepcopy
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
        other_mesh = Mesh1D(2 * m.pi, m.pi, boundary_condition_1=1, boundary_condition_2=3)
        self.assertEqual(self.mesh, other_mesh)
        other_mesh = Mesh1D(m.pi, 3 * m.pi)
        self.assertNotEqual(self.mesh, other_mesh)
        other_mesh = Mesh1D(3 * m.pi, m.pi)
        self.assertNotEqual(self.mesh, other_mesh)
        self.assertNotEqual(self.mesh, 'a')
        self.assertEqual(str(self.mesh),
                         'Mesh1D: [%2.2g; %2.2g], %d nodes' % (self.mesh.physical_boundary_1,
                                                               self.mesh.physical_boundary_2,
                                                               self.mesh.num))

    # def test_copy(self):
    #     mesh_copy = self.mesh.copy()
    #     self.assertEqual(self.mesh, mesh_copy)
    #     mesh_copy = copy(self.mesh)
    #     self.assertEqual(self.mesh, mesh_copy)
    #     mesh_copy = deepcopy(self.mesh)
    #     self.assertEqual(self.mesh, mesh_copy)

    def test_physical_boundaries(self):
        self.assertEqual(self.mesh.physical_boundary_1, m.pi)
        self.assertEqual(self.mesh.physical_boundary_2, 2 * m.pi)
        self.assertEqual(self.mesh.jacobian, m.pi)
        self.mesh.physical_boundary_1 = 0.1
        self.assertEqual(self.mesh.physical_boundary_1, 0.1)
        self.mesh.physical_boundary_2 = 2.1
        self.assertEqual(self.mesh.physical_boundary_2, 2.1)
        with self.assertRaises(ValueError):
            self.mesh.physical_boundary_1 = 3.1
        with self.assertRaises(ValueError):
            self.mesh.physical_boundary_2 = -3.1
        self.assertEqual(self.mesh.jacobian, 2.0)
        with self.assertRaises(TypeError):
            self.mesh.physical_boundary_1 = 'a'
        with self.assertRaises(TypeError):
            self.mesh.physical_boundary_2 = 'b'

    def test_local_nodes(self):
        np.testing.assert_equal(self.mesh.local_nodes, np.array([0.0, 1.0]))
        self.assertEqual(self.mesh.num, 2)
        self.mesh.local_nodes = np.array([0.0, 0.5, 1.0])
        self.assertEqual(self.mesh.num, 3)
        with self.assertRaises(TypeError):
            self.mesh.local_nodes = 1
        with self.assertRaises(TypeError):
            self.mesh.local_nodes = 'a'
        with self.assertRaises(TypeError):
            self.mesh.local_nodes = 'aa'
        with self.assertRaises(TypeError):
            self.mesh.local_nodes = [1.0, 2.0]
        with self.assertRaises(ValueError):
            self.mesh.local_nodes = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            self.mesh.local_nodes = np.array([1e-14, 1.0])
        with self.assertRaises(AttributeError):
            self.mesh.num = 4
        self.assertEqual(self.mesh.num, 3)
        np.testing.assert_equal(self.mesh.physical_nodes, np.array([m.pi, 1.5 * m.pi, 2 * m.pi]))

    def test_coordinate_conversion(self):
        self.assertEqual(self.mesh.to_physical_coordinate(np.array([0.5])), np.array([1.5 * m.pi]))
        self.assertEqual(self.mesh.to_local_coordinate(np.array([1.5 * m.pi])), np.array([0.5]))

    def test_boundary_conditions(self):
        self.assertEqual(self.mesh.boundary_condition_1, 0.0)
        self.assertEqual(self.mesh.boundary_condition_2, 0.0)
        self.mesh.boundary_condition_1 = 3.0
        self.assertEqual(self.mesh.boundary_condition_1, 3.0)
        self.mesh.boundary_condition_1 = 1
        self.assertEqual(self.mesh.boundary_condition_1, 1.0)
        with self.assertRaises(TypeError):
            self.mesh.boundary_condition_1 = None
        with self.assertRaises(TypeError):
            self.mesh.boundary_condition_1 = 'a'
        with self.assertRaises(TypeError):
            self.mesh.boundary_condition_2 = 'a'

    def test_solution_residual(self):
        np.testing.assert_equal(self.mesh.solution, np.zeros(2))
        a = np.linspace(0.0, 1.0, num=11, endpoint=True)
        self.mesh.local_nodes = a
        self.mesh.solution = a * 2
        self.mesh.residual = 0.1 * a
        np.testing.assert_equal(self.mesh.local_nodes, a)
        np.testing.assert_equal(self.mesh.solution, 2 * a)
        np.testing.assert_equal(self.mesh.residual, a * 0.1)
        with self.assertRaises(TypeError):
            self.mesh.solution = 1
        with self.assertRaises(TypeError):
            self.mesh.residual = 1
        with self.assertRaises(TypeError):
            self.mesh.solution = [1.0]
        with self.assertRaises(ValueError):
            self.mesh.solution = np.array([1.0])
        with self.assertRaises(TypeError):
            self.mesh.residual = [1.0]
        with self.assertRaises(ValueError):
            self.mesh.residual = np.array([1.0])
        self.assertEqual(self.mesh.integrational_residual, np.trapz(self.mesh.residual, self.mesh.physical_nodes))

    def test_local_f(self):
        a = np.linspace(0.0, 1.0, num=101, endpoint=True)
        self.mesh.local_nodes = a
        f = np.sin
        local_f = self.mesh.local_f(f)
        np.testing.assert_equal(f(self.mesh.physical_nodes), local_f(self.mesh.local_nodes))

        def f(x, t):
            return np.sin(x) * t
        local_f = self.mesh.local_f(f, 2)
        np.testing.assert_equal(f(self.mesh.physical_nodes, 2), local_f(self.mesh.local_nodes, 2))
        with self.assertRaises(AssertionError):
            self.mesh.local_f('global_f')

    def test_inner_mesh(self):
        other = Mesh1D(0.5 * m.pi, 2.5 * m.pi)
        self.assertTrue(self.mesh.is_inside_of(other))
        other = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
        self.assertTrue(self.mesh.is_inside_of(other))
        other = Mesh1D(0.5 * m.pi, 1.5 * m.pi)
        self.assertFalse(self.mesh.is_inside_of(other))
        other = Mesh1D(1.1 * m.pi, 2.5 * m.pi)
        self.assertFalse(self.mesh.is_inside_of(other))
        other = Mesh1D(1.1 * m.pi, 1.9 * m.pi)
        self.assertFalse(self.mesh.is_inside_of(other))
        self.assertTrue(other.is_inside_of(self.mesh))
        with self.assertRaises(TypeError):
            self.mesh.is_inside_of('x')

    def test_overlap(self):
        other = Mesh1D(0.5 * m.pi, 2.5 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(0.5 * m.pi, 1.5 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(1.1 * m.pi, 2.5 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(1.1 * m.pi, 1.9 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(0.5 * m.pi, 1.0 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(2.0 * m.pi, 3.0 * m.pi)
        self.assertTrue(self.mesh.overlap_with(other))
        self.assertTrue(other.overlap_with(self.mesh))
        other = Mesh1D(0.5 * m.pi, 0.99 * m.pi)
        self.assertFalse(self.mesh.overlap_with(other))
        self.assertFalse(other.overlap_with(self.mesh))
        other = Mesh1D(2.01 * m.pi, 3.0 * m.pi)
        self.assertFalse(self.mesh.overlap_with(other))
        self.assertFalse(other.overlap_with(self.mesh))
        with self.assertRaises(TypeError):
            self.mesh.overlap_with('x')

    # def test_merge(self):
    #     # other bounds self.mesh
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(0.5 * m.pi, 2.5 * m.pi)
    #     self.mesh.merge_with(other)
    #     other.local_nodes = (np.array([0.5, 1.0, 2.0, 2.5]) - 0.5) / 2.0
    #     self.assertEqual(self.mesh, other)
    #
    #     # other is inner to self.mesh
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(1.1 * m.pi, 1.9 * m.pi)
    #     self.mesh.merge_with(other)
    #     merged = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     merged.local_nodes = np.array([1.0, 1.1, 1.9, 2.0]) - 1.0
    #     self.assertEqual(self.mesh, merged)
    #
    #     # other equals to self.mesh
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     self.mesh.merge_with(other)
    #     self.assertEqual(self.mesh, other)
    #     other.merge_with(self.mesh)
    #     self.assertEqual(self.mesh, other)
    #
    #     # other overlaps with self.mesh
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(0.5 * m.pi, 1.5 * m.pi)
    #     self.mesh.merge_with(other)
    #     merged = Mesh1D(0.5 * m.pi, 2 * m.pi)
    #     merged.local_nodes = (np.array([0.5, 1.0, 1.5, 2.0]) - 0.5) / 1.5
    #     self.assertEqual(self.mesh, merged)
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other.merge_with(self.mesh)
    #     self.assertEqual(other, merged)
    #
    #     # other coincide with self.mesh in single point from left side
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(0.5 * m.pi, 1.0 * m.pi)
    #     self.mesh.merge_with(other)
    #     merged = Mesh1D(0.5 * m.pi, 2 * m.pi)
    #     merged.local_nodes = (np.array([0.5, 1.0, 2.0]) - 0.5) / 1.5
    #     self.assertEqual(self.mesh, merged)
    #     # other coincide with self.mesh in single point from right side
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(2.0 * m.pi, 3.0 * m.pi)
    #     self.mesh.merge_with(other)
    #     merged = Mesh1D(m.pi, 3 * m.pi)
    #     merged.local_nodes = (np.array([1.0, 2.0, 3.0]) - 1.0) / 2.0
    #     self.assertEqual(self.mesh, merged)
    #     # meshes do not overlap
    #     self.mesh = Mesh1D(1.0 * m.pi, 2.0 * m.pi)
    #     other = Mesh1D(0.5 * m.pi, 0.99 * m.pi)
    #     self.mesh.merge_with(other)
    #     self.assertEqual(self.mesh, Mesh1D(m.pi, 2 * m.pi))
    #     other.merge_with(self.mesh)
    #     self.assertEqual(other, Mesh1D(0.5 * m.pi, 0.99 * m.pi))
    #     other = Mesh1D(2.01 * m.pi, 3.0 * m.pi)
    #     self.mesh.merge_with(other)
    #     self.assertEqual(self.mesh, Mesh1D(m.pi, 2 * m.pi))
    #     other.merge_with(self.mesh)
    #     self.assertEqual(other, Mesh1D(2.01 * m.pi, 3.0 * m.pi))
    #
    #     self.mesh = Mesh1D(0, 10)
    #     self.mesh.local_nodes = np.linspace(0, 1, num=11)
    #     other = Mesh1D(5, 15)
    #     other.local_nodes = np.linspace(0, 1, num=11)
    #     self.mesh.merge_with(other)
    #     merged = Mesh1D(0, 15)
    #     merged.local_nodes = np.linspace(0, 1, num=16)
    #     self.assertEqual(self.mesh, merged)
    #
    #     self.mesh = Mesh1D(0, 10)
    #     self.mesh.local_nodes = np.linspace(0, 1, num=11)
    #     other = Mesh1D(5, 15)
    #     other.local_nodes = np.linspace(0, 1, num=11)
    #     self.mesh.merge_with(other, priority='self')
    #     merged = Mesh1D(0, 15)
    #     merged.local_nodes = np.linspace(0, 1, num=16)
    #     self.assertEqual(self.mesh, merged)
    #
    #     self.mesh = Mesh1D(0, 10)
    #     self.mesh.local_nodes = np.linspace(0, 1, num=11)
    #     other = Mesh1D(5, 15)
    #     other.local_nodes = np.linspace(0, 1, num=11)
    #     self.mesh.merge_with(other, priority='other')
    #     merged = Mesh1D(0, 15)
    #     merged.local_nodes = np.linspace(0, 1, num=16)
    #     self.assertEqual(self.mesh, merged)
    #
    #     self.mesh = Mesh1D(0, 10)
    #     self.mesh.local_nodes = np.linspace(0, 1, num=11)
    #     other = Mesh1D(5, 15)
    #     other.local_nodes = np.linspace(0, 1, num=11)
    #
    #     with self.assertRaises(ValueError):
    #         self.mesh.merge_with(other, priority='xxx')
    #
    #     with self.assertRaises(AssertionError):
    #         self.mesh.merge_with('x')
