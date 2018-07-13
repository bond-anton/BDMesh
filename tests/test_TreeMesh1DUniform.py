from __future__ import division, print_function
import numpy as np
import unittest

from BDMesh import Mesh1DUniform, TreeMesh1DUniform, Mesh1D


class TestTreeMesh1DUniform(unittest.TestCase):

    def setUp(self):
        self.root_mesh = Mesh1DUniform(0.0, 10.0, physical_step=1.0)
        self.tree = TreeMesh1DUniform(self.root_mesh, refinement_coefficient=2, aligned=True)

    def test_constructor(self):
        with self.assertRaises(TypeError):
            TreeMesh1DUniform(1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh]})
        self.assertEqual(self.tree.levels, [0])
        self.assertEqual(self.tree.root_mesh, self.root_mesh)
        self.assertEqual(self.tree.refinement_coefficient, 2)
        self.assertTrue(self.tree.aligned)
        np.testing.assert_equal(self.tree.crop, np.array([0, 0]))

    def test_refinement_coefficient(self):
        self.assertEqual(self.tree.refinement_coefficient, 2)
        mesh = Mesh1DUniform(1, 8, physical_step=0.5)
        self.tree.add_mesh(mesh=mesh)
        mesh1 = Mesh1DUniform(2, 10, physical_step=0.25)
        self.tree.add_mesh(mesh=mesh1)
        self.tree.refinement_coefficient = 4
        self.assertEqual(self.tree.refinement_coefficient, 4)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh],
                                          1: [Mesh1DUniform(1, 8, physical_step=0.25)],
                                          2: [Mesh1DUniform(2, 10, physical_step=0.125)]})
        with self.assertRaises(TypeError):
            self.tree.refinement_coefficient = 'a'

    def test_aligned(self):
        self.assertTrue(self.tree.aligned)
        self.tree.aligned = False
        self.assertFalse(self.tree.aligned)
        self.tree.aligned = True
        self.assertTrue(self.tree.aligned)
        self.tree.aligned = False
        self.assertFalse(self.tree.aligned)
        self.tree.add_mesh(mesh=Mesh1DUniform(1.3, 8.3, physical_step=0.5))
        self.tree.aligned = True
        self.assertFalse(self.tree.aligned)
        self.tree.aligned = 'a'
        self.assertFalse(self.tree.aligned)

    def test_crop(self):
        np.testing.assert_equal(self.tree.crop, np.array([0, 0]))
        self.tree.crop = [3.0, 2.0]
        np.testing.assert_equal(self.tree.crop, np.array([3, 2]))
        with self.assertRaises(TypeError):
            self.tree.crop = 1
        with self.assertRaises(TypeError):
            self.tree.crop = 'ab'
        with self.assertRaises(TypeError):
            self.tree.crop = 'abc'
        self.tree.crop = [1, 2, 3]
        np.testing.assert_equal(self.tree.crop, np.array([1, 2]))
        self.tree.crop = [5, 5]
        np.testing.assert_equal(self.tree.crop, np.array([5, 4]))
        self.tree.crop = [-1, 1]
        np.testing.assert_equal(self.tree.crop, np.array([0, 1]))
        self.tree.crop = [1, -1]
        np.testing.assert_equal(self.tree.crop, np.array([1, 0]))
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
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh], 2: [mesh1]})
        # testing bad values
        self.assertFalse(self.tree.add_mesh(mesh=Mesh1DUniform(1.3, 8.3, physical_step=0.5)))
        self.assertFalse(self.tree.add_mesh(mesh=Mesh1DUniform(1, 8, physical_step=0.33)))
        with self.assertRaises(TypeError):
            self.tree.add_mesh(mesh='a', level=1)

    def test_loop_refinement(self):
        # adding refinement meshes in a loop
        mesh = Mesh1DUniform(0.0, 2.0, physical_step=0.1)
        self.tree = TreeMesh1DUniform(mesh, refinement_coefficient=2, aligned=True)
        for i in range(5):
            try:
                mesh = Mesh1DUniform(0.2, 1.1, physical_step=mesh.physical_step / self.tree.refinement_coefficient)
                self.tree.add_mesh(mesh)
            except ValueError as e:
                print(e)
                print('iteration =', i)
                break

    def test_trim(self):
        mesh = Mesh1DUniform(1, 8, physical_step=0.5)
        self.tree.add_mesh(mesh=mesh)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh]})
        mesh1 = Mesh1DUniform(2, 10, physical_step=0.25)
        self.tree.add_mesh(mesh=mesh1)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh], 2: [mesh1]})
        self.tree.trim()
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh], 2: [mesh1]})
        self.tree.crop = [2, 2]
        self.tree.trim()
        self.assertEqual(self.tree.tree, {0: [Mesh1DUniform(2, 8, physical_step=1.0)],
                                          1: [Mesh1DUniform(2, 8, physical_step=0.5)],
                                          2: [Mesh1DUniform(2, 8, physical_step=0.25)]})
        self.tree.crop = [0, 1]
        self.tree.trim()
        self.assertEqual(self.tree.tree, {0: [Mesh1DUniform(2, 7, physical_step=1.0)],
                                          1: [Mesh1DUniform(2, 7, physical_step=0.5)],
                                          2: [Mesh1DUniform(2, 7, physical_step=0.25)]})
        self.tree.crop = [1, 0]
        self.tree.trim()
        self.assertEqual(self.tree.tree, {0: [Mesh1DUniform(3, 7, physical_step=1.0)],
                                          1: [Mesh1DUniform(3, 7, physical_step=0.5)],
                                          2: [Mesh1DUniform(3, 7, physical_step=0.25)]})
        mesh1 = Mesh1DUniform(3, 4, physical_step=0.125)
        self.tree.add_mesh(mesh=mesh1)
        self.tree.crop = [1, 0]
        self.tree.trim()
        self.assertEqual(self.tree.tree, {0: [Mesh1DUniform(4, 7, physical_step=1.0)],
                                          1: [Mesh1DUniform(4, 7, physical_step=0.5)],
                                          2: [Mesh1DUniform(4, 7, physical_step=0.25)]})

    def test_flatten(self):
        mesh = Mesh1DUniform(1, 4, physical_step=0.5)
        self.tree.add_mesh(mesh=mesh)
        mesh1 = Mesh1DUniform(2, 3, physical_step=0.25)
        self.tree.add_mesh(mesh=mesh1)
        mesh2 = Mesh1DUniform(9, 10.5, physical_step=0.5)
        self.tree.add_mesh(mesh=mesh2)
        self.assertEqual(self.tree.tree, {0: [self.root_mesh], 1: [mesh, mesh2], 2: [mesh1]})
        flattened = self.tree.flatten()
        self.assertTrue(isinstance(flattened, Mesh1D))
        flat_grid = np.array([0.0, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0,
                              6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 10.5])
        np.testing.assert_allclose(flattened.physical_nodes, flat_grid)
