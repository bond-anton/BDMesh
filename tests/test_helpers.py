from __future__ import division, print_function
from BDMesh._helpers import check_if_integer

import unittest


class TestHelpers(unittest.TestCase):

    def setUp(self):
        pass

    def test_check_if_integer(self):
        self.assertTrue(check_if_integer(1, 1e-10))
        self.assertTrue(check_if_integer(1.0, 1e-10))
        self.assertTrue(check_if_integer(1.0 - 3e-11, 1e-10))
        self.assertTrue(check_if_integer(1.0 + 3e-11, 1e-10))
        self.assertFalse(check_if_integer(1.0 - 1e-10, 1e-10))
        self.assertFalse(check_if_integer(1.0 + 1e-10, 1e-10))
