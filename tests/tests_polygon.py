"""tests for Polygon class in utils.py """
import unittest

import matplotlib.pyplot as plt
import numpy as np
from utils import Polygon, mod2pi


# test polygon
class TestPolygon1(unittest.TestCase):
    def setUp(self):
        self.max_nedges = 12
        self.origin = np.array([0.1316456, -0.265449])

    def test_angles(self):
        for k in range(3, self.max_nedges + 1):
            poly = Polygon(np.array([[np.cos(2*np.pi*i / k),
                                      np.sin(2*np.pi*i / k)] for i in range(k)]), origin=self.origin)
            for i in range(k):
                self.assertAlmostEqual(mod2pi(poly.angles[i]), mod2pi(np.pi - 2 * np.pi / k), places=12)
                self.assertAlmostEqual(mod2pi(poly.polar_angles[i]), mod2pi(np.pi/2 + np.pi * (2*i+1) / k), places=12)
                self.assertAlmostEqual(mod2pi(poly.theta[i]), mod2pi(np.pi * (2*i+1) / k), places=12)

    def test_param(self):
        for k in range(3, self.max_nedges + 1):
            poly = Polygon(np.array([[np.cos(2*np.pi*i / k),
                                      np.sin(2*np.pi*i / k)] for i in range(k)]), origin=self.origin)
            poly2 = Polygon((poly.phi, poly.theta), origin=self.origin)
            for i in range(k):
                self.assertAlmostEqual(np.linalg.norm(poly2.vertices[i] - poly.vertices[i]), 0, places=12)


class TestPolygon2(unittest.TestCase):
    def setUp(self):  # define random polygon with set seed
        self.vertices = np.array([[0,0.], [5, 0], [7, 2], [4, 5], [0, 4]])
        self.poly = Polygon(self.vertices)

    def test_angles(self):
        angles = mod2pi(np.array([np.pi/2, 3*np.pi/4, np.pi/2, 3*np.pi/4-np.arctan(1/4), np.arctan(1/4)+np.pi/2]))
        for i in range(5):
            self.assertAlmostEqual(self.poly.angles[i], angles[i], places=12)

    def test_intersection(self):
        a = np.array([0., 3])
        b = np.array([5., 4])
        vec = b-a
        interior_points = a + np.linspace(0, 1, 100, endpoint=False)[1:].reshape(-1, 1) * vec
        self.assertTrue(np.all(self.poly.is_in(interior_points, 0)))
        exterior_points = a - np.linspace(0, 1, 100, endpoint=False)[:-1].reshape(-1, 1) * vec
        self.assertFalse(np.all(self.poly.is_in(exterior_points, 0)))
        exterior_points = b + np.linspace(0, 1, 100, endpoint=False)[1:].reshape(-1, 1) * vec
        self.assertFalse(np.all(self.poly.is_in(exterior_points, 0)))


class TestPolygon3(unittest.TestCase):
    def setUp(self):
        self.lx, self.ly = 7.64984, 3.654132
        self.vertices = np.array([[0,0], [self.lx, 0], [self.lx, self.ly], [0, self.ly]])
        self.poly = Polygon(self.vertices)

    def test_angles(self):
        angles = mod2pi(np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]))
        for i in range(4):
            self.assertAlmostEqual(self.poly.angles[i], angles[i], places=12)

    def test_intersection(self):
        a = np.array([self.lx/2, 0.])
        b = np.array([self.lx/2, self.ly])
        vec = b-a
        min_dist = 0.1
        interior_points = a + np.linspace(min_dist, 1-min_dist, 100, endpoint=False)[1:].reshape(-1, 1) * vec
        self.assertTrue(np.all(self.poly.is_in(interior_points, min_dist)))  # inside and respect min_dist
        close1 = a + np.linspace(0, min_dist, 100, endpoint=False)[1:-1].reshape(-1, 1) * vec
        self.assertFalse(np.all(self.poly.is_in(close1, min_dist)))  # close to a
        close2 = b - np.linspace(0, min_dist, 100, endpoint=False)[1:-1].reshape(-1, 1) * vec
        self.assertFalse(np.all(self.poly.is_in(close2, min_dist)))  # close to b
        exterior = a - np.linspace(0, 1, 100, endpoint=False)[:-1].reshape(-1, 1) * vec
        self.assertFalse(np.all(self.poly.is_in(exterior, min_dist)))  # outside
