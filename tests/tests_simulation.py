import unittest
from simulation import Simulation
from utils import Polygon
import numpy as np
import matplotlib.pyplot as plt


class TestSimulation(unittest.TestCase):
    def setUp(self):
        p = Polygon(np.array([[-6.87586954, -0.21505878], [-4.57026517, -5.8748617], [-0.21811167, -0.13829098],
                              [3.64630988, 6.08748134], [-2.34224988, 5.68912892], [-4.32104665, 3.6823515]]))
        pos_src = np.mean(p.vertices, axis=0).reshape((1, 2))

        self.sim = Simulation(p.vertices, mic_pos=np.zeros([1, 2]), min_dist=0.005, dx=0.3, fs=256, nfreq=4,
                              pos_src=pos_src, nsrc_schedule=dict(method='per_wavelength', nsrc=30),
                              error_thresh=1e-5, beta=.1, add_src=10)
        self.sim.simulate(1)
        self.dx = 1e-4
        self.points = np.array([[-2.4, 0.4],
                                [1., 4]])

    def test_compute(self):  # check that the helmhotzian is zero
        point = self.points

        sim, dx = self.sim, 1e-4
        sim.src_ampl[:] = 0
        laplacian = (sim.compute(point + np.array([[dx, 0]])) + sim.compute(point - np.array([[dx, 0]])) +
                     sim.compute(point + np.array([[0, dx]])) + sim.compute(point - np.array([[0, dx]])) -
                     4 * sim.compute(point)) / dx ** 2
        helmholtzian = laplacian + sim.wave_numbers[1:]**2*sim.compute(point)
        # check that the array is close to 0
        self.assertTrue(np.allclose(helmholtzian, np.zeros_like(helmholtzian), atol=1e-3))
        print("Helmholtzian 1:", helmholtzian)

    def test_compute_grad1(self):
        point = self.points

        sim, dx = self.sim, 1e-4
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)
        self.assertTrue(np.allclose(grad, sim.compute_grad(point), atol=1e-3))

    def test_corner_grad(self):
        point = self.sim.poly.vertices

        sim, dx = self.sim, 1e-6
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)

        self.assertTrue(np.linalg.norm(grad -sim.compute_corner_grad()), 5)


class TestSimulation2(unittest.TestCase):
    # test against the analytical eigenmode decomposition on a square domain with Neumann conditions
    def setUp(self):
        self.lx, self.ly = 5.2, 3.3
        vertices = np.array([[0, 0], [self.lx, 0.], [self.lx, self.ly], [0, self.ly]])
        poly = Polygon(vertices)
        pos_src = np.array([[2.5, 2.5]])
        fs = 100.
        self.sim = Simulation(poly.vertices, mic_pos=np.zeros([1, 2]), min_dist=0.005, dx=0.3, fs=fs, nfreq=2,
                              pos_src=pos_src, nsrc_schedule=dict(method='per_wavelength', nsrc=30),
                              error_thresh=1e-5, beta=0., add_src=3)
        ntest, nmax = 50, 15000
        points = np.meshgrid(np.linspace(0.1, self.lx - 0.1, ntest), np.linspace(0.1, self.ly - 0.1, ntest))
        points = np.stack(points, axis=2).reshape(-1, 2)
        self.ntest = ntest
        self.val_points = self.sim.compute(points)
        self.val_exact = np.loadtxt("../data/exact_sol_{}_{}_{}.csv".format(0, nmax, ntest), delimiter=",")

    def test_real(self):  # check if the approximation is real-valued
        self.assertAlmostEqual(np.linalg.norm(np.imag(self.val_points)), 0., 5)

    def test_analytical(self):  # check against analytical values
        plot = False
        if plot:  # plot boundary
            self.sim.plot_bound_values(1)
            sol = self.val_exact

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            maxval = np.max(np.abs(np.real(sol)))
            im1 = ax[1].matshow(np.real(sol).reshape(self.ntest, self.ntest),
                                vmin=-maxval, vmax=maxval, extent=[0, self.lx, 0, self.ly], origin='lower')
            im2 = ax[0].matshow(self.val_points.reshape(self.ntest, self.ntest),
                                vmin=-maxval, vmax=maxval, extent=[0, self.lx, 0, self.ly], origin='lower')
            # add colorbar
            fig.colorbar(im1, ax=ax[1], orientation='horizontal')
            fig.colorbar(im2, ax=ax[0], orientation='horizontal')
            ax[0].set_title('Analytical')
            ax[1].set_title('MFS')
            plt.show()

        self.assertAlmostEqual(np.linalg.norm(self.val_points.flatten() - self.val_exact), 0., 5)

    def test_corner_grad(self):
        point = self.sim.poly.vertices

        sim, dx = self.sim, 1e-6
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)

        self.assertTrue(np.linalg.norm(grad - sim.compute_corner_grad()), 10)