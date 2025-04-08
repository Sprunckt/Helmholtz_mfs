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
                              error_thresh=1e-5, beta=.1, add_src=10, simulate=True, nworkers=1, max_tries=1)
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

    def test_compute_grad(self):
        point = self.points

        sim, dx = self.sim, 1e-4
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)
        self.assertTrue(np.allclose(grad, sim.compute_grad(point), atol=1e-3))

    def test_evaluate_error(self):  # check the error for the two frequencies
        error1, error2 = self.sim.evaluate_error(1), self.sim.evaluate_error(2)
        print("error at freqs {} and {}:  {}  and {}".format(self.sim.freqs[1], self.sim.freqs[2], error1, error2))
        self.assertAlmostEqual(error1, 0., 5)
        self.assertAlmostEqual(error2, 0., 5)

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
                              error_thresh=1e-5, beta=0., add_src=3, simulate=True, nworkers=2, max_tries=1)
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


class TestSimulationAdjoint(unittest.TestCase):
    # test that the adjoint computation is done (no test against analytical values)
    def setUp(self):
        p = Polygon(np.array([[-6.87586954, -0.21505878], [-4.57026517, -5.8748617], [-0.21811167, -0.13829098],
                              [3.64630988, 6.08748134], [-2.34224988, 5.68912892], [-4.32104665, 3.6823515]]))
        pos_src = np.mean(p.vertices, axis=0).reshape((1, 2))

        self.sim = Simulation(p.vertices, mic_pos=np.array([[-4.4, 0.4], [-4.3, -4.2],
                                                            [-2.8, -2.2], [0, 4.2], [-0.3, 2.3]]),
                              min_dist=0.005, dx=0.3, fs=256, nfreq=4, pos_src=pos_src,
                              nsrc_schedule=dict(method='per_wavelength', nsrc=30),error_thresh=1e-5, beta=.1,
                              add_src=10, observations=np.arange(20).astype(float).reshape((5, 4)), max_tries=1,
                              nworkers=2, simulate=True)
        self.sim.simulate(adjoint=True)
        self.dx = 1e-4
        self.points = np.array([[-2.4, 0.4],
                                [1., 4]])

    def test_src_ampl(self):  # test that the adjoint amplitudes are correct (skip freq 0 of observations)
        self.assertTrue(np.allclose(self.sim.src_ampl_adj[:, 1:],
                                    self.sim.obs[:, 1:] - self.sim.compute(self.sim.mic_pos, adjoint=False), atol=1e-5))

    def test_evaluate_error(self):  # check the error for the two frequencies
        error1, error2 = self.sim.evaluate_error(1, adjoint=True), self.sim.evaluate_error(2, adjoint=True)
        print("error at freqs {} and {}:  {}  and {}".format(self.sim.freqs[1], self.sim.freqs[2], error1, error2))
        self.assertAlmostEqual(error1, 0., 5)
        self.assertAlmostEqual(error2, 0., 5)

    def test_compute_grad(self):
        point = self.points

        sim, dx = self.sim, 1e-4
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :], adjoint=True) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :], adjoint=True)) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :], adjoint=True) -
                          sim.compute(point - np.array([0, dx / 2])[None, :], adjoint=True)) / dx], axis=-1)
        self.assertTrue(np.allclose(grad, sim.compute_grad(point, adjoint=True), atol=1e-3))

    def test_corner_grad(self):
        point = self.sim.poly.vertices

        sim, dx = self.sim, 1e-6
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :], adjoint=True) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :], adjoint=True)) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :], adjoint=True) -
                          sim.compute(point - np.array([0, dx / 2])[None, :], adjoint=True)) / dx], axis=-1)

        self.assertTrue(np.linalg.norm(grad - sim.compute_corner_grad(adjoint=True)), 5)


class TestSimulationUpdatePoly(unittest.TestCase):
    # update from a starting polygon to another, check that the values hold up
    def setUp(self):
        new_vert = np.array([[-6.875, -0.2150], [-4.570, -5.874], [-0.21811167, -0.1382],
                             [3.646, 6.087], [-2.342, 5.689], [-4.321, 3.6]])
        lx, ly = np.ptp(new_vert[:, 0]), np.ptp(new_vert[:, 1])
        minx, miny = np.min(new_vert, axis=0)
        vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]]) + np.array([[minx, miny]])
        pos_src = np.array([[-2.4, 1.5]])
        poly = Polygon(vertices)
        self.sim = Simulation(poly.vertices,  mic_pos=np.array([[-4.4, 0.4], [-4.3, -4.2],
                                                                [-2.8, -2.2], [0, 4.2], [-0.3, 2.3]]),
                              min_dist=0.005, dx=0.3, fs=256, nfreq=4, pos_src=pos_src,
                              nsrc_schedule=dict(method='per_wavelength', nsrc=30), error_thresh=1e-5, beta=0.,
                              add_src=3, simulate=True, nworkers=2, max_tries=1)

        self.assertTrue(self.sim.update_poly(new_vert))
        self.sim.simulate()

        self.points = np.array([[-2.4, 0.4],
                                [1., 4]])

    def test_compute(self):  # check that the helmhotzian is zero
        point = self.points

        sim, dx = self.sim, 1e-4
        sim.src_ampl[:] = 0
        laplacian = (sim.compute(point + np.array([[dx, 0]])) + sim.compute(point - np.array([[dx, 0]])) +
                     sim.compute(point + np.array([[0, dx]])) + sim.compute(point - np.array([[0, dx]])) -
                     4 * sim.compute(point)) / dx ** 2
        helmholtzian = laplacian + sim.wave_numbers[1:] ** 2 * sim.compute(point)
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

    def test_evaluate_error(self):  # check the error for the two frequencies
        error1, error2 = self.sim.evaluate_error(1), self.sim.evaluate_error(2)
        print("error at freqs {} and {}:  {}  and {}".format(self.sim.freqs[1], self.sim.freqs[2], error1, error2))
        self.assertAlmostEqual(error1, 0., 4)
        self.assertAlmostEqual(error2, 0., 4)

    def test_corner_grad(self):
        point = self.sim.poly.vertices

        sim, dx = self.sim, 1e-6
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)

        self.assertTrue(np.linalg.norm(grad - sim.compute_corner_grad()), 5)

    def test_compute_grad(self):
        point = self.points

        sim, dx = self.sim, 1e-4
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)
        self.assertTrue(np.allclose(grad, sim.compute_grad(point), atol=1e-3))

    def test_src_ampl(self):  # test that the adjoint amplitudes are correct (skip freq 0 of observations)
        self.assertTrue(np.allclose(self.sim.src_ampl_adj[:, 1:],
                                    self.sim.obs[:, 1:] - self.sim.compute(self.sim.mic_pos, adjoint=False), atol=1e-5))

    def test_evaluate_error_adj(self):  # check the error for the two frequencies
        self.sim.simulate(adjoint=True)
        error1, error2 = self.sim.evaluate_error(1, adjoint=True), self.sim.evaluate_error(2, adjoint=True)
        print("error at freqs {} and {}:  {}  and {}".format(self.sim.freqs[1], self.sim.freqs[2], error1, error2))
        self.assertAlmostEqual(error1, 0., 4)
        self.assertAlmostEqual(error2, 0., 4)

    def test_compute_grad_adj(self):
        self.sim.simulate(adjoint=True)
        point = self.points

        sim, dx = self.sim, 1e-4
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :], adjoint=True) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :], adjoint=True)) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :], adjoint=True) -
                          sim.compute(point - np.array([0, dx / 2])[None, :], adjoint=True)) / dx], axis=-1)
        self.assertTrue(np.allclose(grad, sim.compute_grad(point, adjoint=True), atol=1e-3))

    def test_corner_grad_adj(self):
        self.sim.simulate(adjoint=True)
        point = self.sim.poly.vertices

        sim, dx = self.sim, 1e-6
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :], adjoint=True) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :], adjoint=True)) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :], adjoint=True) -
                          sim.compute(point - np.array([0, dx / 2])[None, :], adjoint=True)) / dx], axis=-1)

        self.assertTrue(np.linalg.norm(grad - sim.compute_corner_grad(adjoint=True)), 5)


class TestSimulationLargeAngle(unittest.TestCase):
    def setUp(self):
        alpha = np.pi*0.95
        vert = np.array([[0, 0.], [1., 0.], [0, 1.], [np.cos(alpha), np.sin(alpha)]])
        p = Polygon(vert)
        src_angle = alpha * np.sqrt(2) / 4
        pos_src = np.array([[np.cos(src_angle), np.sin(src_angle)]]) * np.sqrt(3) / 4.
        mic_pos = np.array([[np.cos(np.pi / np.sqrt(5)), np.sin(np.pi / np.sqrt(5))]]) / 2
        self.sim = Simulation(p.vertices, mic_pos=mic_pos.reshape((1, 2)),
                         min_dist=0.005, dx=0.3, fs=256, nfreq=4,
                         pos_src=pos_src, nsrc_schedule=dict(method='per_wavelength', nsrc=50),
                         error_thresh=1e-5, beta=.1, add_src=5, observations=np.ones([1, 4]), simulate=True)
        self.dx = 1e-4
        self.points = self.sim.mic_pos
        print(np.degrees(self.sim.poly.angles))

    def test_compute(self):  # check that the helmhotzian is zero
        point = self.points

        sim, dx = self.sim, 1e-3
        sim.src_ampl[:] = 0
        laplacian = (sim.compute(point + np.array([[dx, 0]])) + sim.compute(point - np.array([[dx, 0]])) +
                     sim.compute(point + np.array([[0, dx]])) + sim.compute(point - np.array([[0, dx]])) -
                     4 * sim.compute(point)) / dx ** 2
        helmholtzian = laplacian + sim.wave_numbers[1:]**2*sim.compute(point)
        # check that the array is close to 0
        print(helmholtzian)
        self.assertTrue(np.allclose(helmholtzian, np.zeros_like(helmholtzian), atol=1e-3))
        print("Helmholtzian 1:", helmholtzian)

    def test_compute_grad(self):
        point = self.points

        sim, dx = self.sim, 1e-4
        grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
                          sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
                         (sim.compute(point + np.array([0, dx / 2])[None, :]) -
                          sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)
        self.assertTrue(np.allclose(grad, sim.compute_grad(point), atol=1e-5))

    def test_evaluate_error(self):  # check the error for the two frequencies
        error1, error2 = self.sim.evaluate_error(1), self.sim.evaluate_error(2)
        print("error at freqs {} and {}:  {}  and {}".format(self.sim.freqs[1], self.sim.freqs[2], error1, error2))
        self.assertAlmostEqual(error1, 0., 5)
        self.assertAlmostEqual(error2, 0., 5)

    # def test_corner_grad(self):
    #     point = self.sim.poly.vertices
    #
    #     sim, dx = self.sim, 1e-8
    #     grad = np.stack([(sim.compute(point + np.array([dx / 2, 0])[None, :]) -
    #                       sim.compute(point - np.array([dx / 2, 0])[None, :])) / dx,
    #                      (sim.compute(point + np.array([0, dx / 2])[None, :]) -
    #                       sim.compute(point - np.array([0, dx / 2])[None, :])) / dx], axis=-1)
    #     #print(grad- sim.compute_corner_grad(), grad.shape, grad)
    #     print(np.linalg.norm(grad- sim.compute_corner_grad(), axis=-1))
    #     print("c", np.linalg.norm(sim.compute_corner_grad(), axis=-1))
    #     print("g", np.linalg.norm(grad, axis=-1))
    #     self.assertAlmostEqual(np.linalg.norm(grad[0] - sim.compute_corner_grad()[0]), 0, 5)