import matplotlib.pyplot as plt
from utils import Polygon
import numpy as np
import unittest
from simulation import Simulation
from shape_optim import ShapeOptimizer
from scipy.spatial import ConvexHull


class TestShapeOpt(unittest.TestCase):
    def test_poly1(self):  # start from a rectangle, move one of the vertices and check the gradient at this vertex
        nfreq, freqmax = 4, 256
        v = np.array([0.5, 1.])  # np.array([1.231231, 0.8785678])
        v /= np.linalg.norm(v)
        lxinit, ly = 2., 2.4876

        # create a rectangular mic array
        poly_mic = Polygon(np.array([[0.2, 0.2], [lxinit - 0.2, 0.2], [lxinit - 0.2, ly - 0.2], [0.2, ly - 0.2]]))
        nmic = 32
        mic_pos = poly_mic.sample(nmic, min_dist=0.1)[0]

        pos_src = np.array([[0.5, ly / 2]])
        h = 0.5  # spacing between test points

        ntest = 5  # number of test points
        jval = np.zeros((5, ntest), dtype=float)
        djval = np.zeros((ntest, 2), dtype=float)
        init_vert = np.array([[0., 0], [lxinit, 0], [lxinit, ly], [0., ly]])

        # generate measurements
        sim = Simulation(init_vert, freqmax, nfreq, pos_src=pos_src, mic_pos=mic_pos, nworkers=1, dx=0.3,
                         min_dist=0.005, nsrc_schedule=dict(method='per_wavelength', nsrc=30),
                         error_thresh=1e-5, beta=0.1, add_src=10, simulate=True)

        obs = np.zeros([len(mic_pos), nfreq], dtype=np.complex128)
        obs[:, 1:] = sim.compute(mic_pos, adjoint=False)
        opt = ShapeOptimizer(init_vert, freqmax, nfreq,
                             source_pos=pos_src, mic_pos=mic_pos, measurements=obs, nworkers=1, dx=0.3, n_int=1000,
                             sim_kwargs=dict(min_dist=0.005, error_thresh=1e-5, beta=0.1, add_src=10,
                                             nsrc_schedule=dict(method='per_wavelength', nsrc=30)))
        hh = 1e-5  # step for finite differences
        for i in range(ntest):  # move vertex 2 in the direction of v
            print("i = ", i)
            vert_mod = init_vert.copy()
            vert_mod[2] = vert_mod[2] + i * h * v
            opt.update_sim(vert_mod, update_state=True, update_adjoint=True)
            djval[i] = opt.partial_derivative(2)

            jval[0, i] = opt.cost_fun()  # for tracking

            vertmod2 = vert_mod.copy()
            vertmod2[2] = vertmod2[2] - np.array([hh / 2, 0.])
            opt.update_sim(vertmod2, update_state=True, update_adjoint=False)
            jval[1, i] = opt.cost_fun()
            vertmod3 = vert_mod.copy()
            vertmod3[2] = vertmod3[2] + np.array([hh / 2, 0.])
            opt.update_sim(vertmod3, update_state=True, update_adjoint=False)
            jval[2, i] = opt.cost_fun()
            vertmod2 = vert_mod.copy()
            vertmod2[2] = vertmod2[2] - np.array([0., hh / 2])
            opt.update_sim(vertmod2, update_state=True, update_adjoint=False)
            jval[3, i] = opt.cost_fun()
            vertmod3 = vert_mod.copy()
            vertmod3[2] = vertmod3[2] + np.array([0., hh / 2])
            opt.update_sim(vertmod3, update_state=True, update_adjoint=False)
            jval[4, i] = opt.cost_fun()

        fdjvalx = (jval[2, :] - jval[1, :]) / hh
        fdjvaly = (jval[4, :] - jval[3, :]) / hh
        # check relative error is below 1e-2
        rel_errx = np.abs((fdjvalx[1:] - djval[1:, 0]) / fdjvalx[1:])
        rel_erry = np.abs((fdjvaly[1:] - djval[1:, 1]) / fdjvaly[1:])
        print("relative error x: {}  relative error y: {}".format(rel_errx, rel_erry))
        plot = False
        if plot:
            xrange = np.arange(0, ntest) * h
            plt.plot(xrange, fdjvalx, label='finite diffx')
            plt.plot(xrange, fdjvaly, label='finite diffy')
            plt.plot(xrange, djval[:, 0], 'k--', label='shape_derivx')
            plt.plot(xrange, djval[:, 1], 'g--', label='shape_derivy')
            plt.plot(xrange, jval[0, :], label='cost function')
            plt.legend()
            plt.show()

        self.assertTrue(np.all(rel_errx < 1e-2), "Relative error in x is too high")
        self.assertTrue(np.all(rel_erry < 1e-2), "Relative error in y is too high")
        self.assertAlmostEqual(djval[0, 0], 0, 5)  # check that the gradient is zero for the initial polygon
        self.assertAlmostEqual(djval[0, 1], 0, 5)


class TestShapeOpt2(unittest.TestCase):
    def test_large_angle(self):
        alpha = np.pi * 0.95
        vert = np.array([[0, 0.], [1., 0.], [0, 1.], [np.cos(alpha), np.sin(alpha)]])
        vert = vert[ConvexHull(vert).vertices]
        src_angle = alpha * np.sqrt(2) / 4
        pos_src = np.array([[np.cos(src_angle), np.sin(src_angle)]]) * np.sqrt(3) / 4.
        mic_pos = np.array([[np.cos(np.pi / np.sqrt(5)), np.sin(np.pi / np.sqrt(5))]]) / 2
        # create fake observations
        obs = np.zeros((1, 4), dtype=np.complex128)
        obs[:, 1:] = (np.linspace(1, 4, 3) * np.exp(1j * np.linspace(0, 2 * np.pi, 3)) +
                      1j * np.linspace(3, 7, 3) * np.exp(3j * np.linspace(0, 2 * np.pi, 3)))
        opt = ShapeOptimizer(vert, 512, 4,
                             source_pos=pos_src, mic_pos=mic_pos, measurements=obs, nworkers=1, dx=0.3, n_int=1500,
                             sim_kwargs=dict(min_dist=0.005, error_thresh=1e-5, beta=0.1, add_src=10,
                                             nsrc_schedule=dict(method='per_wavelength', nsrc=30)))
        print("poly: ", vert)
        h = 1e-6
        # check the gradient at each vertex
        for i in range(len(vert)):
            djval = opt.partial_derivative(i)
            print("i = ", i, " part J_i = ", djval)

            vertmod = vert.copy()
            vertmod[i] = vertmod[i] + np.array([h/2, 0.])
            opt.update_sim(vertmod, update_state=True, update_adjoint=False)
            jval2 = opt.cost_fun()
            vertmod2 = vert.copy()
            vertmod2[i] = vertmod2[i] - np.array([h/2, 0.])
            opt.update_sim(vertmod2, update_state=True, update_adjoint=False)
            jval3 = opt.cost_fun()

            vertmod3 = vert.copy()
            vertmod3[i] = vertmod3[i] + np.array([0., h/2])
            opt.update_sim(vertmod3, update_state=True, update_adjoint=False)
            jval4 = opt.cost_fun()
            vertmod4 = vert.copy()
            vertmod4[i] = vertmod4[i] - np.array([0., h/2])
            opt.update_sim(vertmod4, update_state=True, update_adjoint=False)
            jval5 = opt.cost_fun()
            # check the finite differences
            fd = np.array([(jval2 - jval3) / h, (jval4 - jval5) / h])
            rel_err = np.abs((fd - djval) / fd)
            print("relative error: ", rel_err)
            self.assertAlmostEqual(rel_err[0], 0, 2)  # check that the relative error is below 1 percent
            self.assertAlmostEqual(rel_err[1], 0, 2)


class TestFit(unittest.TestCase):
    def test_fit1(self):  # simple test case: shift one edge of a rectangle by 5cm outwards
        lx, ly = 3.21, 4.541

        optim_vert = np.array([[-lx / 2, -ly / 2], [lx / 2, -ly / 2], [lx / 2, ly / 2], [-lx / 2, ly / 2]])
        print("optim_vert:", optim_vert)
        pos_src = np.zeros([1, 2])
        nmic, nfreqs, fs, beta = 32, 16, 256, 0.1 - 0.05j
        mic_pos = np.stack([np.cos(np.linspace(0, 2 * np.pi, nmic, endpoint=False)),
                            np.sin(np.linspace(0, 2 * np.pi, nmic, endpoint=False))], axis=1) * 0.9
        t = Simulation(optim_vert, fs, nfreqs, pos_src=pos_src, mic_pos=mic_pos, nworkers=2, dx=0.3,
                       min_dist=0.005, nsrc_schedule=dict(method='per_wavelength', nsrc=30),
                       error_thresh=1e-5, beta=beta, add_src=10, simulate=True)
        obs = np.zeros([nmic, t.nfreqs], dtype=complex)
        obs[:, 1:] = t.compute(mic_pos)
        np.save("obs.npy", obs)

        obs = np.load("obs.npy")
        pert_vert = optim_vert.copy()
        pert_vert[1:3, 0] += 0.051
        print("original error: ", np.linalg.norm(pert_vert - optim_vert, axis=-1))

        opt = ShapeOptimizer(pert_vert, fs=fs, nfreq=nfreqs, source_pos=pos_src, mic_pos=mic_pos, measurements=obs,
                             dx=0.3,
                             nworkers=2, n_int=500, verbose=True, sim_kwargs={'beta': beta, 'add_src': 10})
        opt.fit(10, neval_search=7)

        final_vert = opt.sim.poly.vertices
        angles_ind1 = np.argsort(np.arctan2(final_vert[:, 1], final_vert[:, 0]))
        angles_ind2 = np.argsort(np.arctan2(optim_vert[:, 1], optim_vert[:, 0]))

        final_error = np.linalg.norm(final_vert[angles_ind1] - optim_vert[angles_ind2], axis=-1)
        print("final error: ", final_error)
        print("optimal poly: ", optim_vert)
        print("final poly: ", final_vert)
        self.assertTrue(np.all(final_error < 5e-3), "Final error too high")  # tested : error goes from 5cm to < 3mm
