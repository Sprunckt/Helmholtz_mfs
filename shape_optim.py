from simulation import Simulation
import numpy as np
from scipy.optimize import minimize


class ShapeOptimizer:
    def __init__(self, init_vertices: np.ndarray, fs: float, nfreq: int, source_pos: np.ndarray, mic_pos: np.ndarray,
                 measurements: np.ndarray, dx: float, nprocess=1, n_int=1000, sim_kwargs=None, src_ampl=None):
        if sim_kwargs is None:
            sim_kwargs = {"boundsep": 0.05}
        self.sim_p = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx,
                                pos_src=source_pos, src_ampl=src_ampl, **sim_kwargs)
        sim_kwargs_q = sim_kwargs.copy()
        sim_kwargs_q["beta"] = - np.asarray(sim_kwargs_q["beta"])  # todo: handle callable case
        self.sim_q = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, pos_src=mic_pos, **sim_kwargs_q)

        self.source_pos = source_pos
        self.mic_pos = mic_pos
        self.antenna_radius = np.max(np.linalg.norm(self.mic_pos, axis=1))
        self.nprocess = nprocess
        self.observations = measurements  # shape len(freqs)
        self.diffs = np.full(nfreq, np.inf, dtype=complex)
        self.n_int = n_int  # number of quadrature points on the boundary
        self.quad_points, _, self.quad_ids, self.quad_nodes = self.sim_p.poly.sample(self.n_int, return_nodes=True,
                                                                                     min_dist=0.)

        self.quad_intervals = np.zeros([self.sim_p.poly.nedges, 2], dtype=int)
        interv = np.where(np.diff(self.quad_ids) != 0)[0]
        self.quad_intervals[0], self.quad_intervals[-1] = [0, interv[0]], [interv[-1]+1, len(self.quad_ids)-1]
        self.quad_intervals[1:-1] = np.stack([interv[:-1]+1, interv[1:]], axis=1)

        self.p_values = np.zeros((self.n_int, nfreq), dtype=complex)
        self.q_values = np.zeros((self.n_int, nfreq), dtype=complex)
        self.p_grads = np.zeros((self.n_int, nfreq, 2), dtype=complex)
        self.q_grads = np.zeros((self.n_int, nfreq, 2), dtype=complex)
        self.corner_values = np.zeros((len(self.sim_p.poly.vertices), 2, nfreq), dtype=complex)
        self.corner_grads = np.zeros((len(self.sim_p.poly.vertices), 2, nfreq, 2), dtype=complex)
        self.update_sim()

        self.counter = {"ncall":0}

    def fit(self, niter=200, geom_tol=0.5, rmax=20., step=1e-3):
        # optimize using lbgfs-b
        minr = self.antenna_radius + geom_tol
        bounds = [(minr, rmax) for _ in range(self.sim.poly.nedges)] + \
                 [(-4*np.pi, 4*np.pi) for _ in range(self.sim.poly.nedges)]
        p0 = np.concatenate([self.sim.poly.phi, self.sim.poly.theta])
        counter = {"ncall":0}
        print("optimizing...")
        opt_res = minimize(self.cost_fun, p0, method='Nelder-Mead', 
        bounds=bounds, args=(counter,))
        print("done")
        return opt_res

    def update_sim(self, new_vertices=None, update_adjoint=True):
        if new_vertices is not None:
            self.sim_p.update_vertices(new_vertices)
            self.sim_q.update_vertices(new_vertices)

        self.sim_p.reset()
        self.sim_p.simulate(self.nprocess)

        mic_val, _ = self.sim_p.compute(self.mic_pos, self.nprocess)

        self.diffs = mic_val - self.observations

        if update_adjoint:
            self.sim_q.reset()
            self.sim_q.src_ampl = self.diffs
            self.sim_q.simulate(self.nprocess)
            self.q_values = self.sim_q.compute(self.quad_points, self.nprocess, check_in=False)[0]
            cornerval_q = self.sim_q.compute_corner_values()

        self.p_values = self.sim_p.compute(self.quad_points, self.nprocess, check_in=False, use_src=True)[0]

        cornerval_p = self.sim_p.compute_corner_values(use_src=True)

        nvert = len(self.sim_p.poly.vertices)

        for i in range(nvert):  # compute gradients on each edge
            start_ind, end_ind = self.quad_intervals[i]
            self.p_grads[start_ind+1:end_ind] = self.sim_p.compute_grad(self.quad_points[start_ind+1:end_ind],
                                                                        self.nprocess, use_src=True)
            self.p_grads[start_ind] = cornerval_p[1][i, :, 1, :]
            self.p_grads[end_ind] = cornerval_p[1][(i+1) % self.sim_p.poly.nedges, :, 0, :]

            if update_adjoint:
                self.q_grads[start_ind+1:end_ind] = self.sim_q.compute_grad(self.quad_points[start_ind+1:end_ind],
                                                                            self.nprocess)
                self.q_grads[start_ind] = cornerval_q[1][i, :, 1, :]
                self.q_grads[end_ind] = cornerval_q[1][(i+1) % self.sim_p.poly.nedges, :, 0, :]

                self.corner_values[i, 0] = np.real(cornerval_q[0][i].conj() * np.sum(cornerval_p[1][i, :, 0, :] *
                                                                                     self.sim_p.poly.normals[i - 1],
                                                                                     axis=-1))  # dn p * q.conj
                self.corner_values[i, 1] = np.real(cornerval_q[0][i].conj() *
                                                   np.sum(cornerval_p[1][i, :, 1, :] * self.sim_p.poly.normals[i],
                                                          axis=-1))

        self.g_op = self.sim_p.compute_src_term(self.quad_points, self.quad_ids)

    def evaluate_directional_derivative(self, V, V_corners):
        self.update_sim(update_adjoint=True)

        # shape derivative in direction of the vector field V (shape (n_int, 2))
        normals = self.sim_p.poly.normals[self.quad_ids]
        normal_term = np.sum(V*normals, axis=1)
        normal_der_p = np.sum(self.p_grads*normals[:, np.newaxis, :], axis=2)  # shape (n_int, nfreq)
        normal_der_q = np.sum(self.q_grads*normals[:, np.newaxis, :], axis=2)

        vals = normal_term * (np.sum(np.real(self.p_grads * self.q_grads.conj()), axis=(1, 2)) -
                              np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, :] ** 2 *
                                             self.p_values * self.q_values.conj()), axis=1) -
                              2 * np.sum(np.real(normal_der_p * normal_der_q.conj()), axis=1))

        # integrate vals
        vals = self.sim_p.poly.integrate(vals, self.quad_nodes, self.quad_ids)

        renormed_edges = self.sim_p.poly.edges / np.linalg.norm(self.sim_p.poly.edges, axis=1)[:, np.newaxis]
        for i in range(self.sim_p.poly.nedges):
            vals -= np.sum(np.sum(V_corners[i] * (renormed_edges[i] - renormed_edges[i-1])) *
                           np.real(self.corner_values[i, 0, :]))

        return vals

    def partial_derivative(self, i):
        # generate a function psi_i on edges i, i-1 such that psi_i(i) = 1, psi_i(i-1) = 0 = psi_i(i+1)
        psi = [np.linspace(0, 1, len(self.quad_nodes[self.quad_ids == (i-1) % self.sim_p.poly.nedges])),
               np.linspace(1, 0, len(self.quad_nodes[self.quad_ids == i]))]
        ind = [(i-1) % self.sim_p.poly.nedges, i]
        normals = [self.sim_p.poly.normals[i-1], self.sim_p.poly.normals[i]]
        res = np.zeros(2)

        for j in range(2):
            quad_ids = self.quad_ids == ind[j]
            p_val, q_val = self.p_values[quad_ids], self.q_values[quad_ids]
            p_grad, q_grad = self.p_grads[quad_ids], self.q_grads[quad_ids]
            normal = normals[j]
            normal_der_p = np.sum(p_grad * normal[np.newaxis, :], axis=1)
            normal_der_q = np.sum(q_grad * normal[np.newaxis, :], axis=1)

            vals = (np.sum(np.real(p_grad * q_grad.conj()), axis=(1, 2)) -
                    np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, :]**2 * p_val * q_val.conj()),
                           axis=1)
                    - 2*np.sum(np.real(normal_der_p * normal_der_q.conj()), axis=1)
                    )*psi[j]
            res = res + normal*self.sim_p.poly.integrate_edge(vals, self.quad_nodes[quad_ids], ind[j])

        return res

    def cost_fun(self, verbose=False):
        self.counter["ncall"] += 1
        if verbose:
            print("{} calls".format(self.counter["ncall"]))

        return 0.5*np.sum(np.real(self.diffs*self.diffs.conj()))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fs = 240.
    nfreq = 3
    pos_src = np.array([[0., 0.5]])
    dx = 0.2

    eps = 1.

    def perturbation_field(x, y):
        v = np.zeros([len(x), 2])
        indtop = np.where(y == 1.)[0]
        v[indtop, 1] = -eps * (np.abs(x[indtop]) - 1)

        return v


    height = 1.2
    init_vertices = np.array([[-1, 0], [1, 0.], [1, 1.], [0., 1], [-1, 1]])
    vertices = init_vertices + height * perturbation_field(init_vertices[:, 0], init_vertices[:, 1])

    antenna_pos = np.array([[0., 0.5]])
    mic_pos = (0.25 * np.stack([np.cos(np.linspace(0, 2 * np.pi, 25)), np.sin(np.linspace(0, 2 * np.pi, 25))], axis=1)
               + antenna_pos)

    # plt.plot(vertices[:, 0], vertices[:, 1])
    # plt.plot(mic_pos[:, 0], mic_pos[:, 1], 'o')
    # plt.plot(pos_src[:, 0], pos_src[:, 1], 'x')
    # plt.show()

    beta = [0.06 for i in range(len(vertices))]
    print("simulating")
    sim = Simulation(vertices=vertices, fs=fs, nfreq=nfreq, pos_src=pos_src, dx=dx, boundsep=1e-2, min_dist=0.005,
                     nsrc_schedule=dict(freq_marks=np.array([50.]),
                                        method='per_wavelength',
                                        nsrc=np.array([100, 100]), max_relative_error=1e-3),
                     error_thresh=1e-5, cond='robin', beta=beta, add_src=20)
    print("done")
    sim.simulate(2)
    obs, _ = sim.compute(mic_pos, 3)
    np.save('data/obs.npy', obs)

    # plot gradient
    grid = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(0, 1+height, 200))
    grid_shape = grid[0].shape
    grid = np.stack([grid[0].reshape(-1), grid[1].reshape(-1)], axis=1)
    # val, in_mask = sim.compute_grad(grid, 2)[:, 1, :], sim.poly.is_in(grid, 0.005)
    # val[~in_mask] = np.nan
    # plt.imshow(np.linalg.norm(val, axis=-1).reshape(grid_shape), extent=[-1, 1, 0, 1+height], origin='lower', vmax=5.)
    # plt.colorbar()
    # plt.show()

    init_vertices = np.array([[-1, 0], [1, 0.], [1, 1.], [0., 1], [-1, 1]])
    opt = ShapeOptimizer(init_vertices, fs, nfreq, pos_src, mic_pos, obs, nprocess=2, dx=dx,
                         sim_kwargs=dict(boundsep=1e-2, min_dist=0.005,
                                         nsrc_schedule=dict(freq_marks=np.array([50.]), method='per_wavelength',
                                                            nsrc=np.array([100, 100]), max_relative_error=1e-3),
                                         error_thresh=1e-5, cond='robin', beta=beta, add_src=20))

    # val, in_mask = opt.sim_q.compute_grad(grid, 2)[:, 1, :], opt.sim_p.poly.is_in(grid, 0.005)
    # val[~in_mask] = np.nan
    # plt.imshow(np.linalg.norm(val, axis=-1).reshape(grid_shape), extent=[-1, 1, 0, 1 + height], origin='lower', vmax=5.)
    # plt.colorbar()
    # plt.show()

    vec_field = perturbation_field(opt.quad_points[:, 0], opt.quad_points[:, 1])
    vec_field_corners = np.zeros((len(opt.sim_p.poly.vertices), 2))
    shape_der = opt.evaluate_directional_derivative(vec_field, vec_field_corners)
    print("shape derivative: ", shape_der)
    print("partial derivatives: ", [opt.partial_derivative(i) for i in range(opt.sim_p.poly.nedges)])
    # do the same using finite differences

    # opt2 = ShapeOptimizer(np.array([[-1, 0], [1, 0.], [1, 1.], [0., 1], [-1, 1]]), fs, nfreq, pos_src, mic_pos,
    #                       obs, nprocess=2, dx=dx,
    #                       sim_kwargs=dict(boundsep=1e-2, min_dist=0.005,
    #                                       nsrc_schedule=dict(freq_marks=np.array([50.]), method='per_wavelength',
    #                                                          nsrc=np.array([100, 100]), max_relative_error=1e-3),
    #                                       error_thresh=1e-5, cond='robin', beta=beta, add_src=20))
    # vald = opt2.cost_fun()
    # valdz = []
    # dzval = [0.01, 0.005, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
    #          0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003, 0.00002, 0.00001, 7e-6, 5e-6, 1e-6]
    #
    # for dz in dzval:
    #     opt2.update_sim(init_vertices + dz*perturbation_field(init_vertices[:, 0], init_vertices[:, 1]),
    #                     update_adjoint=False)
    #     valdz.append(opt2.cost_fun())
    #
    # fdval = [(valdz[i] - vald)/dzval[i] for i in range(len(dzval))]
    # plt.loglog(dzval, fdval, '-.')
    # plt.hlines(shape_der, dzval[-1], dzval[0])
    # plt.show()

    zval = np.linspace(0, 1.3, 300)
    val = np.zeros(len(zval))
    for i, t in enumerate(zval):
        print(t)
        opt.update_sim(init_vertices + t*perturbation_field(init_vertices[:, 0], init_vertices[:, 1]),
                       update_adjoint=False)
        val[i] = opt.cost_fun()

    plt.plot(zval, val)
    plt.show()

