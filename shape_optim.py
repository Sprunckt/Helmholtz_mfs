from simulation import Simulation
import numpy as np
import time


class ShapeOptimizer:
    def __init__(self, init_vertices: np.ndarray, fs: float, nfreq: int, source_pos: np.ndarray, mic_pos: np.ndarray,
                 measurements: np.ndarray, dx: float, nprocess=1, n_int=1000, origin=None,
                 sim_kwargs=None, src_ampl=None, verbose=False):
        if sim_kwargs is None:
            sim_kwargs = {"boundsep": 0.05}
        self.sim_p = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, origin=origin,
                                pos_src=source_pos, src_ampl=src_ampl, **sim_kwargs)
        sim_kwargs_q = sim_kwargs.copy()
        sim_kwargs_q["beta"] = - np.asarray(sim_kwargs_q["beta"])  # todo: handle callable case
        self.sim_q = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, pos_src=mic_pos, origin=origin,
                                **sim_kwargs_q)

        self.source_pos = source_pos
        self.mic_pos = mic_pos
        self.nmics = len(mic_pos)
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
        self.corner_values = np.zeros((len(self.sim_p.poly.vertices), nfreq), dtype=complex)
        self.verbose = verbose
        self.update_sim()

        self.counter = {"ncall":0}

    def fit(self, niter=100, jtol=1e-5, gtol=1e-6, step=1e-3, alternate=False, parametrization="half-planes"):
        log = dict(cost=[self.cost_fun()], grad=[], vert=[self.sim_p.poly.vertices.copy()])
        if parametrization == "half-planes":
            gradfun = self.partial_derivative_proji
        elif parametrization == "vertices":
            gradfun = self.partial_derivative
        else:
            raiseValueError("unknown parametrization, accepted values:, 'half-planes', 'vertices'")

        it, cost, gradcost, rotation_ind = 0, np.inf, np.inf, 0
        while it < niter and jtol < cost and gtol < gradcost:
            t1 = time.time()
            grad = np.zeros((len(self.sim_p.poly.vertices), 2))
            if alternate:
                grad[rotation_ind] = gradfun(rotation_ind)
            else:
                for i in range(len(self.sim_p.poly.vertices)):
                    grad[i] = gradfun(i)
            tgrad = time.time() - t1

            if parametrization == "half-planes":
                new_param = self.sim_p.poly.projs - step*grad
            else:
                new_param = self.sim_p.poly.vertices - step*grad
            self.update_sim(new_vertices=new_param, update_adjoint=True, parametrization=parametrization)
            cost = self.cost_fun()
            gradcost = np.linalg.norm(grad)
            log["cost"].append(cost)
            log["grad"].append(grad)
            log["vert"].append(self.sim_p.poly.vertices.copy())
            t2 = time.time()
            if self.verbose:
                # print("rel_error: ", self.sim_p.evaluate_error(freq_ind=-1))
                print("it: ", it, "descent ind: ", rotation_ind)
                print("cost: ", cost, "grad_norm: ", gradcost, "ind_desc: ", rotation_ind, 
                      "tgrad: ", tgrad, "titer: ", t2-t1)
                print("grad: ", grad)
                print("current polygon: \n", self.sim_p.poly.vertices)
                print("mics in poly: ", self.sim_p.poly.is_in(self.mic_pos))
                print("source in poly: ", self.sim_p.poly.is_in(self.source_pos.reshape([-1, 2])))
            rotation_ind = (rotation_ind + 1) % self.sim_p.poly.nedges 
            it += 1

        return self.sim_p.poly.vertices, log

    def update_sim(self, new_vertices=None, update_adjoint=True, parametrization="vertices"):
        if new_vertices is not None:
            t1 = time.time()
            self.sim_p.update_poly(new_vertices, parametrization=parametrization)
            self.sim_q.update_poly(new_vertices, parametrization=parametrization)

        self.sim_p.reset()
        self.sim_p.simulate(self.nprocess, verbose=False)

        mic_val, _ = self.sim_p.compute(self.mic_pos, self.nprocess)

        self.diffs = mic_val - self.observations

        if update_adjoint:
            self.sim_q.reset()
            self.sim_q.src_ampl = -self.diffs
            self.sim_q.simulate(self.nprocess, verbose=False)
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

                self.corner_values[i] = np.real(cornerval_q[0][i].conj() * np.sum(cornerval_p[1][i, :, 0, :] *
                                                                                  self.sim_p.poly.normals[i - 1],
                                                                                  axis=-1))  # dn p * q.conj

    def evaluate_directional_derivative(self, V, V_corners):
        # shape derivative in direction of the vector field V (shape (n_int, 2))
        normals = self.sim_p.poly.normals[self.quad_ids]
        normal_term = np.sum(V*normals, axis=1)
        beta_val = self.sim_p.mfs.beta[self.quad_ids]
        vals = normal_term * (np.sum(np.real(self.p_grads * self.q_grads.conj()), axis=(1, 2)) -
                              np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, :] ** 2 *
                                             self.p_values * self.q_values.conj()), axis=1) +
                              2*np.sum(np.real(beta_val*beta_val.conj())[:, np.newaxis] *
                                       self.sim_p.wave_numbers[np.newaxis, :]**2 *
                                       np.real(self.p_values * self.q_values.conj()), axis=1))

        # integrate vals
        vals = self.sim_p.poly.integrate(vals, self.quad_nodes, self.quad_ids)
        renormed_edges = self.sim_p.poly.edges / np.linalg.norm(self.sim_p.poly.edges, axis=1)[:, np.newaxis]
        for i in range(self.sim_p.poly.nedges):
            vals += np.sum(np.sum(V_corners[i] * (renormed_edges[i] - renormed_edges[i-1])) *
                           np.real(self.corner_values[i, :]))

        return vals / self.nmics

    def partial_derivative(self, i):
        # generate a function psi_i on edges i, i-1 such that psi_i(i) = 1, psi_i(i-1) = 0 = psi_i(i+1)
        psi = [np.linspace(0, 1, len(self.quad_nodes[self.quad_ids == (i-1) % self.sim_p.poly.nedges])),
               np.linspace(1, 0, len(self.quad_nodes[self.quad_ids == i]))]
        ind = [(i-1) % self.sim_p.poly.nedges, i]
        normals = [self.sim_p.poly.normals[i-1], self.sim_p.poly.normals[i]]
        res = np.zeros(2)
        beta_val_tot = self.sim_p.mfs.beta[self.quad_ids]
        for j in range(2):
            quad_ids = np.where(self.quad_ids == ind[j])[0]
            p_val, q_val = self.p_values[quad_ids], self.q_values[quad_ids]
            p_grad, q_grad = self.p_grads[quad_ids], self.q_grads[quad_ids]
            normal = normals[j]
            beta_val = beta_val_tot[quad_ids]
            vals = (np.sum(np.real(p_grad * q_grad.conj()), axis=(1, 2)) -
                    np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, :]**2 * p_val * q_val.conj()), axis=1) +
                    2 * np.sum(np.real(beta_val * beta_val.conj())[:, np.newaxis] *
                               self.sim_p.wave_numbers[np.newaxis, :]**2 *
                               np.real(p_val * q_val.conj()), axis=1))*psi[j]
            res = res + normal*self.sim_p.poly.integrate_edge(vals, self.quad_nodes[quad_ids], ind[j])
        renormed_edge1 = self.sim_p.poly.edges[i] / np.linalg.norm(self.sim_p.poly.edges[i])
        renormed_edge2 = self.sim_p.poly.edges[i-1] / np.linalg.norm(self.sim_p.poly.edges[i-1])

        res = res + (renormed_edge1 - renormed_edge2)*np.sum(np.real(self.corner_values[i, :]))
        return res / self.nmics

    def partial_derivative_proji(self, i):
        V, V_corners = np.zeros((self.n_int, 2)), np.zeros((self.sim_p.poly.nedges, 2))
        part_der = np.zeros(2)
        quad_ids = np.where(self.quad_ids == i)[0]
        nquad, indnext = len(quad_ids), (i+1) % self.sim_p.poly.nedges
        px, origin = self.sim_p.poly.projs[i], self.sim_p.poly.origin
        edgei, edgeip = self.sim_p.poly.edges[i-1], self.sim_p.poly.edges[indnext]
        dot_prodi, dot_prodip = np.sum(edgei*(px - origin)), np.sum(edgeip*(px - origin))
        vi, vip = self.sim_p.poly.vertices[i], self.sim_p.poly.vertices[indnext]

        epsix = (2*px[0] - origin[0] - vi[0])/dot_prodi - edgei[0]*(px[0]-origin[0])*(px[0]-vi[0])/dot_prodi**2
        epsiy = (2*px[1] - origin[1] - vi[1])/dot_prodi - edgei[1]*(px[1]-origin[1])*(px[1]-vi[1])/dot_prodi**2
        epsixp = (2*px[0] - origin[0] - vip[0])/dot_prodip - edgeip[0]*(px[0]-origin[0])*(px[0]-vip[0])/dot_prodip**2
        epsiyp = (2*px[1] - origin[1] - vip[1])/dot_prodip - edgeip[1]*(px[1]-origin[1])*(px[1]-vip[1])/dot_prodip**2
        alpha = np.linspace(1, 0, nquad)  # linear mapping for edge parametrization
        V[quad_ids] = alpha[:, np.newaxis]*epsix*edgei + (1-alpha)[:, np.newaxis]*epsixp*edgeip
        V_corners[i], V_corners[indnext] = epsix * edgei, epsixp * edgeip
        part_der[0] = self.evaluate_directional_derivative(V, V_corners)

        V[quad_ids] = alpha[:, np.newaxis]*epsiy*edgei + (1-alpha)[:, np.newaxis]*epsiyp*edgeip
        V_corners[i], V_corners[indnext] = epsiy * edgei, epsiyp * edgeip
        part_der[1] = self.evaluate_directional_derivative(V, V_corners)

        return part_der / self.nmics

    def cost_fun(self, verbose=False):
        self.counter["ncall"] += 1
        if verbose:
            print("{} calls".format(self.counter["ncall"]))

        return 0.5*np.sum(np.real(self.diffs*self.diffs.conj())) / self.nmics


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

