from simulation import Simulation
import numpy as np
from scipy.optimize import minimize, differential_evolution
from utils import mod2pi


class ShapeOptimizer:
    def __init__(self, init_vertices: np.ndarray, fs: float, nfreq: int, source_pos: np.ndarray, mic_pos: np.ndarray,
                 measurements: np.ndarray, dx: float, nprocess=1, n_int=1000, sim_kwargs=None):
        if sim_kwargs is None:
            sim_kwargs = {"boundsep": 0.05}
        self.sim_p = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, pos_src=source_pos, **sim_kwargs)
        self.sim_q = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, pos_src=mic_pos, **sim_kwargs)

        self.source_pos = source_pos
        self.mic_pos = mic_pos
        self.antenna_radius = np.max(np.linalg.norm(self.mic_pos, axis=1))
        self.nprocess = nprocess
        self.observations = measurements  # shape len(freqs)
        self.diffs = np.full(nfreq, np.inf, dtype=complex)
        self.n_int = n_int  # number of quadrature points on the boundary
        self.quad_points, _, self.quad_ids, self.quad_nodes = self.sim_p.poly.sample(self.n_int, return_nodes=True,
                                                                                     min_dist=0.)

        self.quad_intervals = np.zeros(self.sim_p.poly.nedges+1, dtype=int)
        self.quad_intervals[1:-1] = np.where(np.diff(self.quad_ids) != 0)[0]
        self.quad_intervals[-1], self.quad_intervals[0] = -1, -1

        self.p_values = np.zeros((self.n_int, nfreq), dtype=complex)
        self.q_values = np.zeros((self.n_int, nfreq), dtype=complex)
        self.p_grads = np.zeros((self.n_int, nfreq, 2), dtype=complex)
        self.q_grads = np.zeros((self.n_int, nfreq, 2), dtype=complex)
        self.corner_values = np.zeros((len(self.sim_p.poly.vertices), 2, nfreq), dtype=complex)
        self.corner_grads = np.zeros((len(self.sim_p.poly.vertices), 2, nfreq, 2), dtype=complex)
        self.g_op = np.zeros((self.n_int, nfreq), dtype=complex)
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

        self.p_values = self.sim_p.compute(self.quad_points, self.nprocess, check_in=False, use_src=False)[0]

        cornerval_p = self.sim_p.compute_corner_values(use_src=False)

        nvert = len(self.sim_p.poly.vertices)
        for i in range(nvert):  # compute gradients on each edge
            start_ind, end_ind = self.quad_intervals[i]+1, self.quad_intervals[i+1]
            self.p_grads[start_ind+1:end_ind] = self.sim_p.compute_grad(self.quad_points[start_ind+1:end_ind],
                                                                        self.nprocess, use_src=False)
            self.p_grads[start_ind] = cornerval_p[1][i, :, 1, :]
            self.p_grads[end_ind] = cornerval_p[1][(i+1) % self.sim_p.poly.nedges, :, 0, :]

            if update_adjoint:
                self.q_grads[start_ind + 1:end_ind] = self.sim_q.compute_grad(self.quad_points[start_ind + 1:end_ind],
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
                              np.sum(np.real(self.g_op * self.q_values.conj()), axis=1) -
                              2 * np.sum(np.real(normal_der_p * normal_der_q.conj()), axis=1)
                              )
        # integrate vals
        vals = self.sim_p.poly.integrate(vals, self.quad_nodes, self.quad_ids)

        # todo: check if this is correct with the np.roll
        # vals -= np.sum(V_corners[:, :, np.newaxis] *
        #                (np.roll(self.sim_p.poly.edges[:, :, np.newaxis], 1, axis=0) *
        #                 np.real(self.corner_values[:, 0, np.newaxis, :]) +  # dnp \bar q (a_i^-)
        #                 self.sim_p.poly.edges[:, :, np.newaxis]*np.real(self.corner_values[:, 1, np.newaxis, :])))

        # for i in range(self.sim_p.poly.nedges):
        #     vals -= (np.sum(self.sim_p.poly.edges[i-1, :, np.newaxis] * V_corners[i, :, np.newaxis], -1) *
        #                     np.real(self.corner_values[i, 0, :]) +
        #              np.sum(self.sim_p.poly.edges[i, :, np.newaxis] * V_corners[i, :, np.newaxis], -1) *
        #              np.real(self.corner_values[i, 1, :]))

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
            g_op = self.g_op[quad_ids]

            vals = (np.sum(np.real(p_grad * q_grad.conj()), axis=(1, 2)) -
                    np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, :]**2 * p_val * q_val.conj()),
                           axis=1)
                    - np.sum(np.real(g_op * q_val.conj()), axis=1)
                    - 2*np.sum(np.real(normal_der_p * normal_der_q.conj()), axis=1)
                    )*psi[j]
            res = res + normal*self.sim_p.poly.integrate_edge(vals, self.quad_nodes[quad_ids], ind[j])

        return res

    def cost_fun(self, verbose=False):
        self.counter["ncall"] += 1
        if verbose:
            print("{} calls".format(self.counter["ncall"]))

        return 0.5*np.sum(np.real(self.diffs*self.diffs.conj()))

    def cost_fun_params(self, params, counter):
        counter["ncall"] += 1
        phi, theta = params[:self.sim_p.poly.nedges], mod2pi(params[self.sim_p.poly.nedges:])
        self.sim_p.update_params(phi, theta)
        self.sim_p.reset()
        self.sim_p.simulate(self.nprocess)
        obs_sim, _ = self.sim_p.compute(self.mic_pos, self.nprocess)
        diff = obs_sim - self.observations
        err = 0.5*np.sum(diff*diff.conj()).real
        print("{} calls, err {}, params".format(counter["ncall"], err), phi, theta)

        return err

    def cost_fun_tot(self, params, counter):
        counter["ncall"] += 1
        phi, theta = params[:self.sim.poly.nedges], mod2pi(params[self.sim.poly.nedges:])
        self.sim.update_params(phi, theta)
        self.sim.reset()
        self.sim.simulate(self.nprocess)
        obs_sim, _ = self.sim.compute(self.mic_pos, self.nprocess)
        diff = obs_sim - self.observations  # shape (m, nfreq)

        print("{} calls".format(counter["ncall"]))

        return diff
