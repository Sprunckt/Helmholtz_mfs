from simulation import Simulation
import numpy as np
import time
import json
from utils import rotate, mod2pi, random_poly

def array_to_list(arr):
    """Converts arr to a list if arr is an ndarray, leave as it is otherwise."""
    if type(arr) == np.ndarray:
        return arr.tolist()
    else:
        return arr


def dict_to_json(dico, path):
    dd = dico.copy()
    for key in dd:
        dd[key] = array_to_list(dd[key])
    fd = open(path, 'w')
    json.dump(dd, fd)
    fd.close()


class ShapeOptimizer:
    def __init__(self, init_vertices: np.ndarray, fs: float, nfreq: int, source_pos: np.ndarray, mic_pos: np.ndarray,
                 measurements: np.ndarray, dx: float, nprocess=1, n_int=1000, parametrization="half-planes",
                 sim_kwargs=None, src_ampl=None, verbose=False):
        self.parametrization = parametrization

        if parametrization == "half-planes":
            origin = np.reshape(source_pos.copy(), -1)
        else:
            origin = None

        if sim_kwargs is None:
            sim_kwargs = {"boundsep": 0.05}
        self.sim_p = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, origin=origin,
                                pos_src=source_pos, src_ampl=src_ampl, **sim_kwargs)
        sim_kwargs_q = sim_kwargs.copy()
        sim_kwargs_q["beta"] = - np.asarray(sim_kwargs_q["beta"])  # todo: handle callable case
        self.sim_q = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, pos_src=mic_pos,
                                origin=origin, **sim_kwargs_q)

        self.source_pos = source_pos
        self.mic_pos = mic_pos
        self.nmics = len(mic_pos)
        self.nprocess = nprocess
        self.observations = measurements  # shape len(freqs)
        self.diffs = None  # difference between observations and simulated values, only active freqs and mics
        self.n_int = n_int  # number of quadrature points on the boundary

        self.quad_points, self.quad_ids, self.quad_nodes = None, None, None
        self.quad_intervals = np.zeros([self.sim_p.poly.nedges, 2], dtype=int)
        self._update_quadrature_points()

        self.p_grads, self.q_grads, self.p_values, self.q_values = None, None, None, None
        self.corner_values = None
        self.verbose = verbose

        # active frequency indices used in the cost function/gradient computations
        self._active_ind = np.arange(1, self.sim_p.nfreqs)
        self._possible_ind = np.arange(1, self.sim_p.nfreqs)
        self._nactive = self.sim_p.nfreqs - 1
        self.max_freq = self.sim_p.freqs[-1]
        self.min_freq = 0.

        self._active_mic_ind = np.arange(self.nmics)
        self._nactive_mic = self.nmics
        
        self.nfreqs = nfreq
        self.nedges = self.sim_p.poly.nedges
        self._generator = np.random.default_rng()

    def get_params(self):
        return self.sim_p.poly.get_params(self.parametrization).copy()


    def line_search(self, step, step_decrease, neval_search, update_fun, grad_fun, bounds=None, convex_proj=False,
                    in_constraints=None, verbose=False):
        if bounds is None:
            bounds = (-10., 10.)

        curr_params = self.get_params()
        best_cost = np.inf
        start_cost = self.cost_fun()
        curr_step = step
        init_decrease = False  # True if costfun decreases at first step
        grad = grad_fun()
        print("starting cost: ", start_cost) 
        for i in range(neval_search):
            new_params = update_fun(curr_params, curr_step, grad)  # apply update
            if verbose:
                print("step: ", curr_step)
            success = self.update_sim(new_params, update_state=True, update_adjoint=False, convex_proj=convex_proj,
                                      bounds=bounds, check_in=in_constraints, allow_reduce=False)
            curr_step = curr_step * step_decrease

            if not success:
                print("Polygon update failed, skipping")
                continue
            new_cost = self.cost_fun()
            if verbose:
                print("cost: ", new_cost)

            if new_cost < start_cost:
                best_cost = new_cost
                if best_cost < start_cost:
                    if i == 0:
                        init_decrease = True
                    if verbose:
                        print("costfunction decreased, early stopping")
                    break

        if best_cost == np.inf:
            print("No valid step found to update polygon, failure")
        else:
            self._update_adjoint()

        return best_cost, grad, init_decrease

    def _update_quadrature_points(self):
        self.quad_points, _, self.quad_ids, self.quad_nodes = self.sim_p.poly.sample(self.n_int, return_nodes=True,
                                                                                     min_dist=0.)
        interv = np.where(np.diff(self.quad_ids) != 0)[0]
        self.quad_intervals[0], self.quad_intervals[-1] = [0, interv[0]], [interv[-1] + 1, len(self.quad_ids) - 1]
        self.quad_intervals[1:-1] = np.stack([interv[:-1] + 1, interv[1:]], axis=1)

    def _reset_active_indices(self):
        self._active_ind = np.arange(1, self.sim_p.nfreqs)
        self._nactive = self.sim_p.nfreqs - 1
        self._active_mic_ind = np.arange(self.nmics)
        self._nactive_mic = self.nmics

    def _get_active_indices(self):
        return self._active_ind, self._active_mic_ind
    
    def _set_active_indices(self, freq_ind, mic_ind):
        self._nactive = len(freq_ind)
        self._nactive_mic = len(mic_ind)
        self._active_ind = freq_ind.copy()
        self._active_mic_ind = mic_ind.copy()

    def _update_active_indices(self, schedule):  # increase frequency range
        if schedule["update_ind"] < len(schedule["max_freqs"]):
            # update min, max frequencies
            self.max_freq = schedule["max_freqs"][schedule["update_ind"]]
            self.min_freq = schedule["min_freqs"][schedule["update_ind"]]
            print("updating min, max frequencies to: ", (self.min_freq, self.max_freq))
                
            schedule["update_ind"] += 1
            self._possible_ind = self.sim_p.get_ind(self.min_freq, self.max_freq)
            self._active_ind = self._possible_ind.copy()
            self._nactive = len(self._active_ind)
            return True
        else: 
            return False 
        
    def _batch(self, schedule):
        if schedule["freq_batch_size"] > 0:
            self._active_ind = self._generator.choice(self._possible_ind, schedule["freq_batch_size"], replace=False)
            self._nactive = schedule["freq_batch_size"]
        if schedule["mic_batch_size"] > 0:
            self._active_mic_ind = self._generator.choice(np.arange(self.nmics), schedule["mic_batch_size"],
                                                          replace=False)
            self._nactive_mic = schedule["mic_batch_size"]

    def fit(self, niter=100, jtol=1e-5, step=10, neval_search=3, step_decrease=0.85,
            decrease_period=10, increase_period=10, step_increase=1.2, patience=5, decrease_patience=0.01, freq_patience=20,
            convex_proj=False, rotation_step=50, translation_step=10,
            schedule=None, save_path=None, save_freq=20, compute_full_cost=0):
        """Set alternate to inf to compute full grad at each iteration, 0 to always alternate the vertex number.
        schedule is a dictionary with keys: mic_batch_size, freq_batch_size, max_freqs, update_its. max_freqs and
        update_its are lists of the same length, indicating the maximum frequencies and the iteration at which to update
        the maximume frequency to the next value."""
        log = dict(grad=[], vert=[self.sim_p.poly.vertices.tolist()], signals=[])

        if self.parametrization == "half-planes":  # the origin is set at the source position
            in_constraints = self.mic_pos
        else:  # in that case also check the source position
            in_constraints = np.concatenate([self.mic_pos, self.source_pos], axis=0)

        if compute_full_cost > 0:
            self._reset_active_indices()
            self.update_sim(update_state=True, update_adjoint=False)
            log["full_cost"] = [self.cost_fun()]

        if schedule is None:  # default schedule : no batch, start at maximum frequency
            schedule = dict(mic_batch_size=0, freq_batch_size=0, max_freqs=[self.max_freq], 
                            min_freqs=[self.min_freq], step_reset=None)
        
        print("Schedule: ", schedule)
        schedule["update_ind"] = 0
        self._update_active_indices(schedule)
        self.update_sim()

        log["schedule"] = schedule
        assert len(schedule["max_freqs"]) == len(schedule["min_freqs"]), \
            "max_freqs and min_freqs should have same length"

        act_ind, act_ind_mic = self._get_active_indices()
        self._set_active_indices(act_ind, np.arange(self.nmics))
        log["cost"] = [self.cost_fun()]
        signals = self.get_signals()
        log["signals_re"], log["signals_im"] = [signals.real.tolist()], [signals.imag.tolist()]
        self._set_active_indices(act_ind, act_ind_mic)

        log["max_freq"], log["min_freq"] = [self.max_freq], [self.min_freq]
        log["ind_min"], log["ind_max"] = [int(np.min(self._active_ind))], [int(np.max(self._active_ind))]
        log["descent_type"] = [0]
        log["frequency_change"] = []
        it, cost, gradcost, decrease_ind, increase_ind = 0, np.inf, np.inf, 0, 0
        print("initial cost:", log["cost"][0])
        descent_type = 0  # 0: shape step, 1: translation step, 2: rotation step
        steps = [step, translation_step, rotation_step]
        descent_type_str = ["shape", "translation", "rotation"]
        patience_counter, freq_patience_counter = 0, 0
        while it < niter and jtol < cost:
            t1 = time.time()
            if self.verbose:
                print("it: {}, descent type: {}".format(it, descent_type_str[descent_type]))

            # line search: reduce step size until cost function decreases
            if descent_type == 0:  # moving vertices / edges
                cost, grad, init_decreased = self.line_search(steps[descent_type], step_decrease, neval_search,
                                                              update_fun=_step_update_params,
                                                              grad_fun=self.get_gradient, verbose=self.verbose,
                                                              in_constraints=in_constraints, convex_proj=convex_proj)
            elif descent_type == 1:  # translation step
                cost, grad, init_decreased = self.line_search(steps[descent_type], step_decrease, neval_search,
                                                              update_fun=_step_update_translation,
                                                              grad_fun=self.partial_derivative_translation,
                                                              in_constraints=in_constraints)
            elif descent_type == 2:  # rotation step
                cost, grad, init_decreased = self.line_search(steps[descent_type], step_decrease, neval_search,
                                                              update_fun=_step_update_rotation,
                                                              grad_fun=self.partial_derivative_rotation,
                                                              in_constraints=in_constraints)
            else:
                raise ValueError("Descent type not recognized")

            if cost == np.inf:
                print("Optimization failed, stopping")
                break

            if init_decreased:
                decrease_ind = 0  # reset counter, cost function has decreased at first it of line search
                increase_ind += 1
                if increase_ind == increase_period:
                    steps[descent_type] *= step_increase
                    increase_ind = 0
            else:
                decrease_ind += 1
                increase_ind = 0
                if decrease_ind == decrease_period:
                    steps[descent_type] *= step_decrease
                    decrease_ind = 0

            gradcost = np.linalg.norm(grad)

            act_ind, act_ind_mic = self._get_active_indices()
            self._set_active_indices(act_ind, np.arange(self.nmics))
            cost = self.cost_fun()
            log["cost"].append(cost)
            signals = self.get_signals()
            log["signals_re"].append(signals.real.tolist())
            log["signals_im"].append(signals.imag.tolist())
            self._set_active_indices(act_ind, act_ind_mic)  
            log["ind_min"].append(int(np.min(self._active_ind))) 
            log["ind_max"].append(int(np.max(self._active_ind)))

            log["descent_type"].append(descent_type)
            log["grad"].append(grad.tolist())
            log["vert"].append(self.sim_p.poly.vertices.tolist())
            
            log["max_freq"].append(self.max_freq)
            log["min_freq"].append(self.min_freq)

            # updating descent type based on patience
            if patience_counter >= patience:  # meaning at least 'patience' iterations since last descent change
                if np.abs((log["cost"][-1] - log["cost"][-patience]) / log["cost"][-patience]) < decrease_patience:
                    decrease_ind, increase_ind, patience_counter = 0, 0, 0
                    descent_type = (descent_type + 1) % 3
                    print("changing descent type to ", descent_type_str[descent_type])
            else:
                patience_counter += 1
            
            if freq_patience_counter >= freq_patience:
                if np.abs((log["cost"][-1] - log["cost"][-freq_patience]) / log["cost"][-freq_patience]) < decrease_patience:
                    decrease_ind, increase_ind, patience_counter, freq_patience_counter = 0, 0, 0, 0
                    descent_type = 0  # go back to shape optimization
                    print("changing descent type to ", descent_type_str[descent_type])
                    updated = self._update_active_indices(schedule)  # increase frequency range
                    if updated:
                        self.update_sim(update_state=True, update_adjoint=True)
                        log["frequency_change"].append(it)
            else:
                freq_patience_counter += 1

            if (schedule["freq_batch_size"] > 0) or (schedule["freq_batch_size"] > 0):
                self._batch(schedule)
                self.update_sim(update_adjoint=True, update_state=True)
            t2 = time.time()

            if (compute_full_cost > 0) and (((it + 1) % compute_full_cost) == 0):
                self._reset_active_indices()
                self.update_sim(update_state=True, update_adjoint=False)
                log["full_cost"].append(self.cost_fun())
                print("full cost: ", log["full_cost"])

            if self.verbose:
                print("iteration {} done".format(it))
                print("cost: ", cost, "grad_norm k-1: ", gradcost, "titer: ", t2-t1)
                print("current polygon: \n", self.sim_p.poly.vertices)
            it += 1

            if (save_path is not None) and (it % save_freq == 0):
                dict_to_json(log, save_path)

        if save_path is not None:
            dict_to_json(log, save_path)

        return self.sim_p.poly.vertices, log

    def _update_state(self):
        self.sim_p.set_freq_ind_to_simulate(self._active_ind)
        self.sim_p.simulate(self.nprocess, verbose=False)

        mic_val = self.sim_p.compute(self.mic_pos[self._active_mic_ind], self.nprocess, freq_ind=self._active_ind)
        self.diffs = mic_val - self.observations[self._active_mic_ind[:, np.newaxis], self._active_ind[np.newaxis, :]]

        self.p_values = self.sim_p.compute(self.quad_points, self.nprocess, freq_ind=self._active_ind, use_src=True)
        self.cornerval_p = self.sim_p.compute_corner_values(use_src=True, freq_ind=self._active_ind)

        nvert = len(self.sim_p.poly.vertices)
        self.p_grads = np.zeros((self.n_int, self._nactive, 2), dtype=complex)

        for i in range(nvert):  # compute gradients on each edge
            start_ind, end_ind = self.quad_intervals[i]
            self.p_grads[start_ind+1: end_ind] = self.sim_p.compute_grad(self.quad_points[start_ind+1: end_ind],
                                                                         self.nprocess, freq_ind=self._active_ind)
            self.p_grads[start_ind] = self.cornerval_p[1][i, :, 1, :]
            self.p_grads[end_ind] = self.cornerval_p[1][(i + 1) % self.sim_p.poly.nedges, :, 0, :]

    def _update_adjoint(self):
        self.sim_q.set_freq_ind_to_simulate(self._active_ind)
        self.sim_q.src_ampl[:, :] = 0.  # reset source amplitudes
        # update source amplitudes with observation errors, only consider active mics and freqs
        self.sim_q.src_ampl[self._active_mic_ind[:, np.newaxis], self._active_ind[np.newaxis, :]] = - self.diffs
        self.sim_q.simulate(self.nprocess, verbose=False)
        self.q_values = self.sim_q.compute(self.quad_points, self.nprocess, freq_ind=self._active_ind)

        cornerval_q = self.sim_q.compute_corner_values(freq_ind=self._active_ind)

        self.q_grads = np.zeros((self.n_int, self._nactive, 2), dtype=complex)
        self.corner_values = np.zeros((self.sim_p.poly.nedges, self._nactive))
        for i in range(len(self.sim_p.poly.vertices)):  # compute gradients on each edge
            start_ind, end_ind = self.quad_intervals[i]

            self.q_grads[start_ind + 1:end_ind] = self.sim_q.compute_grad(self.quad_points[start_ind + 1:end_ind],
                                                                          self.nprocess, freq_ind=self._active_ind)
            self.q_grads[start_ind] = cornerval_q[1][i, :, 1, :]
            self.q_grads[end_ind] = cornerval_q[1][(i + 1) % self.sim_p.poly.nedges, :, 0, :]

            self.corner_values[i] = np.real(cornerval_q[0][i].conj() * np.sum(self.cornerval_p[1][i, :, 0, :] *
                                                                              self.sim_p.poly.normals[i - 1],
                                                                              axis=-1))  # dn p * q.conj

    def update_sim(self, new_params=None, update_state=True, update_adjoint=True,
                   bounds=None, check_in=None, convex_proj=False, allow_reduce=False):
        if new_params is not None:
            success = self.sim_p.update_poly(new_params, parametrization=self.parametrization, convex_proj=convex_proj,
                                             bounds=bounds, check_in=check_in)
            if (not allow_reduce) and (self.sim_p.poly.nedges != self.nedges): 
                success = False
            if success:
                self.sim_q.update_poly(new_params, parametrization=self.parametrization, convex_proj=convex_proj)
                self._update_quadrature_points()

        else:
            success = True

        if success:
            if update_state:
                self._update_state()
            if update_adjoint:
                self._update_adjoint()
        return success
        
    def evaluate_directional_derivative(self, V, V_corners):
        # shape derivative in direction of the vector field V (shape (n_int, 2))
        normals = self.sim_p.poly.normals[self.quad_ids]
        normal_term = np.sum(V*normals, axis=1)
        beta_val = self.sim_p.mfs.beta[self.quad_ids]
        vals = normal_term * (np.sum(np.real(self.p_grads * self.q_grads.conj()), axis=(1, 2)) -
                              np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, self._active_ind] ** 2 *
                                             self.p_values * self.q_values.conj()), axis=1) +
                              2*np.sum(np.real(beta_val*beta_val.conj())[:, np.newaxis] *
                                       self.sim_p.wave_numbers[np.newaxis, self._active_ind]**2 *
                                       np.real(self.p_values * self.q_values.conj()), axis=1))

        # integrate vals
        vals = self.sim_p.poly.integrate(vals, self.quad_nodes, self.quad_ids)
        renormed_edges = self.sim_p.poly.edges / np.linalg.norm(self.sim_p.poly.edges, axis=1)[:, np.newaxis]
        for i in range(self.sim_p.poly.nedges):
            vals += np.sum(np.sum(V_corners[i] * (renormed_edges[i] - renormed_edges[i-1])) *
                           np.real(self.corner_values[i, :]))

        return vals / self._nactive_mic / self._nactive

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
                    np.sum(np.real(self.sim_p.wave_numbers[np.newaxis, self._active_ind]**2 * p_val * q_val.conj()),
                           axis=1) +
                    2 * np.sum(np.real(beta_val * beta_val.conj())[:, np.newaxis] *
                               self.sim_p.wave_numbers[np.newaxis, self._active_ind]**2 *
                               np.real(p_val * q_val.conj()), axis=1))*psi[j]
            res = res + normal*self.sim_p.poly.integrate_edge(vals, self.quad_nodes[quad_ids], ind[j])
        renormed_edge1 = self.sim_p.poly.edges[i] / np.linalg.norm(self.sim_p.poly.edges[i])
        renormed_edge2 = self.sim_p.poly.edges[i-1] / np.linalg.norm(self.sim_p.poly.edges[i-1])

        res = res + (renormed_edge1 - renormed_edge2)*np.sum(np.real(self.corner_values[i, :]))
        return res / self._nactive_mic / self._nactive

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

        return part_der  # no normalization as it is done in evaluate_directional_derivative

    def partial_derivative_rotation(self, center=None):
        """Partial derivative according to a rotation centered at the given point"""
        V, V_corners = np.zeros((self.n_int, 2)), np.zeros((self.sim_p.poly.nedges, 2))
        rot_der = np.array([[0, 1], [-1, 0]])
        if center is None:
            center = np.mean(self.sim_p.poly.vertices, axis=0)
        # shape (nedges, 2)
        origin_vect_rot_der = np.sum(rot_der[None, :, :] *
                                     (self.sim_p.poly.vertices - np.reshape(center, [1, 2]))[:, None, :], axis=2)
        for i in range(self.sim_p.poly.nedges):
            quad_ids = np.where(self.quad_ids == i)[0]
            nquad = len(quad_ids)
            alpha = np.linspace(1, 0, nquad)

            V[quad_ids] = (alpha[:, None]*origin_vect_rot_der[i] +
                           (1 - alpha)[:, None]*origin_vect_rot_der[(i+1) % self.sim_p.poly.nedges])
            V_corners[i] = origin_vect_rot_der[i]

        return self.evaluate_directional_derivative(V, V_corners)

    def partial_derivative_translation(self):
        der = np.zeros(2)
        V, V_corners = np.zeros((self.n_int, 2)), np.zeros((self.sim_p.poly.nedges, 2))
        V[:, 0], V_corners[:, 0] = 1., 1.
        der[0] = self.evaluate_directional_derivative(V, V_corners)
        V[:, 0], V_corners[:, 0] = 0., 0.
        V[:, 1], V_corners[:, 1] = 1., 1.
        der[1] = self.evaluate_directional_derivative(V, V_corners)
        return der

    def cost_fun(self):
        return 0.5*np.sum(np.real(self.diffs * self.diffs.conj())) / self._nactive_mic / self._nactive

    def cost_fun_full(self):
        return 0.5*np.real(self.diffs*self.diffs.conj())
    
    def get_signals(self):
        return self.sim_p.compute(self.mic_pos[self._active_mic_ind], self.nprocess, freq_ind=self._active_ind)

    def _get_gradfun(self):
        if self.parametrization == "half-planes":
            gradfun = self.partial_derivative_proji
        elif self.parametrization == "vertices":
            gradfun = self.partial_derivative
        else:
            raise ValueError("unknown parametrization, accepted values:, 'half-planes', 'vertices'")
        return gradfun 
    
    def get_gradient(self):
        gradfun = self._get_gradfun()
        grad = np.zeros((len(self.sim_p.poly.vertices), 2))

        for i in range(len(self.sim_p.poly.vertices)):
            grad[i] = gradfun(i)
        return grad 

    def evaluate_costfun_dir(self, direction, step, nb_step):
        res = np.zeros(nb_step)
        params = self.get_params()
        for i in range(nb_step):
            print("step {}, displacement {}".format(i, step*(i+1)))
            params = params + step*direction 
            self.update_sim(params, update_adjoint=False, update_state=True)
            res[i] = self.cost_fun()
        return res

    def random_multiscale_fit(self, bounds, n_steps_final, n_steps_ini, max_freq_ini, min_freq_ini,
                              fit_params, n_ini=10, save_path=None):
        """Optimize shape using randomized initialization and refining frequencies at each restart.
        - bounds: list of 2 tuples (for x and y), lower and upper bounds for the random vertices.
        - n_steps_final: number of steps for the final (refined) optimization
        - n_steps_ini: number of steps for the initial random optimization
        - max_freq_ini, min_freq_ini: initial frequency range
        - fit_params: dictionary containing the optimization parameters (should contain a valid schedule)
        - n_ini: number of random initializations
        - save_path: path to save the optimization results
        """

        in_constraints = np.concatenate([self.mic_pos, self.source_pos], axis=0)
        max_freqs_copy = fit_params["schedule"]["max_freqs"].copy()
        min_freqs_copy = fit_params["schedule"]["min_freqs"].copy()
        fit_params["niter"] = n_steps_ini
        fit_params["schedule"]["max_freqs"], fit_params["schedule"]["min_freqs"] = [max_freq_ini], [min_freq_ini]
        best_cost = np.inf
        old_steps = [fit_params["step"],fit_params["rotation_step"], fit_params["translation_step"]]
        print("Starting random initialization")
        best_res = None
        for j in range(n_ini):
            if save_path is not None:  # contain base path
                fit_params["save_path"] = save_path.split(".json")[0] + "_{}.json".format(j)
            print("Initialization number: ", j)
            vertices = random_poly(bounds, self.sim_p.poly.nedges, in_constraints=in_constraints)
            self.update_sim(vertices, update_adjoint=False, update_state=False)
            res, log = self.fit(**fit_params)
            if res[-1] < best_cost:
                best_cost = res[-1]
                best_res = res

        print("Random initialization done, best cost: ", best_cost)
        print("Starting multiscale optimization")
        fit_params["schedule"]["max_freqs"], fit_params["schedule"]["min_freqs"] = max_freqs_copy, min_freqs_copy
        fit_params["step"] = old_steps[0]
        fit_params["rotation_step"]= old_steps[1]
        fit_params["translation_step"] = old_steps[2]
        fit_params["niter"] = n_steps_final
        if best_res is not None:
            self.update_sim(best_res, update_adjoint=False, update_state=False)
        else:
            print("No valid initialization found, stopping")
            return None

        return self.fit(**fit_params)  # return the final optimization result


def _step_update_params(params, step, grad):
    return params - step * grad


def _step_update_translation(verts, step, grad):
    translated_vert = verts - step * np.reshape(grad, [1, 2])
    return translated_vert


def _step_update_rotation(verts, step, grad):
    barycenter = np.mean(verts, axis=0)[None, :]
    rotated_vert = rotate(points=verts, angle=mod2pi(-step * grad), center=barycenter)
    return rotated_vert

