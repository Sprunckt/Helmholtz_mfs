from simulation import Simulation
import numpy as np
import time
import json
from utils import rotate, mod2pi


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
        
    def line_search(self, step, step_decrease, neval_search, grad, bounds=None, convex_proj=False,
                    early_stopping=False, reset=False, in_constraints=None, verbose=False):
        if bounds is None:
            bounds = (-10., 10.)

        curr_params = self.get_params()
        best_params = curr_params
        best_cost = np.inf
        start_cost = self.cost_fun()
        best_weights = self.sim_p.mfs.room_bound.copy(), self.sim_p.vpos.copy(), self.sim_p.ampl.copy()
        new_step, curr_step, best_step = step, step, step
        init_decrease = False  # True if costfun decreases at first step 
        print("starting cost: ", start_cost) 
        for i in range(neval_search):
            curr_step = new_step
            new_params = curr_params - curr_step*grad
            if verbose:
                print("step: ", curr_step)
            success = self.update_sim(new_params, update_state=True, update_adjoint=False, convex_proj=convex_proj,
                                      bounds=bounds, check_in=in_constraints, allow_reduce=False)
            new_step = curr_step * step_decrease

            if not success:
                print("Polygon update failed, skipping")
                continue
            new_cost = self.cost_fun()
            if verbose:
                print("cost: ", new_cost)

            if new_cost < best_cost:
                best_cost = new_cost
                best_weights = self.sim_p.mfs.room_bound.copy(), self.sim_p.vpos.copy(), self.sim_p.ampl.copy()
                best_params, best_step = new_params.copy(), curr_step
                if best_cost < start_cost:
                    if i == 0:
                        init_decrease = True
                    if early_stopping:
                        if verbose:
                            print("costfunction decreased, early stopping")
                        break

        self.update_sim(best_params, update_state=False, update_adjoint=False, convex_proj=convex_proj)
        self.sim_p.load(*best_weights)
        self._update_adjoint()
        if reset:
            best_step = step
        if best_cost == np.inf:
            print("No valid step found to update polygon, failure")
        return best_cost, best_step, init_decrease

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

    def _update_active_indices(self, schedule, it):
        updated = False
        if it == schedule["update_its"][schedule["update_ind"]]:  # update min, max frequencies, do a rotation fit
            self.max_freq = schedule["max_freqs"][schedule["update_ind"]]
            self.min_freq = schedule["min_freqs"][schedule["update_ind"]]
            print("updating min, max frequencies to: ", (self.min_freq, self.max_freq), "at iteration: ", it)
                
            schedule["update_ind"] += 1
            self._possible_ind = self.sim_p.get_ind(self.min_freq, self.max_freq)
            self._active_ind = self._possible_ind.copy()
            self._nactive = len(self._active_ind)
            updated = True

        if schedule["freq_batch_size"] > 0:
            self._active_ind = self._generator.choice(self._possible_ind, schedule["freq_batch_size"], replace=False)
            self._nactive = schedule["freq_batch_size"]
            updated = True
        if schedule["mic_batch_size"] > 0:
            self._active_mic_ind = self._generator.choice(np.arange(self.nmics), schedule["mic_batch_size"],
                                                          replace=False)
            self._nactive_mic = schedule["mic_batch_size"]
            updated = True

        return updated

    def rotation_fit(self, rotation_step, niter, jtol=1e-5, gtol=1e-5, step_decrease=0.85, nlinesearch=20,
                     step_increase=1.1, increase_period=3, decrease_period=3, in_constraints=None):
        """Do a local gradient descent on the rotation of the polygon, using the barycenter of the polygon as rotation
        center. ! Only works with vertices parametrization (due to updating the simulation with vertex coordinates) !"""
        tstart = time.time()
        curr_verts = self.sim_p.poly.vertices.copy()
        barycenter = np.mean(curr_verts, axis=0)[None, :]
        decrease_ind, increase_ind = 0, 0
        it = 0
        start_cost, grad = np.inf, np.inf
        print("Starting rotation descent")
        while it < niter and np.abs(grad) > gtol and start_cost > jtol:
            start_cost = self.cost_fun()

            grad = self.partial_derivative_rotation(barycenter)
            print("it {}, starting cost: {}, grad: {}".format(it, start_cost, grad))
            print("current polygon: \n", curr_verts)
            init_decrease = False
            for j in range(nlinesearch):
                rotated_vert = rotate(points=curr_verts,
                                      angle=mod2pi(-rotation_step*grad), center=barycenter)
                success = self.update_sim(rotated_vert, update_state=True, update_adjoint=False, check_in=in_constraints)
                if success:
                    cost = self.cost_fun()
                    print("step: ", rotation_step, " cost: ", cost)
                    if cost < start_cost:
                        curr_verts = rotated_vert
                        self._update_adjoint()  # updating adjoint for next step
                        print("Cost decreased, continuing")
                        if j == 0:
                            init_decrease = True
                        break
                    else:
                        rotation_step *= step_decrease
                else:
                    print("Polygon update failed, decreasing step")
                    rotation_step *= step_decrease
            it += 1
            if init_decrease:
                decrease_ind = 0
                increase_ind += 1
                if increase_ind == increase_period:
                    rotation_step *= step_increase
                    increase_ind = 0
            else:
                decrease_ind += 1
                increase_ind = 0
                if decrease_ind == decrease_period:
                    rotation_step *= step_decrease
                    decrease_ind = 0
        print("Rotation descent done in {}s".format(time.time() - tstart))

        return rotation_step

    def translation_fit(self, translation_step, niter, jtol=1e-5, gtol=1e-5, step_decrease=0.85, nlinesearch=20,
                        step_increase=1.1, increase_period=3, decrease_period=3, in_constraints=None):
        """Do a local gradient descent on the translation of the polygon 
        ! Only works with vertices parametrization (due to updating the simulation with vertex coordinates) !"""
        tstart = time.time()
        curr_verts = self.sim_p.poly.vertices.copy()
        decrease_ind, increase_ind = 0, 0
        it = 0
        start_cost, grad = np.inf, np.inf
        print("Starting translation descent")

        while it < niter and np.sum(np.abs(grad)) > gtol and start_cost > jtol:
            start_cost = self.cost_fun()

            grad = self.partial_derivative_translation()
            print("it {}, starting cost: {}, grad: {}".format(it, start_cost, grad))
            print("current polygon: \n", curr_verts)
            init_decrease = False
            for j in range(nlinesearch):
                translated_vert = curr_verts - translation_step * np.reshape(grad, [1, 2])
                success = self.update_sim(translated_vert, update_state=True,
                                          update_adjoint=False, check_in=in_constraints)
                if success:
                    cost = self.cost_fun()
                    print("step: ", translation_step, " cost: ", cost)
                    if cost < start_cost:
                        curr_verts = translated_vert
                        self._update_adjoint()  # updating adjoint for next step
                        print("Cost decreased, continuing")
                        if j == 0:
                            init_decrease = True
                        break
                    else:
                        translation_step *= step_decrease
                else:
                    print("Polygon update failed, decreasing step")
                    translation_step *= step_decrease
            it += 1
            if init_decrease:
                decrease_ind = 0
                increase_ind += 1
                if increase_ind == increase_period:
                    translation_step *= step_increase
                    increase_ind = 0
            else:
                decrease_ind += 1
                increase_ind = 0
                if decrease_ind == decrease_period:
                    translation_step *= step_decrease
                    decrease_ind = 0
        print("Translation descent done in {}s".format(time.time() - tstart))

        return translation_step

    def fit(self, niter=100, jtol=1e-5, gtol=1e-6, step=1e-3, alternate=np.inf, neval_search=3, step_decrease=0.85,
            early_stopping=True, reset_step=False, decrease_period=10, increase_period=10, step_increase=1.2,
            convex_proj=False, rotation_params=None, translation_params=None,
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
                            min_freqs=[self.min_freq], update_its=[0, np.inf], step_reset=None,
                            )
        
        print("Schedule: ", schedule)
        if rotation_params is not None:
            rotation_params["in_constraints"] = in_constraints
        schedule["update_ind"] = 0
        self._update_active_indices(schedule, 0)
        self.update_sim()

        log["schedule"] = schedule
        assert len(schedule["max_freqs"]) == len(schedule["update_its"]), \
            "max_freqs and update_its should have same length"

        schedule["update_its"].append(np.inf)

        act_ind, act_ind_mic = self._get_active_indices()
        self._set_active_indices(act_ind, np.arange(self.nmics))
        log["cost"] = [self.cost_fun()]
        signals = self.get_signals()
        log["signals_re"], log["signals_im"] = [signals.real.tolist()], [signals.imag.tolist()]
        self._set_active_indices(act_ind, act_ind_mic)

        log["max_freq"], log["min_freq"]= [self.max_freq], [self.min_freq]
        log["ind_min"], log["ind_max"] = [float(np.min(self._active_ind))], [float(np.max(self._active_ind))]

        gradfun = self._get_gradfun()  # select the shape gradient based on parametrization

        it, cost, gradcost, alternate_ind, decrease_ind, increase_ind = 0, np.inf, np.inf, 0, 0, 0
        print("initial cost:", log["cost"][0])
        while it < niter and jtol < cost and gtol < gradcost:
            t1 = time.time()

            grad = np.zeros((len(self.sim_p.poly.vertices), 2))
            if it >= alternate:
                grad[alternate_ind] = gradfun(alternate_ind)
                alternate_ind = (alternate_ind + 1) % self.sim_p.poly.nedges
            else:
                for i in range(len(self.sim_p.poly.vertices)):
                    grad[i] = gradfun(i)
                    
            tgrad = time.time() - t1

            if self.verbose:
                print("it: ", it, "descent ind: ", alternate_ind)
                print("grad: ", grad)

            # reduce step size until cost function decreases
            cost, step, init_decreased = self.line_search(step, step_decrease, neval_search, grad, verbose=self.verbose,
                                                          early_stopping=early_stopping, reset=reset_step,
                                                          in_constraints=in_constraints, convex_proj=convex_proj)
            
            if init_decreased:
                decrease_ind = 0  # reset counter, cost function has decreased at first it of line search
                increase_ind += 1
                if increase_ind == increase_period:
                    step *= step_increase
                    increase_ind = 0
            else:
                decrease_ind += 1
                increase_ind = 0
                if decrease_ind == decrease_period:
                    step *= step_decrease
                    decrease_ind = 0

            gradcost = np.linalg.norm(grad)

            if translation_params is not None:
                nt_step = self.translation_fit(**translation_params)
                translation_params["translation_step"] = nt_step

            if rotation_params is not None:
                nr_step = self.rotation_fit(**rotation_params)
                rotation_params["rotation_step"] = nr_step


            act_ind, act_ind_mic = self._get_active_indices()
            self._set_active_indices(act_ind, np.arange(self.nmics))
            cost = self.cost_fun()
            log["cost"].append(cost)
            signals = self.get_signals()
            log["signals_re"].append(signals.real.tolist())
            log["signals_im"].append(signals.imag.tolist())
            self._set_active_indices(act_ind, act_ind_mic)  
            log["ind_min"].append(float(np.min(self._active_ind))) 
            log["ind_max"].append(float(np.max(self._active_ind)))

            log["grad"].append(grad.tolist())
            log["vert"].append(self.sim_p.poly.vertices.tolist())
            
            log["max_freq"].append(self.max_freq)
            log["min_freq"].append(self.min_freq)

            t2 = time.time()

            if (compute_full_cost > 0) and (((it + 1) % compute_full_cost) == 0):
                self._reset_active_indices()
                self.update_sim(update_state=True, update_adjoint=False)
                log["full_cost"].append(self.cost_fun())
                print("full cost: ", log["full_cost"])

            if self.verbose:
                print("cost k: ", cost, "grad_norm k-1: ", gradcost, "ind_desc: ", alternate_ind,
                      "tgrad: ", tgrad, "titer: ", t2-t1)
                print("current polygon: \n", self.sim_p.poly.vertices)
            it += 1

            updated = self._update_active_indices(schedule, it)
            if updated:
                self.update_sim(update_state=True, update_adjoint=True)

            if (save_path is not None) and (it % save_freq == 0):
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

    def partial_derivative_rotation(self, center):
        """Partial derivative according to a rotation centered at the given point"""
        V, V_corners = np.zeros((self.n_int, 2)), np.zeros((self.sim_p.poly.nedges, 2))
        rot_der = np.array([[0, 1], [-1, 0]])
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

