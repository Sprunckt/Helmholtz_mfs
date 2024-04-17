from simulation import Simulation
import numpy as np
import time
import json


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
        
        self._active_mic_ind = np.arange(self.nmics)
        self._nactive_mic = self.nmics
        
        self.nfreqs = nfreq
        self.nedges = self.sim_p.poly.nedges
        self._generator = np.random.default_rng()

    def get_params(self):
        return self.sim_p.poly.get_params(self.parametrization)
        
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

    def _update_active_indices(self, schedule, it):
        updated = False
        if it == schedule["update_its"][schedule["update_ind"]]:
            self.max_freq = schedule["max_freqs"][schedule["update_ind"]]
            print("updating max frequency to: ", self.max_freq, "at iteration: ", it)
            schedule["update_ind"] += 1
            self._possible_ind = self.sim_p.get_ind(0., self.max_freq)
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

    def fit(self, niter=100, jtol=1e-5, gtol=1e-6, step=1e-3, alternate=np.inf, neval_search=3, step_decrease=0.85,
            early_stopping=True, reset_step=False, decrease_period=10, convex_proj=False,
            schedule=None, save_path=None, save_freq=20, compute_full_cost=0):
        """Set alternate to inf to compute full grad at each iteration, 0 to always alternate the vertex number.
        schedule is a dictionary with keys: mic_batch_size, freq_batch_size, max_freqs, update_its. max_freqs and
        update_its are lists of the same length, indicating the maximum frequencies and the iteration at which to update
        the maximume frequency to the next value."""
        log = dict(grad=[], vert=[self.sim_p.poly.vertices.tolist()], signals=[])

        if compute_full_cost > 0:
            self._reset_active_indices()
            self.update_sim(update_state=True, update_adjoint=False)
            log["full_cost"] = [self.cost_fun()]

        if schedule is None:  # default schedule : no batch, start at maximum frequency
            schedule = dict(mic_batch_size=0, freq_batch_size=0, max_freqs=[self.max_freq], update_its=[0, niter+1])
        schedule["update_ind"] = 0
        self._update_active_indices(schedule, 0)
        self.update_sim()

        log["schedule"] = schedule

        log["cost"] = [self.cost_fun()]
        log["max_freq"] = []
        gradfun = self._get_gradfun()  # select the shape gradient based on parametrization

        if self.parametrization == "half-planes":  # the origin is set at the source position
            in_constraints = self.mic_pos
        else:  # in that case also check the source position
            in_constraints = np.concatenate([self.mic_pos, self.source_pos], axis=0)

        it, cost, gradcost, rotation_ind, decrease_ind = 0, np.inf, np.inf, 0, 0
        print("initial cost:", log["cost"][0])
        while it < niter and jtol < cost and gtol < gradcost:
            t1 = time.time()

            grad = np.zeros((len(self.sim_p.poly.vertices), 2))
            if it >= alternate:
                grad[rotation_ind] = gradfun(rotation_ind)
                rotation_ind = (rotation_ind + 1) % self.sim_p.poly.nedges 
            else:
                for i in range(len(self.sim_p.poly.vertices)):
                    grad[i] = gradfun(i)
                    
            tgrad = time.time() - t1

            if self.verbose:
                print("it: ", it, "descent ind: ", rotation_ind)
                print("grad: ", grad)

            # reduce step size until cost function decreases
            cost, step, init_decreased = self.line_search(step, step_decrease, neval_search, grad, verbose=self.verbose,
                                                          early_stopping=early_stopping, reset=reset_step,
                                                          in_constraints=in_constraints, convex_proj=convex_proj)

            if init_decreased:
                decrease_ind = 0  # reset counter, cost function has decreased at first it of line search
            else:
                decrease_ind += 1

                if decrease_ind == decrease_period:
                    step *= step_decrease
            
            gradcost = np.linalg.norm(grad)

            log["cost"].append(cost)
            log["grad"].append(grad.tolist())
            log["vert"].append(self.sim_p.poly.vertices.tolist())
            log["signals"].append(self.cost_fun_full().tolist())
            log["max_freq"].append(self.max_freq)
            t2 = time.time()
            if self.verbose:
                print("cost k: ", cost, "grad_norm k-1: ", gradcost, "ind_desc: ", rotation_ind,
                      "tgrad: ", tgrad, "titer: ", t2-t1)
                print("current polygon: \n", self.sim_p.poly.vertices)

            if (compute_full_cost > 0) and (((it + 1) % compute_full_cost) == 0):
                self._reset_active_indices()
                self.update_sim(update_state=True, update_adjoint=False)
                log["full_cost"].append(self.cost_fun())
                print("full cost: ", log["full_cost"])
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

        return part_der # no normalization as it is done in evaluate_directional_derivative

    def cost_fun(self):
        return 0.5*np.sum(np.real(self.diffs * self.diffs.conj())) / self._nactive_mic / self._nactive

    def cost_fun_full(self):
        return 0.5*np.real(self.diffs*self.diffs.conj())

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

