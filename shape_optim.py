from simulation import Simulation
import numpy as np
import time
from utils import random_poly, dict_to_json


def cost_check_stop(log, patience, tol):
    if len(log["cost"]) > patience:
        if np.abs((log["cost"][-1] - log["cost"][-patience]) / log["cost"][-patience]) < tol:
            return True
    return False


def vert_check_stop(log, patience, tol):
    if len(log["vert"]) > patience:
        if np.max(np.linalg.norm(np.array(log["vert"][-1]) - np.array(log["vert"][-patience]), axis=1)) < tol:
            return True
    return False


class ShapeOptimizer:
    def __init__(self, init_vertices: np.ndarray, fs: float, nfreq: int, source_pos: np.ndarray, mic_pos: np.ndarray,
                 measurements: np.ndarray, dx: float, nworkers=1, n_int=1000,
                 sim_kwargs=None, src_ampl=None, verbose=False, min_length=0.25, min_edges=None, max_angle=3.):


        self.p_grads, self.q_grads, self.p_values, self.q_values = None, None, None, None
        self.corner_values = None

        self.min_length, self.max_angle = min_length, max_angle
        origin = np.mean(init_vertices, axis=0)
        self.sim = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, origin=origin, nworkers=nworkers,
                              observations=measurements, pos_src=source_pos, src_ampl=src_ampl, mic_pos=mic_pos,
                              simulate=True, **sim_kwargs)
        self.n_int = n_int  # number of quadrature points on the boundary
        self.quad_points, self.quad_ids, self.quad_nodes = None, None, None
        self.quad_intervals = np.zeros([self.sim.poly.nedges, 2], dtype=int)
        self._update_quadrature_points()

        # active frequency indices used in the cost function/gradient computations
        self._active_ind = np.arange(1, self.sim.nfreqs)
        self._possible_ind = np.arange(1, self.sim.nfreqs)
        self._nactive = self.sim.nfreqs - 1
        self.max_freq = self.sim.freqs[-1]
        self.min_freq = 0.
        self.min_iter_freq, self.target_iter_freq = 0, 0

        self._update_state(simulate=False)  # update corner values
        self.nmics = len(mic_pos)
        self.diffs = None  # difference between observations and simulated values, only active freqs and mics
        self.verbose = verbose
        
        if min_edges is None:
            self.nedges = self.sim.poly.nedges
        else:
            self.nedges = min_edges
        self._generator = np.random.default_rng()
        self._update_adjoint()

    def get_params(self):
        return self.sim.poly.vertices.copy()

    def _penalize_grad(self, in_constraints, eps):
        """Return the gradient of cost function penalization and the indices of the affected microphones."""
        edge_dist = self.sim_p.poly.get_edge_distances(in_constraints)  # shape (nconstr, nedges)
        penal_ind = edge_dist >= self.sim_p.poly.distances[None, :]  # same shape
        ind_out = np.where(penal_ind)
        nout = len(ind_out[0])
        if nout > 0:
            grad_penal = np.zeros([len(in_constraints), self.sim_p.poly.nedges, 2])
            scal_prod = np.sum((self.sim_p.poly.projs[ind_out[1]] - self.sim_p.poly.origin[None, :]) *
                               (in_constraints[ind_out[0]] - self.sim_p.poly.projs[ind_out[1]]), axis=1)
            vec = self.sim_p.poly.origin[None, :] - 2*self.sim_p.poly.projs[ind_out[1]] + in_constraints[ind_out[0]]

            t = (np.sum((self.sim_p.poly.origin - self.sim_p.poly.vertices) * self.sim_p.poly.edges,
                        axis=1) / self.sim_p.poly.lengths**2)[:, None]
            t2 = (-np.sum((self.sim_p.poly.origin - np.roll(self.sim_p.poly.vertices, -1, axis=0)) *
                          self.sim_p.poly.edges, axis=1)/self.sim_p.poly.lengths**2)[:, None]
            grad_t = (2*self.sim_p.poly.edges*t -
                      (self.sim_p.poly.origin[None, :] - 2*self.sim_p.poly.vertices +
                       np.roll(self.sim_p.poly.vertices, -1, axis=0))) / self.sim_p.poly.lengths[:, None]**2
            grad_t2 = (-2*self.sim_p.poly.edges*t2 -
                       (self.sim_p.poly.origin[None, :] - 2*np.roll(self.sim_p.poly.vertices, -1, axis=0) +
                           self.sim_p.poly.vertices)) / self.sim_p.poly.lengths[:, None]**2
            grad_penal[ind_out[0], ind_out[1]] = (((1-t)[ind_out[1]]*vec + grad_t[ind_out[1]] *
                                                  np.sum(self.sim_p.poly.edges[ind_out[1]]*vec, axis=1)[:, None]) *
                                                  scal_prod[:, None])
            grad_penal[ind_out[0],
            (ind_out[1] + 1) % self.sim_p.poly.nedges] += (((1-t2)[ind_out[1]]*vec - grad_t2[ind_out[1]] *
                                                            np.sum(self.sim_p.poly.edges[ind_out[1]]*vec,
                                                                   axis=1)[:, None]) * scal_prod[:, None])
            grad_penal = np.sum(grad_penal, axis=0) / len(in_constraints)
        else:
            grad_penal = np.zeros([self.sim_p.poly.nedges, 2])

        return grad_penal / eps

    def _penalize(self, in_constraints, eps):
        return np.sum(np.maximum(np.sum((in_constraints[:, None, :] - self.sim_p.poly.projs[None, :, :]) *
                      (self.sim_p.poly.projs[None, :, :] - self.sim_p.poly.origin[None, None, :]), axis=2), 
                      0)**2) / eps / 2. / len(in_constraints)

    def line_search(self, step, step_decrease, neval_search, update_fun, grad_fun, bounds=(-15., 15.),
                    convex_proj=False, in_constraints=None, verbose=False, penalization=np.inf):
        tstart = time.time()
        curr_params = self.get_params()

        start_cost = self.cost_fun()
        curr_step = step
        init_decrease = False  # True if costfun decreases at first step
        grad = grad_fun()
        if penalization != np.inf:   # warning, penalization does not work with varying mic ind yet
            grad_pen = self._penalize_grad(in_constraints, penalization)
            hard_in_constraints = None
            grad += grad_pen
            start_cost += self._penalize(in_constraints, penalization)
        else:
            hard_in_constraints = in_constraints
        print("starting cost: ", start_cost) 
        for i in range(neval_search):
            new_params = update_fun(curr_params, curr_step, grad)  # apply update
            if verbose:
                print("step: ", curr_step, flush=True)
            success = self.update_sim(new_params, update_state=True, update_adjoint=False, convex_proj=convex_proj,
                                      bounds=bounds, check_in=hard_in_constraints, allow_reduce=self.nedges)
            curr_step = curr_step * step_decrease

            if not success:
                new_cost = np.inf
                print("Polygon update failed, skipping", flush=True)
                continue

            new_cost = self.cost_fun()
            if penalization != np.inf:
                pen = self._penalize(in_constraints, penalization)
                if verbose and pen > 0:
                    print("applying penalization : " + str(pen), flush=True)
                new_cost += pen
            if verbose:
                print("cost: ", new_cost, flush=True)

            if new_cost < start_cost:
                if i == 0:
                    init_decrease = True
                if verbose:
                    print("costfunction decreased, early stopping", flush=True)
                break

        if new_cost == np.inf:
            print("No valid step found to update polygon, failure", flush=True)
        
        if verbose:
            print("line search time: ", time.time()-tstart, flush=True)
        
        return new_cost, grad, init_decrease

    def _update_quadrature_points(self):
        self.quad_points, _, self.quad_ids, self.quad_nodes = self.sim.poly.sample(self.n_int, return_nodes=True,
                                                                                   min_dist=0.)
        interv = np.where(np.diff(self.quad_ids) != 0)[0]
        self.quad_intervals = np.zeros([self.sim.poly.nedges, 2], dtype=int)
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

    def _increase_frequency_range(self, schedule):  # increase frequency range
        if schedule["update_ind"] < len(schedule["max_freqs"]):
            # update min, max frequencies
            self.max_freq = schedule["max_freqs"][schedule["update_ind"]]
            self.min_freq = schedule["min_freqs"][schedule["update_ind"]]
            self.min_iter_freq = schedule["min_iter_freq"][schedule["update_ind"]]
            self.target_iter_freq = schedule["target_iter_freq"][schedule["update_ind"]]
            print("updating min, max frequencies to: ", (self.min_freq, self.max_freq))
                
            schedule["update_ind"] += 1
            self._possible_ind = self.sim_p.get_ind(self.min_freq, self.max_freq)
            self._active_ind = self._possible_ind.copy()
            if schedule["freq_batch_size"] > 0:
                self._nactive = np.minimum(schedule["freq_batch_size"], len(self._active_ind))
            else: 
                self._nactive = len(self._active_ind)
            return True
        else: 
            return False 
        
    def _batch(self, schedule):
        update_state = False
        if schedule["freq_batch_size"] > 0:
            self._active_ind = self._generator.choice(self._possible_ind, self._nactive, replace=False)
            update_state = True
        if schedule["mic_batch_size"] > 0:
            self._active_mic_ind = self._generator.choice(np.arange(self.nmics), schedule["mic_batch_size"],
                                                          replace=False)
            self._nactive_mic = schedule["mic_batch_size"]

        return update_state
  
    def fit(self, niter=100, jtol=1e-5, step=10, neval_search=3, step_decrease=0.85, min_iter=500,
            decrease_period=10, increase_period=10, step_increase=1.2, tol=0.01, stop_check="cost",
            patience=20, convex_proj=False, schedule=None, save_path=None, save_freq=20,
            old_log=None, penalization=np.inf):
        """Set alternate to inf to compute full grad at each iteration, 0 to always alternate the vertex number.
        schedule is a dictionary with keys: mic_batch_size, freq_batch_size, max_freqs, update_its. max_freqs and
        update_its are lists of the same length, indicating the maximum frequencies and the iteration at which to update
        the maximume frequency to the next value."""

        if stop_check == "cost":
            stop_check = cost_check_stop
        elif stop_check == "vertices":
            stop_check = vert_check_stop
        else:
            print("stopping criterion '{}' unknown".format(stop_check))
            exit(1)

        if old_log is None:
            log = dict(grad=[], vert=[self.sim_p.poly.vertices.tolist()], signals=[])
        else:
            log = old_log

        if self.parametrization == "half-planes":  # the origin is set at the source position
            in_constraints = self.sim.mic_pos
        else:  # in that case also check the source position
            in_constraints = np.concatenate([self.sim.mic_pos, self.sim.src_pos], axis=0)

        if schedule is None:  # default schedule : no batch, start at maximum frequency
            schedule = dict(mic_batch_size=0, freq_batch_size=0, max_freqs=[self.max_freq], min_iter_freq=[min_iter],
                            target_iter_freq=[min_iter], min_freqs=[self.min_freq], step_reset=None)
        
        print("Schedule: ", schedule)
        schedule["update_ind"] = 0
        
        self._increase_frequency_range(schedule)
        self._batch(schedule)

        if self.sim_p.poly.nedges != len(self.sim_p.mfs.beta):   # todo: remove
            self.sim_p.mfs.beta = np.array([self.sim_p.mfs.beta[0] for _ in range(self.sim_p.poly.nedges)])
            self.sim_q.mfs.beta = - self.sim_p.mfs.beta
        
        self.update_sim(update_state=True, update_adjoint=True)

        log["schedule"] = schedule
        assert len(schedule["max_freqs"]) == len(schedule["min_freqs"]), \
            "max_freqs and min_freqs should have same length"

        if old_log is None:
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
            log["initial_vert"] = self.sim_p.poly.vertices.copy()
            print("initial cost:", log["cost"][0])
        else:
            print("initial cost:", log["cost"][-1])

        it, cost, gradcost, decrease_ind, increase_ind = 0, np.inf, np.inf, 0, 0
        patience_counter, freq_it_counter = 0, 0
        while it < niter:
            t1 = time.time()
            if self.verbose:
                print("it: {}".format(it))

            # line search: reduce step size until cost function decreases
            cost, grad, init_decreased = self.line_search(step, step_decrease, neval_search,
                                                          update_fun=_step_update_params, penalization=penalization,
                                                          grad_fun=self.get_gradient, verbose=self.verbose,
                                                          in_constraints=in_constraints, convex_proj=convex_proj,)

            if cost == np.inf:
                print("Optimization failed, stopping", flush=True)
                break

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

            act_ind, act_ind_mic = self._get_active_indices()
            self._set_active_indices(act_ind, np.arange(self.nmics))
            cost = self.cost_fun()
            log["cost"].append(cost)
            
            signals = self.get_signals()
            log["signals_re"].append(signals.real.tolist()), log["signals_im"].append(signals.imag.tolist())
            self._set_active_indices(act_ind, act_ind_mic)  
            log["ind_min"].append(int(np.min(self._active_ind))), log["ind_max"].append(int(np.max(self._active_ind)))

            log["grad"].append(grad.tolist()), log["vert"].append(self.sim_p.poly.vertices.tolist())
            
            log["max_freq"].append(self.max_freq), log["min_freq"].append(self.min_freq)

            update_state, stopping = False, False   # update state or not
            freq_it_counter += 1
            if patience_counter >= patience:
                stopping = stop_check(log, patience, tol)
            else:
                patience_counter += 1

            # check for costfun value/if patience arg met and if min number of it for freq range was reached
            stopping = (stopping or (cost < jtol) or (freq_it_counter >= self.target_iter_freq)) and (freq_it_counter >= self.min_iter_freq)  
            if stopping:  # check if frequency range can be extended
                updated = self._increase_frequency_range(schedule)  # increase frequency range
                if updated:  # increasing freq range, continuing
                    decrease_ind, increase_ind, patience_counter, patience_counter, freq_it_counter = 0, 0, 0, 0, 0
                    update_state = True
                    log["frequency_change"].append(it)
                else:
                    if it >= min_iter:
                        break
                    
            update_state = (self._batch(schedule) or update_state)
        
            self.update_sim(update_adjoint=True, update_state=update_state)
            if schedule["mic_batch_size"] > 0:
                self._update_obs()
                
            t2 = time.time()

            if self.verbose:
                print("iteration {} done, freq it counter {}".format(it, freq_it_counter))
                print("cost: ", cost, "grad_norm k-1: ", gradcost, "titer: ", t2-t1)
                print("current polygon: \n", self.sim_p.poly.vertices, flush=True)
            it += 1

            if (save_path is not None) and (it % save_freq == 0):
                dict_to_json(log, save_path)

        if save_path is not None:
            dict_to_json(log, save_path)

        return self.sim.poly.vertices, log, cost

    def _update_state(self, simulate=True):
        if simulate:
            self.sim.simulate(self._active_ind, adjoint=False, verbose=False)

        self.p_values = self.sim.compute(self.quad_points, self._active_ind, adjoint=False)
        self.cornerval_p = self.sim.compute_corner_values(self._active_ind, adjoint=False)

        nvert = len(self.sim.poly.vertices)
        self.p_grads = np.zeros((self.n_int, self._nactive, 2), dtype=complex)
        for i in range(nvert):  # compute gradients on each edge
            start_ind, end_ind = self.quad_intervals[i]
            self.p_grads[start_ind+1: end_ind] = self.sim.compute_grad(self.quad_points[start_ind+1: end_ind],
                                                                       self._active_ind, adjoint=False)
            self.p_grads[start_ind] = self.cornerval_p[1][i, :, :]
            self.p_grads[end_ind] = self.cornerval_p[1][(i + 1) % self.sim.poly.nedges, :, :]

    def _update_adjoint(self):
        self.sim.simulate(self._active_ind, adjoint=True, verbose=False)
        self.q_values = self.sim.compute(self.quad_points, self._active_ind, adjoint=True)

        cornerval_q = self.sim.compute_corner_values(freq_ind=self._active_ind, adjoint=True)

        self.q_grads = np.zeros((self.n_int, self._nactive, 2), dtype=complex)
        self.corner_values = np.zeros((self.sim.poly.nedges, self._nactive))
        for i in range(len(self.sim.poly.vertices)):  # compute gradients on each edge
            start_ind, end_ind = self.quad_intervals[i]

            self.q_grads[start_ind + 1:end_ind] = self.sim.compute_grad(self.quad_points[start_ind + 1:end_ind],
                                                                        self._active_ind, adjoint=True)
            self.q_grads[start_ind] = cornerval_q[1][i, :, :]
            self.q_grads[end_ind] = cornerval_q[1][(i + 1) % self.sim.poly.nedges, :, :]

            self.corner_values[i] = np.real(cornerval_q[0][i].conj() * np.sum(self.cornerval_p[1][i, :, :] *
                                                                              self.sim.poly.normals[i - 1],
                                                                              axis=-1))  # dn p * q.conj

    def update_sim(self, new_params=None, update_state=True, update_adjoint=True,
                   bounds=(-15, 15), check_in=None, convex_proj=False, allow_reduce=3, verbose=False):
        if new_params is not None:
            success = self.sim.update_poly(new_params, convex_proj=convex_proj, bounds=bounds, check_in=check_in,
                                           min_length=self.min_length, max_angle=self.max_angle)
            
            if self.sim.poly.nedges < allow_reduce:
                success = False
            if success:
                self._update_quadrature_points()
        else:
            success = True

        if success:
            if verbose:
                print("Poly updated:", self.sim.poly.vertices, self.sim.poly.angles, flush=True)
            if update_state:
                self._update_state()
            if update_adjoint:
                self._update_adjoint()
        return success

    def partial_derivative(self, i):
        # generate a function psi_i on edges i, i-1 such that psi_i(i) = 1, psi_i(i-1) = 0 = psi_i(i+1)
        psi = [np.linspace(0, 1, len(self.quad_nodes[self.quad_ids == (i-1) % self.sim.poly.nedges])),
               np.linspace(1, 0, len(self.quad_nodes[self.quad_ids == i]))]
        ind = [(i-1) % self.sim.poly.nedges, i]
        normals = [self.sim.poly.normals[i-1], self.sim.poly.normals[i]]
        res = np.zeros(2)
        for j in range(2):
            quad_ids = np.where(self.quad_ids == ind[j])[0]
            p_val, q_val = self.p_values[quad_ids], self.q_values[quad_ids]
            p_grad, q_grad = self.p_grads[quad_ids], self.q_grads[quad_ids]
            normal = normals[j]
            vals = (np.sum(np.real(p_grad * q_grad.conj()), axis=(1, 2)) -
                    np.sum(np.real(self.sim.wave_numbers[np.newaxis, self._active_ind]**2 * p_val * q_val.conj()),
                           axis=1) +
                    2 * np.sum(np.real(self.sim.beta * np.conj(self.sim.beta)) *
                               self.sim.wave_numbers[np.newaxis, self._active_ind]**2 *
                               np.real(p_val * q_val.conj()), axis=1))*psi[j]
            res = res + normal*self.sim.poly.integrate_edge(vals, self.quad_nodes[quad_ids], ind[j])
        renormed_edge1 = self.sim.poly.edges[i] / np.linalg.norm(self.sim.poly.edges[i])
        renormed_edge2 = self.sim.poly.edges[i-1] / np.linalg.norm(self.sim.poly.edges[i-1])

        res = res + (renormed_edge1 - renormed_edge2)*np.sum(np.real(self.corner_values[i, :]))
        return res / self.nmics / self._nactive

    def cost_fun(self):
        return 0.5*np.sum(np.real(self.sim.src_ampl_adj *  # contains the diffs between observations and state values
                                  self.sim.src_ampl_adj.conj())) / self.nmics / self._nactive

    def cost_fun_full(self):
        return 0.5*np.real(self.diffs*self.diffs.conj())
    
    def get_signals(self):
        return self.sim_p.compute(self.sim.mic_pos[self._active_mic_ind], self.nprocess, freq_ind=self._active_ind)
    
    def get_gradient(self):
        grad = np.zeros((len(self.sim_p.poly.vertices), 2))

        for i in range(len(self.sim_p.poly.vertices)):
            grad[i] =  self.partial_derivative(i)
        return grad 

    def random_multiscale_fit(self, bounds, n_steps_final, n_steps_ini, max_freq_ini, min_freq_ini, 
                              fit_params, n_ini=10,  patience_final=None, n_edges=None, save_path=None,
                              br_params=None):
        """Optimize shape using randomized initialization and refining frequencies at each restart.
        - bounds: list of 2 tuples (for x and y), lower and upper bounds for the random vertices, or a float
        (maximum norm of the vertices)
        - n_steps_final: number of steps for the final (refined) optimization
        - n_steps_ini: number of steps for the initial random optimization
        - max_freq_ini, min_freq_ini: initial frequency range
        - fit_params: dictionary containing the optimization parameters (should contain a valid schedule)
        - n_ini: number of random initializations
        - save_path: path to save the optimization results
        """
        if br_params is None:
            n_keep, n_iter = [], []
        else:
            n_keep, n_iter = br_params[0], br_params[1]
        tstart = time.time()
        if n_edges is None:
            n_edges = self.sim_p.poly.nedges
        in_constraints = np.concatenate([self.sim.mic_pos, self.sim.src_pos], axis=0)
        max_freqs_copy = fit_params["schedule"]["max_freqs"].copy()
        min_freqs_copy = fit_params["schedule"]["min_freqs"].copy()
        fit_params["niter"] = n_steps_ini
        fit_params["schedule"]["max_freqs"], fit_params["schedule"]["min_freqs"] = [max_freq_ini], [min_freq_ini]
        if br_params is None:
            fit_params["save_freq"] = 5
        else:
            fit_params["save_freq"] = np.maximum(np.max(br_params[1]), n_steps_ini) + 5
            
        old_step = fit_params["step"]
        print("Starting random initialization")
        parametrization = self.parametrization
        
        costs, params, logs = np.zeros(n_ini), [], []
        curr_ini = 0
        fit_params["decrease_period"], old_period = 1, fit_params.get("decrease_period")  # speed up initialization
        fit_params["step_decrease"], old_decrease = 0.6, fit_params.get("step_decrease")
        while curr_ini < n_ini:
            self.sim_p.mfs.beta = np.array([self.sim_p.mfs.beta[0] for _ in range(n_edges)])
            self.sim_q.mfs.beta = - self.sim_p.mfs.beta

            if save_path is not None:  # contain base path
                fit_params["save_path"] = save_path.split(".json")[0] + "_{}.json".format(curr_ini)
            print("Initialization number: ", curr_ini)
            vertices = random_poly(bounds, n_edges, in_constraints=in_constraints, min_length=.5, min_dist=0.5).vertices  # old 3*np.pi/4, min°length 2
            self.parametrization = "vertices"
            success = self.update_sim(vertices, update_adjoint=False, update_state=False)
            print("polyshape" , vertices.shape)
            if not success:
                print("Initialization failed, trying again", flush=True)
                continue
            self.parametrization = parametrization 
            res, log, cost = self.fit(**fit_params)
            if cost == np.inf:  # optimization failed, try another init
                print("Initialization failed, trying again", flush=True)
                continue 

            if (fit_params["schedule"]["freq_batch_size"] > 0) or (fit_params["schedule"]["mic_batch_size"] > 0):
                self._reset_active_indices()
                if (fit_params["schedule"]["freq_batch_size"] > 0):
                    self._update_state()
                cost = self.cost_fun()
            params.append(self.get_params())
            logs.append(log)
            costs[curr_ini] = cost
            curr_ini += 1

        ini_num = np.arange(n_ini)
        for i in range(len(n_keep)):
            best_ind = np.argsort(costs)[:n_keep[i]]
            ini_num = ini_num[best_ind]
            print("best ind: {}\n best costs: {}".format(ini_num, costs[best_ind]))
            new_params, costs, new_logs = [], np.zeros(n_keep[i]), []
            fit_params["niter"] = n_iter[i]
            for j in range(n_keep[i]):
                print("Initialization number: ", ini_num[j], flush=True)
                fit_params["old_log"] = logs[best_ind[j]]
                if save_path is not None:  # contain base path
                    fit_params["save_path"] = save_path.split(".json")[0] + "_{}.json".format(ini_num[j])
                self.update_sim(params[best_ind[j]], update_adjoint=False, update_state=False)
                res, log, cost = self.fit(**fit_params)
                new_logs.append(log)
                if (fit_params["schedule"]["freq_batch_size"] > 0) or (fit_params["schedule"]["mic_batch_size"] > 0):
                    self._reset_active_indices()
                    if (fit_params["schedule"]["freq_batch_size"] > 0) and (cost != np.inf):
                        self._update_state()
                    cost = self.cost_fun()
                costs[j] = cost
                new_params.append(self.get_params())
            params = new_params
            logs = new_logs

        best_ind = np.argmin(costs)
        best_cost = costs[best_ind]
        self.sim_p.mfs.beta = np.array([self.sim_p.mfs.beta[0] for _ in range(len(params[best_ind]))])
        self.sim_q.mfs.beta = - self.sim_p.mfs.beta

        # restore decrease parameters for final fit
        fit_params["step_decrease"], fit_params["decrease_period"] = old_decrease, old_period  
        self.update_sim(params[best_ind], update_adjoint=False, update_state=False)

        tend1 = time.time()
        print("Random initialization done, best cost: {}, time: {}".format(best_cost, tend1 - tstart))
        print("Starting multiscale optimization")
        fit_params["old_log"] = None
        fit_params["schedule"]["max_freqs"], fit_params["schedule"]["min_freqs"] = max_freqs_copy, min_freqs_copy
        fit_params["step"] = old_step
        fit_params["save_freq"] = 50

        fit_params["niter"] = n_steps_final
        fit_params["save_path"] = save_path
        if patience_final is not None:
            fit_params["patience"] = patience_final

        res, log, _ = self.fit(**fit_params)
        print("total time: {}, final cost: {}".format(time.time() - tstart, log["cost"][-1]))
        return res, log, log["cost"][-1]  # return the final optimization result

def _step_update_params(params, step, grad):
    return params - step * grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import Polygon
    nfreq = 2
    freqmax = 256
    dx = 0.3
    beta = 0.1
    v = np.array([0.5, 1.])#np.array([1.231231, 0.8785678])
    v /= np.linalg.norm(v)
    lxinit, ly = 2., 2.4876
    h = 0.1
    poly_mic = Polygon(np.array([[0.2, 0.2], [lxinit - 0.2, 0.2], [lxinit - 0.2, ly - 0.2], [0.2, ly - 0.2]]))
    nmic = 32

    mic_pos = poly_mic.sample(nmic, min_dist=0.1)[0]
    # mic_pos = np.array([[0.5,0.25]])

    # plt.plot(poly_mic.vertices[:, 0], poly_mic.vertices[:, 1])
    # plt.plot(mic_pos[:, 0], mic_pos[:, 1], 'o')
    # plt.show()
    pos_src = np.array([[0.5, ly / 2]])

    ntest = 5
    jval = np.zeros((5, ntest), dtype=float)
    djval = np.zeros((4, ntest, 2), dtype=float)
    init_vert = np.array([[0., 0], [lxinit, 0], [lxinit, ly], [0., ly]])

    sim = Simulation(init_vert, freqmax, nfreq,
                         pos_src=pos_src, mic_pos=mic_pos,
                         nworkers=1, dx=dx,
                        min_dist=0.005,
                                         nsrc_schedule=dict(method='per_wavelength', nsrc=30),
                                         error_thresh=1e-5, beta=beta, add_src=10, simulate=True)

    obs = np.zeros([len(mic_pos), nfreq], dtype=np.complex128)
    obs[:, 1:] = sim.compute(mic_pos, adjoint=False)
    opt = ShapeOptimizer(init_vert, freqmax, nfreq,
                         source_pos=pos_src, mic_pos=mic_pos, measurements=obs,
                         nworkers=1, dx=dx, n_int=500,
                         sim_kwargs=dict(min_dist=0.005,
                                         nsrc_schedule=dict(method='per_wavelength', nsrc=np.array([30])),
                                         error_thresh=1e-5, beta=beta, add_src=10))
    hh = 1e-5
    for i in range(ntest):
        print("i = ", i)
        vert_mod = init_vert.copy()
        vert_mod[2] = vert_mod[2] + i*h*v
        opt.update_sim(vert_mod, update_state=True, update_adjoint=True)
        for l in range(4):
            djval[l, i] = opt.partial_derivative(l)

        jval[0, i] = opt.cost_fun()  # for tracking

        vertmod2 = vert_mod.copy()
        vertmod2[2] = vertmod2[2] - np.array([hh/2, 0.])
        opt.update_sim(vertmod2, update_state=True, update_adjoint=False)
        jval[1, i] = opt.cost_fun()
        vertmod3 = vert_mod.copy()
        vertmod3[2] = vertmod3[2] + np.array([hh/2, 0.])
        opt.update_sim(vertmod3, update_state=True, update_adjoint=False)
        jval[2, i] = opt.cost_fun()
        vertmod2 = vert_mod.copy()
        vertmod2[2] = vertmod2[2] - np.array([0., hh/2])
        opt.update_sim(vertmod2, update_state=True, update_adjoint=False)
        jval[3, i] = opt.cost_fun()
        vertmod3 = vert_mod.copy()
        vertmod3[2] = vertmod3[2] + np.array([0., hh/2])
        opt.update_sim(vertmod3, update_state=True, update_adjoint=False)
        jval[4, i] = opt.cost_fun()

    fdjvalx = (jval[2, :] - jval[1, :]) / hh
    fdjvaly = (jval[4, :] - jval[3, :]) / hh
    xrange = np.arange(0, ntest)
    plt.plot(xrange, fdjvalx, label='finite diffx')
    plt.plot(xrange, fdjvaly, label='finite diffy')
    plt.plot(xrange, djval[2, :, 0], 'k--', label='shape_derivx')
    plt.plot(xrange, djval[2, :, 1], 'g--', label='shape_derivy')
    plt.plot(xrange, jval[0,:], label='cost function')
    plt.legend()
    plt.show()
    print(djval)
