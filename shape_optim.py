from simulation import Simulation
import numpy as np
from scipy.optimize import minimize


class ShapeOptimizer:
    def __init__(self, init_vertices: np.ndarray, fs: float, nfreq: int, source_pos: np.ndarray, mic_pos: np.ndarray,
                 measurements: np.ndarray, dx: float, nprocess=1, sim_kwargs=None):
        if sim_kwargs is None:
            sim_kwargs = {"boundsep": 0.05}
        self.sim = Simulation(vertices=init_vertices, fs=fs, nfreq=nfreq, dx=dx, pos_src=source_pos, **sim_kwargs)
        self.source_pos = source_pos
        self.mic_pos = mic_pos
        self.antenna_radius = np.max(np.linalg.norm(self.mic_pos, axis=1))
        self.nprocess = nprocess
        self.observations = measurements  # shape len(freqs)

    def fit(self, niter=200, geom_tol=0.5, rmax=20., step=1e-3):
        # optimize using lbgfs-b
        minr = self.antenna_radius + geom_tol
        bounds = [(minr, rmax) for _ in range(self.sim.poly.nedges)] + \
                 [(-4*np.pi, 4*np.pi) for _ in range(self.sim.poly.nedges)]
        p0 = np.concatenate([self.sim.poly.phi, self.sim.poly.theta])
        print("optimizing...")
        opt_res = minimize(self.cost_fun, p0, method='L-BFGS-B', bounds=bounds)
        print("done")
        return opt_res

    def cost_fun(self, params):
        phi, theta = params[:self.sim.poly.nedges], params[self.sim.poly.nedges:]
        self.sim.poly.update_param(phi, theta)
        self.sim.simulate(self.nprocess)
        obs_sim, _ = self.sim.compute(self.mic_pos, self.nprocess)
        diff = obs_sim - self.observations
        return 0.5*np.sum(diff*diff.conj()).real
