

from shape_optim import ShapeOptimizer
import numpy as np
from simulation import Simulation
import time
import matplotlib.pyplot as plt
import json
nfreq = 20 
freqmax = 256

dx = 0.3
nmic = 16
rmic = 0.5
nedges = 4
nproc = 4  # parallel processes

np.random.seed(1)
mic_pos = np.stack([np.cos(np.linspace(0, 2*np.pi, nmic, endpoint=False)),
                    np.sin(np.linspace(0, 2*np.pi, nmic, endpoint=False))], axis=1)*rmic


vert = np.array([[2.4, 2.1], [-2.3, 0.4], [.7, -2.5], [3.1, -1.804]])
src_pos = np.array([[0., 0.]])

beta = .1
print("Exact vertices: ", vert)

# generating observations
sim = Simulation(vert, fs=freqmax, nfreq=nfreq, pos_src=src_pos, mic_pos=mic_pos, nworkers=nproc, dx=dx,
                 min_dist=0.005, nsrc_schedule=dict(method='per_wavelength', nsrc=50),
                 error_thresh=1e-5, beta=beta, add_src=15, simulate=True)

obs = np.zeros([nmic, sim.nfreqs], dtype=complex)
obs[:, 1:] = sim.compute(mic_pos)

tstart = time.time()

placeholder = np.array([[ 1.83825764,  3.21168541], [-2.90519902 , 0.38595159],
                        [1.06803508, -2.51761209], [2.71397514, -0.41132825]])
opt = ShapeOptimizer(placeholder, fs=freqmax, nfreq=nfreq, verbose=True, nworkers=nproc,
                     source_pos=src_pos.reshape([-1, 2]), mic_pos=mic_pos, measurements=obs,
                     dx=dx, n_int=1000, min_edges=nedges,
                     sim_kwargs={'beta': beta, 'add_src': 10, 'nsrc_schedule': dict(method='per_wavelength', nsrc=30)})

max_freqs = [100, 256]
min_iter_freq = [5, 20]
target_iter_freq = [5, np.inf]

min_freq = 1e-6
schedule = dict(max_freqs=max_freqs, 
                min_freqs=[min_freq for _ in max_freqs], min_iter_freq=min_iter_freq,
                target_iter_freq=target_iter_freq)
save_path = "test.json"
fit_params = dict(step=3., neval_search=5, save_path=save_path, patience=10, jtol=1e-6, min_iter=10,
        step_decrease=0.8, step_increase=1.2, increase_period=6, decrease_period=2, 
        save_freq=10, schedule=schedule, tol=0.005, penalization=10., stop_check="vertices")
res, log, _ = opt.random_multiscale_fit(bounds=5., n_steps_final=30, n_steps_ini=20, max_freq_ini=100,
                                        min_freq_ini=min_freq, fit_params=fit_params, patience_final=40, n_ini=1, save_path=save_path, n_edges=nedges)

log["exact_poly"] = vert
log["mic_pos"], log["src_pos"] = mic_pos, src_pos

f, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[1].plot(log["exact_poly"][:, 0], log["exact_poly"][:, 1], 'k-', label="exact")
ax[1].plot([log["exact_poly"][-1, 0], log["exact_poly"][0, 0]], [log["exact_poly"][-1, 1], log["exact_poly"][0, 1]], 'k-')
fd = open(save_path.split(".json")[0] + "_0.json", "r")
dump = json.load(fd)
fd.close()
ini_vert = np.array(dump["vert"][0])
ini_cost = np.array(dump["cost"])

ax[1].plot(ini_vert[:, 0], ini_vert[:, 1], 'r-', label="initial poly")
ax[1].plot([ini_vert[-1, 0], ini_vert[0, 0]], [ini_vert[-1, 1], ini_vert[0, 1]], 'r-')
ax[1].plot(log["mic_pos"][:, 0], log["mic_pos"][:, 1], 'r+', label="mics")
ax[1].plot(res[:, 0], res[:, 1], 'b-', label="reconstructed")
ax[1].plot([res[-1, 0], res[0, 0]], [res[-1, 1], res[0, 1]], 'b-')

full_cost = np.concatenate([ini_cost, log["cost"]])
ax[0].plot(full_cost, 'b-')
ax[0].set_yscale('log')
ax[0].set_xlabel("iteration"), ax[0].set_ylabel("cost")
ax[1].set_xlabel("x (m)"), ax[1].set_ylabel("y (m)")
ax[1].legend()
ax[1].axis('equal')
plt.show()
print("total time: ", time.time() - tstart)
print("final poly: ", res)


