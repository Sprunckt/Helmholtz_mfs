import numpy as np
from utils import c
from numba import jit, float64, int64, prange


@jit(float64[:](int64, float64[:], float64[:], float64[:], float64[:]), parallel=True, nopython=True)
def compute_time_sol_numba(N, dim, source_pos, compute_pos, t):
    # compute the partial sum of the eigenmode decomposition of the solution up to order N
    V, nt, d = np.prod(dim), len(t), len(dim)
    res = np.zeros(nt, dtype=np.float64)
    if d == 3:
        for it in prange(nt):
            for i in range(-N, N+1):
                for j in range(-N, N + 1):
                    for k in range(-N, N + 1):
                        xi, xj, xk = np.pi*i/dim[0], np.pi*j/dim[1], np.pi*k/dim[2]
                        norm = np.sqrt(xi**2 + xj**2 + xk**2)
                        res[it] += (c*np.sin(norm*t[it]*c) *
                                    np.cos(xi*source_pos[0]) * np.cos(xj*source_pos[1]) * np.cos(xk*source_pos[2]) *
                                    np.cos(xi*compute_pos[0]) * np.cos(xj*compute_pos[1]) * np.cos(xk*compute_pos[2])
                                    / (norm+1e-32)) / V / 2
            res[it] += c ** 2 * t[it] / V / 2
    elif d == 2:
        for it in prange(nt):
            for i in range(-N, N+1):
                for j in range(-N, N + 1):
                    xi, xj = np.pi*i/dim[0], np.pi*j/dim[1]
                    norm = np.sqrt(xi**2 + xj**2)
                    res[it] += (c*np.sin(norm*t[it]*c) *
                                np.cos(xi*source_pos[0]) * np.cos(xj*source_pos[1]) *
                                np.cos(xi*compute_pos[0]) * np.cos(xj*compute_pos[1])
                                / (norm+1e-32)) / V / 2
            res[it] += c ** 2 * t[it] / V / 2

    return res


@jit(float64[:](int64, float64[:], float64[:], float64[:], float64[:]), parallel=True, nopython=True)
def compute_freq_sol_numba(N, dim, source_pos, compute_pos, f):
    # compute the partial sum of the eigenmode decomposition of the solution up to order N
    V, nf, d = np.prod(dim), len(f), len(dim)
    res = np.zeros(nf, dtype=np.float64)
    if d == 3:
        for it in prange(nf):
            for i in range(-N, N+1):
                for j in range(-N, N + 1):
                    for k in range(-N, N + 1):
                        xi, xj, xk = np.pi*i/dim[0], np.pi*j/dim[1], np.pi*k/dim[2]
                        norm = np.sqrt(xi**2 + xj**2 + xk**2)
                        res[it] += (np.cos(xi*source_pos[0]) * np.cos(xj*source_pos[1]) * np.cos(xk*source_pos[2]) *
                                    np.cos(xi*compute_pos[0]) * np.cos(xj*compute_pos[1]) * np.cos(xk*compute_pos[2])
                                    / (norm**2 - f[it]**2)) / V / 2
    elif d == 2:
        for it in prange(nf):
            for i in range(-N, N+1):
                for j in range(-N, N + 1):
                    xi, xj = np.pi*i/dim[0], np.pi*j/dim[1]
                    norm = np.sqrt(xi**2 + xj**2)
                    res[it] += (np.cos(xi*source_pos[0]) * np.cos(xj*source_pos[1]) *
                                np.cos(xi*compute_pos[0]) * np.cos(xj*compute_pos[1])
                                / (norm**2 - f[it]**2)) / V / 2

    return res
