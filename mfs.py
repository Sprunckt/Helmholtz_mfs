import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from scipy.fft import irfft, rfft
from utils import plot_complex, rectangle, polygon
import pyroomacoustics as pra

c = 343


def fundamental_solution(points, sources, k):
    """Compute the fundamental solution at the points given the sources, return a matrix of shape npoints, nsources."""
    return (1j/4) * hankel1(0, k*np.linalg.norm(points[:, np.newaxis, :] - sources[np.newaxis, :, :], axis=-1))


def fundamental_solution_derivative(dist, k):
    """Compute the derivative of the fundamental solution at the points given the distances,
    return a matrix of shape npoints, nsources."""
    return -(1j/4) * k * hankel1(1, k*dist)


def compute_sol(alpha, points, sourcesv, k, source=None):
    """Compute the solution at the points given the virtual sources and the associated coefficients."""

    if source is None:
        return np.sum(np.reshape(alpha, -1)[np.newaxis, :] * fundamental_solution(points, sourcesv, k), axis=1)
    else:
        return (np.sum(np.reshape(alpha, -1)[np.newaxis, :] *
                       fundamental_solution(points, sourcesv, k), axis=1) +
                np.reshape(fundamental_solution(points, source, k), -1))


def compute_exact_sol(points, image_src, k, weights=None):
    """Compute the exact solution at the points given the source position."""
    if weights is None:
        weights = np.ones(len(image_src))
    return np.sum(weights[np.newaxis, :] * fundamental_solution(points, image_src, k), axis=1)


def solve(beta, normvec, bound, boundv, source, k, source_type='dirac', cond='dirichlet'):
    """Solve the Helmholtz equation with the given boundary conditions and return the coefficients of the solution in
    the MFS.
    args: - beta: impedance at the boundary
            - normvec: normal vector to the boundary
            - bound: boundary points
            - boundv: virtual boundary points for the MFS
            - source: position of a point source (dirac source term) if source_type=='dirac', otherwise a
            renormalization point (ie. a point where the solution value is set to 1)
            - source_type: 'dirac' or 'renormalization'. If set to 'renormalization', the solution is set to 1 at the
            renormalization point, and the last boundary point of the vector is ignored to match the number of points
            and virtual sources.
            function if it is a callable (accepts a (n,2) shaped array of points an a float k and returns a (n,) shaped
            array of values)
            - k: wavenumber
            - cond: boundary condition, 'dirichlet', 'neumann', 'robin'
    """

    # assemble the system matrix and rhs
    npoints = len(bound)
    A, b = np.zeros((npoints, npoints), dtype=complex), np.zeros((npoints, 1), dtype=complex)

    if source_type == 'dirac':
        indstop = npoints
    elif source_type == 'renormalization':
        indstop = -1
    else:
        raise ValueError('source_type must be "dirac" or "renormalization"')

    if cond == "neumann":
        diffs = bound[:indstop, np.newaxis, :] - boundv[np.newaxis, :, :]
        norms = np.linalg.norm(diffs, axis=-1)
        A[:indstop, :] = (np.sum(normvec[:indstop, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, np, 2) -> (np, np)
                          norms * fundamental_solution_derivative(norms, k))

        if source_type == 'dirac':  # inhomogenous equation with dirac source
            diffs = bound[:, np.newaxis, :] - source[np.newaxis, :, :]  # rhs computation
            norms = np.linalg.norm(diffs, axis=-1)
            b[:] = - (np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) *
                             fundamental_solution_derivative(norms, k) / norms)

    elif cond == "dirichlet":
        A[:indstop, :] = fundamental_solution(bound[:indstop], boundv, k)
        if source_type == 'dirac':
            b[:] = - fundamental_solution(bound, source, k)

    elif cond == "robin":
        diffs = bound[:indstop, np.newaxis, :] - boundv[np.newaxis, :, :]
        norms = np.linalg.norm(diffs, axis=-1)
        A[:indstop, :] = (np.sum(normvec[:indstop, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, np, 2)-> (np, np)
                          norms * fundamental_solution_derivative(norms, k) +
                          fundamental_solution(bound[:indstop], boundv, k) * beta[:indstop, np.newaxis] * 1j * k)

        if source_type == 'dirac':
            diffs = bound[:, np.newaxis, :] - source[np.newaxis, :, :]
            norms = np.linalg.norm(diffs, axis=-1)
            b[:] = -(np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) * fundamental_solution_derivative(norms, k)
                     / norms +
                     fundamental_solution(bound, source, k) * beta[:, np.newaxis] * 1j * k)

    if source_type == 'renormalization':  # homogeneous equation with anchor point
        A[-1, :] = np.ones(npoints)
        b[-1] = 1.

    return np.linalg.solve(A, b)


boundary_cond = "neumann"
# boundary points
npoints = 2000
lx, ly = 3., 3.
# bound, norm_vec = ellipse(npoints, 1, 2.)
#bound, norm_vec = rectangle(npoints, lx, ly)
#bound, norm_vec = polygon(npoints, np.array([[0, 0], [lx, 0], [lx, ly], [lx/2, ly], [lx/2, ly/2], [0, ly/2]]))
bound, norm_vec = polygon(npoints, np.array([[0, 0], [lx, 0.], [lx, ly], [lx/2, ly/2], [0, ly]]))
dx = 0.2

# virtual boundary points
#boundv, _ = rectangle(npoints, lx + dx, ly + dx, 0.02) - np.array([dx/2, dx/2])
#boundv = translate_boundary(bound, norm_vec, 0.2)
#boundv, _ = polygon(npoints, vertices=np.array([[-dx, -dx], [lx+dx, -dx], [lx+dx, ly+dx], [lx/2-dx, ly+dx],
#                                                [lx/2-dx, ly/2+dx], [-dx, ly/2+dx]]))
boundv, _ = polygon(npoints,
                    vertices=np.array([[-dx, -dx], [lx + dx, -dx], [lx + dx, ly + 2*dx], [lx / 2, ly/2 + dx],
                                       [- dx, ly + 2*dx]]))

beta = np.ones(len(bound), dtype=complex)*(6-6j)
#pos_src = np.array([[2.5, 2.5]])
pos_src = np.array([[2.6, 2.]])

f = 200
tmax = 0.02

# plot boundary
plot = True
if plot:
    plt.plot(bound[:, 0], bound[:, 1], '.-', label='boundary')
    plt.quiver(bound[:, 0], bound[:, 1], norm_vec[:, 0], norm_vec[:, 1])
    plt.plot(boundv[:, 0], boundv[:, 1], '.', label='virtual boundary')
    plt.plot(pos_src[:, 0], pos_src[:, 1], 'o', label='source')
    plt.legend()
    plt.show()

npo = 100
xx, yy = np.meshgrid(np.linspace(0, lx, npo), np.linspace(0., ly, npo))
points = np.stack([xx.ravel(), yy.ravel()], axis=1)

k = 2*np.pi*f/c
alpha = solve(beta, norm_vec, bound, boundv, pos_src, k, source_type='dirac', cond=boundary_cond)

# plot the solution
sol = compute_sol(alpha, points, boundv, k, pos_src)

plot_complex(xx, yy, sol, extent=[0, lx, 0, ly], vmin=-2, vmax=2)
