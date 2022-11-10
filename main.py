from utils import plot_complex, sample_rectangle, sample_polygon, out_polygon
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from mfs import compute_sol, solve, c

boundary_cond = "neumann"
# boundary points
npoints = 300
lx, ly = 5., 3.

#vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [lx/2, ly/2], [0, ly]])
vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]])

poly = Polygon(vertices)
bound, norm_vec = sample_rectangle(npoints, lx, ly)
#bound, norm_vec = sample_polygon(npoints, np.array([[0, 0], [lx, 0], [lx, ly], [lx/2, ly], [lx/2, ly/2], [0, ly/2]]))
#bound, norm_vec = sample_polygon(npoints, vertices)
dx = 0.2

# virtual boundary points
boundv, _ = sample_rectangle(npoints, lx + dx, ly + dx, 0.02) - np.array([dx/2, dx/2])
#boundv = translate_boundary(bound, norm_vec, 0.2)
#boundv, _ = sample_polygon(npoints, vertices=np.array([[-dx, -dx], [lx+dx, -dx], [lx+dx, ly+dx], [lx/2-dx, ly+dx],
#                                                [lx/2-dx, ly/2+dx], [-dx, ly/2+dx]]))
#boundv, _ = sample_polygon(npoints, vertices=np.array([[-dx, -dx], [lx + dx, -dx], [lx + dx, ly + 2*dx],
#                                                      [lx / 2, ly/2 + dx], [- dx, ly + 2*dx]]))

beta = np.ones(len(bound), dtype=complex)*(6-6j)
#pos_src = np.array([[2.5, 2.5]])
pos_src = np.array([[2.5, 2.5]])

f = 100

#boundv, ampl = greedy_mfs_2d(poly, bound, norm_vec, pos_src, neumann_operator, -3, lx + 3, -3, ly + 3, 150, k=2*np.pi*f,
#                             niter=npoints, tol=1e-3, nkeep=10, min_dist=0.2)

# plot boundary
plot = False
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
out_points = out_polygon(points, poly)

sol[out_points] = np.nan + 1j*np.nan
plot_complex(xx, yy, sol, extent=[0, lx, 0, ly], vmin=-2, vmax=2)
