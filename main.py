from utils import plot_complex, translate_boundary, Polygon, c, plot_boundary
import numpy as np
from mfs import compute_sol, solve


boundary_cond = "neumann"

# boundary points
npoints = 200
lx, ly = 5., 3.
f = 100

vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]])

poly = Polygon(vertices)
bound, norm_vec = poly.sample(npoints, "uniform")
dx = 0.1
boundv = translate_boundary(bound, norm_vec, dx)

beta = np.ones(len(bound), dtype=complex)*(6-6j)
pos_src = np.array([[2.5, 2.5]])

# plot boundary
plot = False
if plot:
    plot_boundary(bound, norm_vec, boundv, pos_src)

npo = 100
xx, yy = np.meshgrid(np.linspace(0, lx, npo), np.linspace(0., ly, npo))
points = np.stack([xx.ravel(), yy.ravel()], axis=1)

k = 2*np.pi*f/c
alpha = solve(beta, norm_vec, bound, boundv, pos_src, k, source_type='dirac', cond=boundary_cond)

# plot the solution
sol = compute_sol(alpha, points, boundv, k, pos_src)
out_points = poly.is_out(points, poly)

sol[out_points] = np.nan + 1j*np.nan
plot_complex(xx, yy, sol, extent=[0, lx, 0, ly], vmin=-2, vmax=2)
