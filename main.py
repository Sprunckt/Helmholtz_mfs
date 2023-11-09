from utils import plot_complex, Polygon, c, plot_boundary
import numpy as np
from mfs import HelmholtzMFS, SampledBoundary


boundary_cond = "neumann"

# boundary points
lx, ly = 5., 3.
f = 100
k = 2 * np.pi * f / c

vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]])

poly = Polygon(vertices)
dx = 0.2
nbound, nboundv = 200, 100

beta = 6-6j
pos_src = np.array([[2.5, 2.5]])


npo = 100
xx, yy = np.meshgrid(np.linspace(0, lx, npo), np.linspace(0., ly, npo))
points = np.stack([xx.ravel(), yy.ravel()], axis=1)

room_bound = SampledBoundary(poly, 100, distribution="uniform", min_dist=0.01, sample_type='total')
virtual_bound = SampledBoundary(poly, 100, dx=dx, distribution="uniform", min_dist=0.00, sample_type='total')

smfs = HelmholtzMFS(room_bound=room_bound, virtual_bound=virtual_bound, pos_src=pos_src, poly=poly,
                    cond=boundary_cond, beta=[beta for _ in range(len(vertices))], add_fun=5)

# plot boundary and source
plot_boundary(smfs.room_bound.bound, smfs.room_bound.normvec, smfs.virtual_bound.bound, pos_src)

# solve the system
smfs.mfs_fit(k=k, solver='lstsq')

# plot the solution
sol = smfs.evaluate_sol(points, k=k)

plot_complex(xx, yy, sol, extent=[0, lx, 0, ly])
smfs.plot_bound_values(1000, k=k)
