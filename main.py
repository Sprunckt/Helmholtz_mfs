from utils import plot_complex, Polygon, c, plot_boundary
import numpy as np
from mfs import HelmholtzMFS


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

smfs = HelmholtzMFS(nbound=nbound, nsrc=100, pos_src=pos_src, k=k, poly=poly,
                    dx=dx, cond=boundary_cond, beta=[beta for _ in range(len(vertices))], src_nb="total", add_fun=0)

# plot boundary and source
plot_boundary(smfs.bound, smfs.normvec, smfs.boundv, pos_src)

# solve the system
smfs.mfs_fit(solver='lstsq')

# plot the solution
sol = smfs.evaluate_sol(points)

plot_complex(xx, yy, sol, extent=[0, lx, 0, ly])
smfs.plot_bound_values(1000)
