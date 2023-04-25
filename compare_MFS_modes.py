"""Compare the Method of Fundamental Solutions to the analytical solution in 2D on different rectangles."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import plot_boundary, Polygon, Ellipse
from mfs import compute_sol, solve, neumann_operator, SlidingMFS, robin_operator, dirichlet_operator

import pyroomacoustics as pra
std_off = pra.constants.get("frac_delay_length") // 2
lx, ly = 5.2, 3.3
nplot = 3000
dx = 0.2
plot_sol = False
plot_bound = False
plot_bound_cond = True
test_type = "comsol_polygon"
bound_cond = "neumann"
beta = 0.006378610727567917+0.009574047932848483j
exact_path = "data/"
kwargs = {}

if bound_cond == "neumann":
    op = neumann_operator
elif bound_cond == "robin":
    op = robin_operator
else:
    op = dirichlet_operator

if test_type in ["comsol_polygon", "distance_polygon", "sliding_polygon"]:
    vertices = np.array([[2.2, 0], [4.3, 0.], [5.7, 1.9], [3.3, 5.1], [1.7, 5.5], [-0.75, 2.75]])
    pos_src = np.array([[1, 2.5]])
else:
    vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]])
    pos_src = np.array([[2.5, 2.5]])

poly = Polygon(vertices)
print("Perimeter: {}".format(np.sum(poly.lengths)))
distribution = "uniform"


def plot_bound_values(op, coeff, sourcesv, coeffs_id, show=True, name="", plot3d=False, **kwargs):
    mfs_val, src_val, edges, plot_nodes = [], [], [t*nplot for t in range(1, poly.nedges)], []
    for j in range(poly.nedges):
        plot_nodes.append(poly.sample_edge(nplot, j, distribution="uniform"))
        norms = np.repeat(poly.normals[j][np.newaxis, :], nplot, axis=0)
        mfs_val.append(op(plot_nodes[-1], norms, coeff, sourcesv, k, **kwargs))
        src_val.append(op(plot_nodes[-1], norms, np.array([-1.]), np.reshape(pos_src, (1, -1)), k, **kwargs))

    mfs_val, src_val = np.concatenate(mfs_val), np.concatenate(src_val)
    error_real = np.mean(np.abs(np.real(mfs_val) - np.real(src_val)))
    error_imag = np.mean(np.abs(np.imag(mfs_val) - np.imag(src_val)))

    if plot3d:
        plot_nodes = np.concatenate(plot_nodes, axis=0)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_title("Real part, nsrc={}, mean error={:3f}".format(npo, error_real))
        # plot the boundary values above the polygon
        ax1.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.real(mfs_val), label="MFS")
        ax1.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.real(src_val), '--', label="Source contribution")

        ax2 = fig.add_subplot(222, projection='3d')
        ax2.set_title("Imaginary part, mean error={:3f}".format(error_imag))
        ax2.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.imag(mfs_val), label="MFS")
        ax2.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.imag(src_val), '--', label="Source contribution")

        for ax in [ax1, ax2]:             # plot the polygon on the z=0 plane
            ax.plot(poly.vertices[:, 0], poly.vertices[:, 1], np.zeros(poly.nedges), 'k')
            ax.plot([poly.vertices[0, 0], poly.vertices[-1, 0]],
                    [poly.vertices[0, 1], poly.vertices[-1, 1]], [0, 0.], 'k')
        # set equal aspect ratio for the 3d plot
        max_val = np.max(np.abs(plot_nodes), axis=0)
        min_val = np.min(np.abs(plot_nodes), axis=0)
        ax1.set_xlim([min_val[0], max_val[0]]), ax1.set_ylim([min_val[1], max_val[1]])
        ax2.set_xlim([min_val[0], max_val[0]]), ax2.set_ylim([min_val[1], max_val[1]])
        plt.legend()

    else:
        fig, ax = plt.subplots(2, 2, figsize=(20, 6))
        ax[0, 0].set_ylim([np.min(src_val.real), np.max(src_val.real)])
        ax[0, 1].set_ylim([np.min(src_val.imag), np.max(src_val.imag)])

        ax[0, 0].set_title("Real part, nsrc={}, mean error={:3f}".format(npo, error_real))
        ax[0, 1].set_title("Imaginary part, mean error={:3f}".format(error_imag))
        ax[0, 0].plot(np.real(mfs_val), label="MFS")
        ax[0, 0].plot(np.real(src_val), '--', label="Source contribution")
        ax[0, 1].plot(np.imag(mfs_val), label="MFS")
        ax[0, 1].plot(np.imag(src_val), '--', label="Source contribution")
        # add vertical bars at each edge change
        ax[0, 1].vlines(edges, np.min(src_val.real), np.max(src_val.real), colors="k", linestyles="dashed")
        ax[0, 1].vlines(edges, np.min(src_val.imag), np.max(src_val.imag), colors="k", linestyles="dashed")
        plt.legend()
        plt.tight_layout()

    # add a subplot under both plots to show the coefficients
    ax3 = fig.add_subplot(212)
    ax3.set_title("Coefficients")
    ax3.semilogy(np.abs(coeff.real), '.', label="real part")
    ax3.semilogy(np.abs(coeff.imag), '.', label="imaginary part")
    # plot a vline at each edge change (given by value changes in coeffs_id)
    ax3.vlines(np.where(np.diff(coeffs_id) != 0)[0], np.minimum(np.min(coeff.real), np.min(coeff.imag)),
               np.maximum(np.max(coeff.real), np.max(coeff.imag)), colors="k", linestyles="dashed")
    plt.legend()
    if show:
        plt.show()
    else:  # save the figure
        plt.savefig("data/fig/Neumann_edge_{}_nsrc={}.png".format(name, npo))


if test_type == "convergence":
    freqs = np.loadtxt(exact_path + "freqs.txt", delimiter=",")[:1]

    nb_bound = np.array([200, 300, 500, 800, 1200, 1500, 2000, 2500, 3000, 3500, 4000])
    conditions, l2_rel_error = np.zeros([len(freqs), len(nb_bound)]), np.zeros([len(freqs), len(nb_bound)])

    for itf, fs in enumerate(freqs):
        k = 2*np.pi*fs/343

        ntest, nmax = 50, 15000
        points = np.meshgrid(np.linspace(0.1, lx-0.1, ntest), np.linspace(0.1, ly-0.1, ntest))
        points = np.stack(points, axis=2).reshape(-1, 2)

        exact_sol = np.loadtxt(exact_path + "exact_sol_{}_{}_{}.csv".format(itf, nmax, ntest), delimiter=",")
        exact_sol = exact_sol.flatten()

        for itp, npo in enumerate(nb_bound):

            print("fitting for frequency {} Hz with {} sources".format(fs, npo))
            smfs = SlidingMFS(nbound=npo, nsrc=500, nobs=200, pos_src=pos_src, k=k, poly=poly, dx=dx,
                              cond=bound_cond, beta=beta)
            alpha, conditions[itf, itp] = smfs.gram_fit()

            if plot_bound:  # plot boundary
                plot_boundary(smfs.bound, smfs.normvec, smfs.boundv, pos_src)

            print("min and max values : ", np.min(alpha), np.max(alpha))

            print("Computing solution".format(fs, npo))
            sol = compute_sol(smfs.ampl, points, smfs.boundv, k, pos_src)
            print("Done")

            print("imaginary part norm: {}".format(np.linalg.norm(np.imag(sol))))

            l2_rel_error[itf, itp] = np.linalg.norm(sol - exact_sol) / np.linalg.norm(exact_sol)
            print("condition number: {}, L2 error: {}".format(conditions[itf, itp], l2_rel_error[itf, itp]))
            if plot_sol:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                maxval = np.max(np.abs(np.real(sol)))
                im1 = ax[1].matshow(np.real(sol).reshape(ntest, ntest),
                                    vmin=-maxval, vmax=maxval, extent=[0, lx, 0, ly], origin='lower')
                im2 = ax[0].matshow(exact_sol.reshape(ntest, ntest),
                                    vmin=-maxval, vmax=maxval, extent=[0, lx, 0, ly], origin='lower')
                # add colorbar
                cbar1 = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
                cbar2 = fig.colorbar(im2, ax=ax[0], orientation='horizontal')
                ax[0].set_title('Analytical')
                ax[1].set_title('MFS')
                plt.show()
            if plot_bound_cond:
                plot_bound_values(op, alpha, smfs.boundv, coeffs_id=smfs.boundv_ids, show=True, plot3d=True)
    # do a linear regression on the log-log plot
    log_nb_bound = np.log(nb_bound)
    log_l2_error, log_l2_rel_error = np.log(conditions), np.log(l2_rel_error)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for itf in range(len(freqs)):
        intercept_l2, slope_l2 = np.polynomial.polynomial.polyfit(log_nb_bound, log_l2_rel_error[itf], 1)
        ax[0].loglog(nb_bound, conditions[itf, :], 'o-')
        ax[1].loglog(nb_bound, l2_rel_error[itf, :], 'o-', label='{} Hz slope: {:.3}'.format(freqs[itf], slope_l2))

    ax[0].set_xlabel('Number of boundary points')
    ax[0].set_ylabel('Condition number')
    ax[1].set_xlabel('Number of boundary points')
    ax[1].set_ylabel('L2 relative error')
    ax[0].legend(), ax[1].legend()
    plt.show()

elif test_type == "comsol_polygon":
    if bound_cond == "neumann":
        fem_data = np.load("data/Helmholtz_Neumann_polygon.npy")
    else:
        fem_data = np.load("data/Helmholtz_Robin_polygon2.npy")
    # extracting frequencies from keys
    freqs = np.loadtxt("data/freqs_polygon.txt", dtype=float)[:1]
    print("frequency list :", freqs)
    points, fem_data = np.real(fem_data[:, :2]), fem_data[:, 2:]
    skip_len = 1500
    points, fem_data = points[::skip_len, :], fem_data[::skip_len, :]
    print("Number of data points: ", len(points))
    npoints = [1500, 2000, 2500, 3000]
    l2_comsol_mfs = np.zeros((len(npoints), len(freqs)))
    conditions = np.zeros((len(npoints), len(freqs)))
    # convergence analysis
    for indf, f in enumerate(freqs):
        data_val = fem_data[:, indf]  # FEM solution for the given frequency
        k = 2 * np.pi * f / 343

        for indp, npo in enumerate(npoints):
            print("Solving for frequency {} Hz with {} sources".format(f, npo))
            smfs = SlidingMFS(nbound=npo, nsrc=500, nobs=200, pos_src=pos_src, k=k, poly=poly,
                              dx=dx, cond=bound_cond, beta=beta)

            if plot_bound:  # plot boundary
                plot_boundary(smfs.bound, smfs.normvec, smfs.boundv, smfs.pos_src)
            alpha, conditions[indp, indf] = smfs.gram_fit()  # solve the system
            sol = compute_sol(smfs.ampl, points, smfs.boundv, k, pos_src)
            print("Done")
            print("min and max values : r part {},{} ; i part {},{} ".format(np.min(alpha.real), np.max(alpha.real),
                                                                             np.min(alpha.imag), np.max(alpha.imag)))

            if plot_sol:
                val = np.linspace(np.min(np.real(data_val)), np.max(np.real(data_val)), 100)
                val2 = np.linspace(np.min(np.real(sol)), np.max(np.real(sol)), 100)

                fig, ax = plt.subplots(1, 3)
                ax[0].tricontourf(points[:, 0], points[:, 1], np.real(data_val), levels=val)
                ax[1].tricontourf(points[:, 0], points[:, 1], np.real(sol), val2)
                rel_error = np.abs((np.real(sol) - data_val) / (data_val + 1e-16))
                rel_error[rel_error > 2] = 2
                res = ax[2].tricontourf(points[:, 0], points[:, 1], rel_error)
                plt.colorbar(res)
                ax[0].set_aspect('equal'), ax[1].set_aspect('equal'), ax[2].set_aspect('equal')

                plt.show()
            if plot_bound_cond:
                if bound_cond == "robin":
                    kwargs["beta"] = smfs.beta

                plot_bound_values(op, alpha, smfs.boundv, coeffs_id=smfs.boundv_ids, show=True, plot3d=True,
                                  name="poly_dir_lsqr_500_uniform_mean.png", **kwargs)
            l2_comsol_mfs[indp, indf] = np.linalg.norm(sol - data_val) / np.linalg.norm(data_val)
            print("l2 error :", l2_comsol_mfs[indp, indf])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(len(freqs)):
        ax[1].loglog(npoints, l2_comsol_mfs[:, i], label=str(freqs[i]) + ' Hz')
        ax[0].loglog(npoints, conditions[:, i], 'o-')
    plt.legend()
    plt.show()

elif test_type == "comsol_modes":
    fem_data = np.load("data/Helmholtz_Neumann_1.npy")

    # extracting frequencies from keys
    freqs = np.loadtxt("data/freqs_polygon.txt", dtype=float)[:1]  # same frequencies tested for polygon
    print("frequency list :", freqs)

    points, fem_data = fem_data[:, :2], fem_data[:, 2:]
    skip_len = 1500
    points, fem_data = points[::skip_len, :], fem_data[::skip_len, :]

    np.save("data/points_rect.npy", points)

    print("Number of data points: ", len(points))
    boundary_cond = "neumann"
    npoints = [50, 200, 300, 500, 800, 1200, 1500, 2000, 2500, 3000, 3500, 4000]
    l2_comsol_mfs = np.zeros((len(npoints), len(freqs)))
    l2_mfs_analytical = np.zeros((len(npoints), len(freqs)))
    l2_comsol_analytical = np.zeros(len(freqs))

    # convergence analysis
    for indf, f in enumerate(freqs):
        data_val = fem_data[:, indf]  # FEM solution for the given frequency
        k = 2 * np.pi * f / 343
        pos_src = np.array([[2.5, 2.5]])

        exact_sol = np.load("data/exact_sol_rect_{}_{}.npy".format(int(f), 15000)).squeeze()
        l2_comsol_analytical[indf] = np.linalg.norm(exact_sol - data_val) / np.linalg.norm(exact_sol)
        print("l2 error fem-analytical :", l2_comsol_analytical[indf])
        for indp, npo in enumerate(npoints):
            # boundary points
            bound, norm_vec = poly.sample(npo, distribution=distribution)

            # virtual boundary points
            boundv = bound + dx * norm_vec

            # plot boundary
            if plot_bound:
                plot_boundary(bound, norm_vec, boundv, pos_src)

            alpha = solve(0, norm_vec, bound, boundv, pos_src, k, source_type='dirac', cond=boundary_cond)
            print("Solving for frequency {} Hz with {} sources".format(f, npo))
            sol = compute_sol(alpha, points, boundv, k, pos_src)
            print("Done")

            if plot_sol:
                val = np.linspace(-0.5, 0.5, 100)
                fig, ax = plt.subplots(3)
                ax[0].set_title("FEM solution"), ax[1].set_title("MFS solution"), ax[2].set_title("Relative error")
                ax[0].tricontourf(points[:, 0], points[:, 1], data_val, val)
                ax[1].tricontourf(points[:, 0], points[:, 1], np.real(sol), val)
                rel_error = np.abs((np.real(sol) - data_val) / (data_val + 1e-16))
                rel_error[rel_error > 2] = 2
                res = ax[2].tricontourf(points[:, 0], points[:, 1], rel_error)
                plt.colorbar(res)
                plt.show()

            l2_comsol_mfs[indp, indf] = np.linalg.norm(np.real(sol) - data_val) / np.linalg.norm(data_val)
            print("l2 error mfs-fem :", l2_comsol_mfs[indp, indf])
            l2_mfs_analytical[indp, indf] = np.linalg.norm(np.real(sol) - exact_sol) / np.linalg.norm(exact_sol)
            print("l2 error mfs-analytical :", l2_mfs_analytical[indp, indf])

    for i in range(len(freqs)):
        line, = plt.loglog(npoints, l2_comsol_mfs[:, i], label=str(freqs[i]) + ' Hz')
        c = line.get_color()
        plt.hlines(l2_comsol_analytical[i], npoints[0], npoints[-1], linestyles='dotted', colors=c)
        # darkening the color
        c = [0.75 * x for x in colors.to_rgba(c)[:3]] + [1, ]
        line, = plt.loglog(npoints, l2_mfs_analytical[:, i], marker="1", color=c, linestyle="dashed")

    plt.legend()
    plt.show()

if test_type == "distance":
    dx_val = np.array([0.1, 0.2, 0.5, 1., 1.5, 2., 3., 4., 5., 10.])

    nb_bound = [100, 300, 500, 1000]
    freq = 100
    k = 2 * np.pi * freq / 343
    pos_src = np.array([[2.5, 2.5]])
    ntest, nmax = 50, 15000
    points = np.meshgrid(np.linspace(0.1, lx - 0.1, ntest), np.linspace(0.1, ly - 0.1, ntest))
    points = np.stack(points, axis=2).reshape(-1, 2)
    exact_sol = np.loadtxt(exact_path + "exact_sol_{}_{}_{}.csv".format(0, nmax, ntest), delimiter=",")

    conditions, l2_rel_error = np.zeros([len(dx_val), len(nb_bound)]), np.zeros([len(dx_val), len(nb_bound)])
    for ind, dx in enumerate(dx_val):
        for indb, nb in enumerate(nb_bound):
            print("dx = ", dx, "nb = ", nb)
            bound, norm_vec = poly.sample(nb, distribution=distribution)
            boundv = bound + dx * norm_vec
            # solve the system
            alpha, mat = solve(0, norm_vec, bound, boundv, pos_src, k, method='direct',
                               source_type='dirac', cond=bound_cond, return_mat=True)
            print("Solving for frequency {} Hz with {} sources".format(freq, nb))
            sol = compute_sol(alpha, points, boundv, k, pos_src)
            print("Done")

            # compute the relative error
            l2_rel_error[ind, indb] = np.linalg.norm(np.real(sol) - exact_sol) / np.linalg.norm(exact_sol)
            conditions[ind, indb] = np.linalg.cond(mat)

    fig, ax = plt.subplots(2)
    ax[0].set_title("Condition number"), ax[1].set_title("Relative error")
    for i in range(len(nb_bound)):
        ax[0].loglog(dx_val, conditions[:, i], label=str(nb_bound[i]) + " sources")
        ax[1].loglog(dx_val, l2_rel_error[:, i], label=str(nb_bound[i]) + " sources")
    ax[0].legend(), ax[1].legend()
    ax[1].set_xlabel("dx"), ax[0].set_xlabel("dx")
    plt.tight_layout()
    plt.show()

if test_type == "distance_polygon":
    dx_val = np.array([0.1, 0.2, 0.5, 1., 1.5, 2., 3., 4., 5., 10.])

    nb_bound = [800, 1000, 1300]
    fem_data = np.load("data/Helmholtz_Neumann_polygon.npy")

    # extracting frequencies from keys
    freq = np.loadtxt("data/freqs_polygon.txt", dtype=float)[0]
    print("freq = ", freq)
    k = 2 * np.pi * freq / 343

    points, fem_data = fem_data[:, :2], fem_data[:, 2:]
    skip_len = 1500
    points, fem_data = points[::skip_len, :], fem_data[::skip_len, :]
    fem_data = fem_data[:, 0]
    conditions, l2_rel_error = np.zeros([len(dx_val), len(nb_bound)]), np.zeros([len(dx_val), len(nb_bound)])
    for ind, dx in enumerate(dx_val):
        for indb, nb in enumerate(nb_bound):
            print("dx = ", dx, "nb = ", nb)
            bound, norm_vec = poly.sample(nb, distribution=distribution)
            boundv = bound + dx * norm_vec

            # for sampling on an ellipse
            # barycenter = np.mean(vertices, axis=0)
            # distances = np.linalg.norm(vertices - barycenter, axis=1)
            # maxdist = np.max(distances)
            # ellipse = Ellipse(radiusx=maxdist + dx, radiusy=maxdist + dx, center=barycenter)
            # boundv, _ = ellipse.sample(nb)

            # solve the system
            alpha, mat = solve(0, norm_vec, bound, boundv, pos_src, k,
                               source_type='dirac', cond=bound_cond, method='direct', return_mat=True)
            print("Solving for frequency {} Hz with {} sources".format(freq, nb))
            sol = compute_sol(alpha, points, boundv, k, pos_src)
            print("Done")

            l2_rel_error[ind, indb] = np.linalg.norm(np.real(sol) - fem_data) / np.linalg.norm(fem_data)
            conditions[ind, indb] = np.linalg.cond(mat)

    fig, ax = plt.subplots(2)
    ax[0].set_title("Condition number"), ax[1].set_title("Relative error")
    for i in range(len(nb_bound)):
        ax[0].loglog(dx_val, conditions[:, i], label=str(nb_bound[i]) + " sources")
        ax[1].loglog(dx_val, l2_rel_error[:, i], label=str(nb_bound[i]) + " sources")
    ax[0].legend(), ax[1].legend()
    ax[1].set_xlabel("dx"), ax[0].set_xlabel("dx")
    plt.tight_layout()
    plt.show()

if test_type == "sliding_polygon":

    fem_data = np.load("data/Helmholtz_Neumann_polygon.npy")

    # extracting frequencies from keys
    freqs = np.loadtxt("data/freqs_polygon.txt", dtype=float)[:1]
    print("frequency list :", freqs)

    points, fem_data = fem_data[:, :2], fem_data[:, 2:]
    skip_len = 1500
    points, fem_data = points[::skip_len, :], fem_data[::skip_len, :]
    print("Number of data points: ", len(points))
    boundary_cond = "neumann"
    npoints = [500, 800, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4001, 5000]
    l2_comsol_mfs = np.zeros((len(npoints), len(freqs)))
    conditions = np.zeros((len(npoints), len(freqs)))
    # convergence analysis
    for indf, f in enumerate(freqs):

        data_val = fem_data[:, indf]  # FEM solution for the given frequency
        k = 2 * np.pi * f / 343

        for indp, npo in enumerate(npoints):

            smfs = SlidingMFS(nbound=npo, nsrc=500, nobs=200, pos_src=pos_src, k=k, poly=poly, dx=dx, cond="neumann")
            ini = np.concatenate([np.real(smfs.ampl), np.imag(smfs.ampl), smfs.paramsv])

            if plot_bound:
                plot_boundary(smfs.bound, smfs.normvec, smfs.boundv, pos_src)
            smfs.src_slide_step(verbose=True)
            if plot_bound:
                plot_boundary(smfs.bound, smfs.normvec, smfs.boundv, pos_src)

            smfs.mfs_fit()
            alpha = smfs.ampl
            print("min and max values : ", np.min(alpha), np.max(alpha))
            print("Solving for frequency {} Hz with {} sources".format(f, npo))
            sol = compute_sol(alpha, points, smfs.boundv, k, pos_src)
            print("Done")

            if plot_sol:
                val = np.linspace(np.min(data_val), np.max(data_val), 100)
                val2 = np.linspace(np.min(np.real(sol)), np.max(np.real(sol)), 100)

                fig, ax = plt.subplots(1, 3)

                ax[0].tricontourf(points[:, 0], points[:, 1], data_val, val)
                ax[1].tricontourf(points[:, 0], points[:, 1], np.real(sol), val2)
                rel_error = np.abs((np.real(sol) - data_val) / (data_val + 1e-16))
                rel_error[rel_error > 2] = 2
                res = ax[2].tricontourf(points[:, 0], points[:, 1], rel_error)
                plt.colorbar(res)
                ax[0].set_aspect('equal'), ax[1].set_aspect('equal'), ax[2].set_aspect('equal')

                plt.show()
            if plot_bound_cond:
                plot_bound_values(alpha, smfs.boundv, show=False, coeffs_id=smfs.boundv_ids,
                                  name="lstsq_500_uniform_mdist2001")

            l2_comsol_mfs[indp, indf] = np.linalg.norm(np.real(sol) - data_val) / np.linalg.norm(data_val)
            print("l2 error :", l2_comsol_mfs[indp, indf])

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(freqs)):
        ax.loglog(npoints, l2_comsol_mfs[:, i], label=str(freqs[i]) + ' Hz')
    plt.legend()
    plt.show()
