"""Compare the Method of Fundamental Solutions to the analytical solution in 2D on different rectangles."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import sample_polygon, plot_boundary, Polygon, out_polygon
from mfs import compute_sol, solve

import pyroomacoustics as pra
std_off = pra.constants.get("frac_delay_length") // 2
lx, ly = 5.2, 3.3
dx = 0.2
plot_sol = True
plot_bound = True
test_type = "comsol_polygon"
exact_path = "data/"

if test_type == "convergence":
    freqs = np.loadtxt(exact_path + "freqs.txt", delimiter=",")

    nb_bound = np.array([200, 300, 500, 800, 1200, 1500, 2000, 2500, 3000, 3500])
    l2_error, l2_rel_error = np.zeros([len(freqs), len(nb_bound)]), np.zeros([len(freqs), len(nb_bound)])

    for itf, fs in enumerate(freqs):
        k = 2*np.pi*fs/343
        # boundary points
        vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]])
        pos_src = np.array([[2.5, 2.5]])

        ntest, nmax = 50, 15000
        points = np.meshgrid(np.linspace(0.1, lx-0.1, ntest), np.linspace(0.1, ly-0.1, ntest))
        points = np.stack(points, axis=2).reshape(-1, 2)

        exact_sol = np.loadtxt(exact_path + "exact_sol_{}_{}_{}.csv".format(itf, nmax, ntest), delimiter=",")
        exact_sol = exact_sol.flatten()

        for itp, npo in enumerate(nb_bound):
            bound, norm_vec = sample_polygon(npo, np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]]))
            boundv = bound + dx * norm_vec  # virtual boundary points

            if plot_bound:  # plot boundary
                plot_boundary(bound, norm_vec, boundv, pos_src)

            # compute the solution
            alpha = solve(0., norm_vec, bound, boundv, pos_src, k, source_type='dirac', cond='neumann')
            print("Solving for frequency {} Hz with {} sources".format(fs, npo))
            sol = compute_sol(alpha, points, boundv, k, pos_src)
            print("Done")

            print("imaginary part norm: {}".format(np.linalg.norm(np.imag(sol))))

            l2_error[itf, itp] = np.linalg.norm(np.abs(sol - exact_sol))
            l2_rel_error[itf, itp] = np.linalg.norm(sol - exact_sol) / np.linalg.norm(exact_sol)
            print("L1 error: {}, L2 error: {}".format(l2_error[itf, itp], l2_rel_error[itf, itp]))
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

    # do a linear regression on the log-log plot
    log_nb_bound = np.log(nb_bound)
    log_l2_error, log_l2_rel_error = np.log(l2_error), np.log(l2_rel_error)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for itf in range(len(freqs)):
        intercept_l1, slope_l1 = np.polynomial.polynomial.polyfit(log_nb_bound, log_l2_error[itf], 1)
        intercept_l2, slope_l2 = np.polynomial.polynomial.polyfit(log_nb_bound, log_l2_rel_error[itf], 1)
        ax[0].loglog(nb_bound, l2_error[itf, :], 'o-', label='{} Hz slope: {:.3}'.format(freqs[itf], slope_l1))
        ax[1].loglog(nb_bound, l2_rel_error[itf, :], 'o-')

    ax[0].set_xlabel('Number of boundary points')
    ax[0].set_ylabel('L2 error')
    ax[1].set_xlabel('Number of boundary points')
    ax[1].set_ylabel('L2 relative error')
    ax[0].legend(), ax[1].legend()
    plt.show()

elif test_type == "comsol_polygon":
    fem_data = np.load("data/Helmholtz_Neumann_polygon.npy")
    vertices = np.array([[2.2, 0], [4.3, 0.], [5.7, 1.9], [3.3, 5.1], [1.7, 5.5], [-0.75, 2.75]])

    # extracting frequencies from keys
    freqs = np.loadtxt("data/freqs_polygon.txt", dtype=float)[:1]
    print("frequency list :", freqs)

    points, fem_data = fem_data[:, :2], fem_data[:, 2:]
    skip_len = 1500
    points, fem_data = points[::skip_len, :], fem_data[::skip_len, :]
    print("Number of data points: ", len(points))
    boundary_cond = "neumann"
    npoints = [200, 300, 500, 800, 1200, 1500, 2000, 2500, 3000, 3500, 4000]
    l2_comsol_mfs = np.zeros((len(npoints), len(freqs)))

    # convergence analysis
    for indf, f in enumerate(freqs):
        data_val = fem_data[:, indf]  # FEM solution for the given frequency
        k = 2 * np.pi * f / 343
        pos_src = np.array([[1., 2.5]])

        for indp, npo in enumerate(npoints):
            # boundary points
            bound, norm_vec = sample_polygon(npo, vertices)

            # virtual boundary points
            boundv = bound + dx * norm_vec

            # plot boundary
            if plot_bound:
                plot_boundary(bound, norm_vec, boundv, pos_src)

            poly = Polygon(vertices)

            alpha = solve(0, norm_vec, bound, boundv, pos_src, k, source_type='dirac', cond=boundary_cond)
            print("Solving for frequency {} Hz with {} sources".format(f, npo))
            sol = compute_sol(alpha, points, boundv, k, pos_src)
            print("Done")

            # removing the points out of the polygon
            out_points = out_polygon(points, poly)

            sol[out_points] = np.nan  # + 1j * np.nan
            data_val[np.isnan(sol)] = 0
            sol[np.isnan(sol)] = 0

            if plot_sol:
                val = np.linspace(np.min(data_val), np.max(data_val), 100)
                fig, ax = plt.subplots(3)

                ax[0].tricontourf(points[:, 0], points[:, 1], data_val, val)
                ax[1].tricontourf(points[:, 0], points[:, 1], np.real(sol), val)
                rel_error = np.abs((np.real(sol) - np.real(data_val)) / (np.real(data_val) + 1e-16))
                rel_error[rel_error > 2] = 2
                res = ax[2].tricontourf(points[:, 0], points[:, 1], rel_error)
                plt.colorbar(res)
                plt.show()

            l2_comsol_mfs[indp, indf] = np.linalg.norm(np.real(sol) - data_val) / np.linalg.norm(data_val)
            print("l2 error :", l2_comsol_mfs[indp, indf])

    for i in range(len(freqs)):
        line, = plt.loglog(npoints, l2_comsol_mfs[:, i], label=str(freqs[i]) + ' Hz')
    plt.legend()
    plt.show()

elif test_type == "comsol_modes":
    fem_data = np.load("data/Helmholtz_Neumann_1.npy")
    vertices = np.array([[0, 0], [lx, 0.], [lx, ly], [0, ly]])

    # extracting frequencies from keys
    freqs = np.loadtxt("data/freqs_polygon.txt", dtype=float)  # same frequencies tested for polygon
    print("frequency list :", freqs)

    points, fem_data = fem_data[:, :2], fem_data[:, 2:]
    skip_len = 1500
    points, fem_data = points[::skip_len, :], fem_data[::skip_len, :]
    print(points.shape, fem_data.shape)
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
            bound, norm_vec = sample_polygon(npo, vertices)

            # virtual boundary points
            boundv = bound + dx * norm_vec

            # plot boundary
            if plot_bound:
                plot_boundary(bound, norm_vec, boundv, pos_src)

            poly = Polygon(vertices)

            alpha = solve(0, norm_vec, bound, boundv, pos_src, k, source_type='dirac', cond=boundary_cond)
            print("Solving for frequency {} Hz with {} sources".format(f, npo))
            sol = compute_sol(alpha, points, boundv, k, pos_src)
            print("Done")

            # removing the points out of the polygon
            out_points = out_polygon(points, poly)

            sol[out_points] = np.nan  # + 1j * np.nan
            data_val[np.isnan(sol)] = 0
            sol[np.isnan(sol)] = 0

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
