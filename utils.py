from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
c = 343


def plot_complex(xx, yy, data, extent, vmin=None, vmax=None):
    """ plot real, imaginary and modulus values of the solution with separate colorbars as well as their contours """

    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    for i, d in enumerate([np.real(data), np.imag(data), np.abs(data)]):
        im = ax[0, i].imshow(d.reshape(xx.shape), extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        ax[0, i].set_title(['real', 'imag', 'abs'][i])
        divider = make_axes_locatable(ax[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        im = ax[1, i].contour(xx, yy, d.reshape(xx.shape), levels=10, colors='k', linewidths=0.5)
        divider = make_axes_locatable(ax[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    plt.show()


def sample_rectangle(npoints, lx, ly, dedge=0.015):
    dx = 2*(lx + ly) / npoints
    nx = int(np.rint(lx / dx))
    ny = (npoints - 2*nx) // 2
    res = npoints - 2*nx - 2*ny

    x = np.linspace(dedge, lx-dedge, nx, endpoint=True)
    yright = np.linspace(dedge, ly-dedge, ny, endpoint=True)
    yleft = np.linspace(dedge, ly-dedge, ny+res, endpoint=True)
    return (np.stack([np.concatenate([x, lx*np.ones_like(yright), x[::-1], np.zeros_like(yleft)]),
                     np.concatenate([np.zeros_like(x), yright, ly*np.ones_like(x), yleft[::-1]])], axis=1),
            np.stack([np.concatenate([np.zeros_like(x), np.ones_like(yright), np.zeros_like(x), -np.ones_like(yleft)]),
                      np.concatenate([-np.ones_like(x), np.zeros_like(yright), np.ones_like(x), np.zeros_like(yleft)])],
                     axis=1))


def sample_polygon(npoints, vertices, dedge=0.005):
    """Compute the boundary and outside normal vectors of a polygon given its vertices given in counter-clockwise order.
    The number of points per edge is weighted by edge length."""
    lengths = np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1)
    lengths /= np.sum(lengths)
    lengths = np.round(lengths * npoints).astype(int)
    lengths[-1] += npoints - np.sum(lengths)
    boundary = np.zeros((npoints, 2))
    normal = np.zeros((npoints, 2))
    indstart = 0
    for i, (v1, v2, l) in enumerate(zip(vertices, np.roll(vertices, -1, axis=0), lengths)):
        edge = v2 - v1
        normal[indstart:indstart + l, :] = np.reshape(np.array([edge[1], -edge[0]]), (1, 2))
        boundary[indstart:indstart + l, :] = v1 + np.linspace(dedge, 1-dedge, l, endpoint=True)[:, np.newaxis] * edge
        indstart += l
    return boundary, normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]


def sample_ellipse(npoints, radiusx, radiusy):
    theta = np.linspace(0, 2*np.pi, npoints)
    normvec = np.stack([radiusy*np.cos(theta), radiusx*np.sin(theta)], axis=1)
    return (np.stack([radiusx*np.cos(theta), radiusy*np.sin(theta)], axis=1),
            normvec / np.linalg.norm(normvec, axis=1)[:, np.newaxis])


def translate_boundary(boundary, normal, offset):
    return boundary + offset*normal


def out_polygon(points, vertices, min_dist=0.0):
    """Check if points (shape (npoints, d)) are outside and at a distance superior to min_dist of a (not necessarily
    convex) polygon given by its vertices (shape (nvertices, d)) in counter-clockwise order, or by a Polygon object.
    Return an array of bool."""
    outside = np.zeros(points.shape[0], dtype=bool)
    if type(vertices) is not Polygon:
        poly = Polygon(vertices)
    else:
        poly = vertices

    for i, p in enumerate(points):
        pp = Point(p)
        outside[i] = (not poly.contains(pp)) and (poly.distance(pp) > min_dist)
    return outside
