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


def chebyshev_nodes(a, b, N):
    """Compute N Chebyshev nodes on the interval [a, b]."""
    return 0.5*(a + b) + 0.5*(b - a)*np.cos(np.pi*(2*np.arange(N)[::-1] + 1) / (2*N))


def sample_polygon(npoints, vertices):
    """Compute the boundary and outside normal vectors of a polygon using Chebyshev nodes to sample the edges."""
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
        boundary[indstart:indstart + l, :] = v1 + chebyshev_nodes(0, 1, l)[:, np.newaxis] * edge
        indstart += l
    return boundary, normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]


def sample_cuboid_chebyshev(npoints, lx, ly, lz):
    """Compute the boundary and outside normal vectors of a cuboid using Chebyshev nodes to sample the faces, weight
    by face surface."""
    ax, ay, az = ly*lz, lx*lz, lx*ly
    npfx = int(np.rint(ax / (ax + ay + az) * npoints)) // 2  # number of points for each x face
    npfy = int(np.rint(ay / (ax + ay + az) * npoints)) // 2
    npfz = int(np.rint(az / (ax + ay + az) * npoints)) // 2
    print('npoints per face: ', npfx, npfy, npfz)
    yx = chebyshev_nodes(0, ly, int(np.sqrt(npfx * ly / lz)))
    zx = chebyshev_nodes(0, lz, int(npfx/len(yx)))

    points = np.meshgrid(yx, zx, indexing='ij')
    points = np.stack([np.zeros_like(points[0]), points[0], points[1]], axis=2).reshape((-1, 3))
    n = len(points)
    normals = np.stack([-np.ones(n), np.zeros(n), np.zeros(n)], axis=1)

    points = np.concatenate([points, points + np.array([lx, 0, 0])], axis=0)
    normals = np.concatenate([normals, -normals], axis=0)

    xy = chebyshev_nodes(0, lx, int(np.sqrt(npfy * lx/lz)))
    zy = chebyshev_nodes(0, lz, int(npfy / len(xy)))

    pointsy = np.meshgrid(xy, zy, indexing='ij')
    pointsy = np.stack([pointsy[0], np.zeros_like(pointsy[0]), pointsy[1]], axis=2).reshape((-1, 3))
    n = len(pointsy)
    normalsy = np.stack([np.zeros(n), -np.ones(n), np.zeros(n)], axis=1)

    points = np.concatenate([points, pointsy, pointsy + np.array([0, ly, 0])], axis=0)
    normals = np.concatenate([normals, normalsy, -normalsy], axis=0)

    xz = chebyshev_nodes(0, lx, int(np.sqrt(npfz * lx / ly)))
    yz = chebyshev_nodes(0, ly, int(npfz/len(xz)))

    pointsz = np.meshgrid(xz, yz, indexing='ij')
    pointsz = np.stack([pointsz[0], pointsz[1], np.zeros_like(pointsz[0])], axis=2).reshape((-1, 3))
    n = len(pointsz)
    normalsz = np.stack([np.zeros(n), np.zeros(n), -np.ones(n)], axis=1)

    points = np.concatenate([points, pointsz, pointsz + np.array([0, 0, lz])], axis=0)
    normals = np.concatenate([normals, normalsz, -normalsz], axis=0)

    return points, normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]


def sample_ellipse(npoints, radiusx, radiusy):
    theta = np.linspace(0, 2*np.pi, npoints)
    normvec = np.stack([radiusy*np.cos(theta), radiusx*np.sin(theta)], axis=1)
    return (np.stack([radiusx*np.cos(theta), radiusy*np.sin(theta)], axis=1),
            normvec / np.linalg.norm(normvec, axis=1)[:, np.newaxis])


def compute_normals(boundary):
    """Compute the normal vectors to the boundary given by its vertices in counter-clockwise order. Use the formula :
    n = (-y_{i+1} + y_{i-1}, -x_{i-1} + x_{i+1})/2"""
    normals = np.stack([-np.roll(boundary[:, 1], 1) + np.roll(boundary[:, 1], -1),
                        -np.roll(boundary[:, 0], -1) + np.roll(boundary[:, 0], 1)], axis=1)
    return normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]


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


def plot_boundary(bound, norm_vec, boundv, pos_src):
    plt.plot(bound[:, 0], bound[:, 1], '.-', label='boundary')
    plt.quiver(bound[:, 0], bound[:, 1], norm_vec[:, 0], norm_vec[:, 1])
    plt.plot(boundv[:, 0], boundv[:, 1], '.', label='virtual boundary')
    plt.plot(pos_src[:, 0], pos_src[:, 1], 'o', label='source')
    plt.axis('equal')
    plt.legend()
    plt.show()
