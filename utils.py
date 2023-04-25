from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
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


def sample_polygon(npoints, vertices, distribution="linear"):
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
        if distribution == "linear":
            nodes = np.linspace(0, 1, l+2)[1:-1]
        else:
            nodes = chebyshev_nodes(0, 1, l)
        boundary[indstart:indstart + l, :] = v1 + edge * nodes[:, np.newaxis]
        indstart += l
    return boundary, normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]


def sample_points_on_circle(center, radius, start_vec, end_vec, num_points):
    # Compute the start and end angles of the circular arc
    start_vec = start_vec / np.linalg.norm(start_vec)
    end_vec = end_vec / np.linalg.norm(end_vec)

    start_angle = np.arctan2(start_vec[1], start_vec[0]) % (2*np.pi)
    end_angle = np.arctan2(end_vec[1], end_vec[0])% (2*np.pi)
    if end_angle < start_angle:
        end_angle += 2*np.pi

    angles = np.linspace(start_angle, end_angle, num_points)

    points = center + radius*np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return points


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
        poly = ShapelyPolygon(vertices)
    else:
        poly = vertices

    for i, p in enumerate(points):
        pp = ShapelyPoint(p)
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


class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices
        edges = np.roll(vertices, -1, axis=0) - vertices
        self.normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        self.normals /= np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        self.nedges = len(vertices)
        self.lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)  # length of each edge
        # compute angles between edges
        self.angles = np.zeros(self.nedges)
        for i in range(self.nedges):
            self.angles[i] = np.pi - np.arccos(np.dot(edges[i], edges[(i + 1) % self.nedges]) /
                                       (self.lengths[i] * self.lengths[(i + 1) % self.nedges]))

    def enlarge(self, factor):
        """Enlarge the polygon by a factor, by moving the vertices along their normals."""
        barycenter = np.mean(self.vertices, axis=0)
        vertices_shifted = self.vertices - barycenter
        self.vertices = vertices_shifted * factor + barycenter

    def sample_edge(self, n, edge_id, distribution='uniform', min_dist=None, return_nodes=False):
        if distribution == "uniform":
            if min_dist is None:
                nodes = np.linspace(0, 1, n + 2)[1:-1]
            else:
                relative_dist = min_dist / self.lengths[edge_id]
                nodes = np.linspace(relative_dist, 1 - relative_dist, n, endpoint=True)

        else:
            nodes = chebyshev_nodes(0, 1, n)

        v1, v2 = self.vertices[edge_id], self.vertices[(edge_id + 1) % self.nedges]
        res = v1 + nodes[:, np.newaxis]*(v2 - v1)

        if return_nodes:
            return res, nodes
        else:
            return res

    def sample(self, npoints, distribution='uniform', min_dist=None, return_ids=False, return_nodes=False):
        lengths = self.lengths / np.sum(self.lengths)
        lengths = np.round(lengths * npoints).astype(int)
        lengths[-1] += npoints - np.sum(lengths)
        boundary, normals = np.zeros((npoints, 2)), np.zeros((npoints, 2))
        indstart = 0
        ids, params = np.zeros(npoints, dtype=int), []
        for i in range(self.nedges):
            n = lengths[i]
            normals[indstart:indstart + n] = self.normals[i]
            boundary[indstart:indstart + n], nodes = self.sample_edge(n, i, distribution,
                                                                      min_dist=min_dist, return_nodes=True)
            indstart += n
            if return_ids:
                ids[indstart - n:indstart] = i
            if return_nodes:
                params.append(nodes)

        res = (boundary, normals)
        if return_ids:
            res += (ids,)
        if return_nodes:
            params = np.concatenate(params)
            res += (params,)

        return res

    def sample_rounded(self, npoints, radius):
        """Sample on an enlarged version of the polygon (boundary translated by radius in the normal direction), with
            rounded edges."""
        # compute number of points for each edge and rounded corner
        lengths = np.zeros(2 * len(self.vertices))
        lengths[::2] = np.linalg.norm(self.vertices - np.roll(self.vertices, -1, axis=0), axis=1)
        normals = np.roll(self.vertices, -1, axis=0) - self.vertices
        normals = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # add rounded corners, ponderate each corner by angle
        angles = np.arccos(np.sum(normals * np.roll(normals, -1, axis=0), axis=1))
        lengths[1::2] = np.roll(angles, 1) * radius
        lengths /= np.sum(lengths)
        lengths = np.round(lengths * npoints).astype(int)
        lengths[-1] += npoints - np.sum(lengths)

        boundary = np.zeros((npoints, 2))
        indstart = 0
        for i in range(len(lengths)):
            le = lengths[i]
            v1, v2 = self.vertices[i // 2], self.vertices[(i // 2 + 1) % len(self.vertices)]
            if i % 2 == 0:
                edge = v2 - v1
                boundary[indstart:indstart + le, :] = v1 + normals[i // 2] * radius + edge * np.linspace(0, 1, le)[
                                                                                             :, np.newaxis]
            else:
                vect1, vect2 = normals[i // 2 - 1], normals[i // 2]
                boundary[indstart:indstart + le, :] = sample_points_on_circle(v1, radius, vect1, vect2, le + 2)[1:-1, :]
            indstart += le

        return boundary

    def is_out(self, points, min_dist=0.0):
        """Check if points (shape (npoints, d)) are outside and at a distance superior to min_dist of a (not necessarily
         convex) polygon given by its vertices (shape (nvertices, d)) in counter-clockwise order, or by a Polygon object.
        Return an array of bool."""
        outside = np.zeros(points.shape[0], dtype=bool)
        poly = ShapelyPolygon(self.vertices)

        for i, p in enumerate(points):
            pp = ShapelyPoint(p)
            outside[i] = (not poly.contains(pp)) and (poly.distance(pp) > min_dist)
        return outside

    def get_normal(self, edge_id):
        return self.normals[edge_id]

    def parametrization_to_coord(self, edge_id, param):
        """Given an array of integers edge_id and an array of floats param (comprised between 0 and 1), return the
        corresponding coordinates in 2D"""
        v1, v2 = self.vertices[edge_id], self.vertices[(edge_id + 1) % self.nedges]
        return v1 + param[:, np.newaxis] * (v2 - v1)


class Ellipse:
    def __init__(self, radiusx, radiusy, center=None):
        self.radiusx = radiusx
        self.radiusy = radiusy
        if center is None:
            self.center = np.zeros(2)
        else:
            self.center = center.copy()

    def sample(self, npoints):
        theta = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
        normvec = np.stack([self.radiusy * np.cos(theta), self.radiusx * np.sin(theta)], axis=1)
        return (np.stack([self.radiusx*np.cos(theta), self.radiusy*np.sin(theta)], axis=1) + self.center[np.newaxis, :],
                normvec / np.linalg.norm(normvec, axis=1)[:, np.newaxis])
