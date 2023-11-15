from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
import pygmsh

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


def chebyshev_nodes(a, b, N):
    """Compute N Chebyshev nodes on the interval [a, b]."""
    return 0.5*(a + b) + 0.5*(b - a)*np.cos(np.pi*(2*np.arange(N)[::-1] + 1) / (2*N))


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
    def __init__(self, inp, origin=None):

        self.polar_angles, self.phi, self.theta = None, None, None
        self.vertices, self.nedges, self.lengths, self.normals = None, None, None, None
        if origin is None:
            self.origin = None
        else:
            self.origin = origin

        # tuple (distances to origin, normal angles)
        if type(inp) is tuple and len(inp) == 2 and type(inp[0]) is np.ndarray and type(inp[1]) is np.ndarray:
            self.update_param(*inp)

        elif type(inp) is np.ndarray:  # vertices
            if self.origin is None:
                self.origin = np.mean(inp, axis=0)
                print('Polygon origin automatically set to {}'.format(self.origin))

            self.update_vertices(inp)

        else:
            raise TypeError('Polygon must be initialized with a numpy array or a tuple of numpy arrays')

    def shift(self, factor):
        """Shift the polygon by a factor, by moving the vertices along the bisecting line.
        Return a new 2D Polygon object."""

        barycenter = np.mean(self.vertices[:self.nedges, :2], axis=0)
        vertices_shifted = self.vertices[:self.nedges, :2] - barycenter
        vertices = (vertices_shifted * factor / np.linalg.norm(vertices_shifted, axis=-1)[:, np.newaxis] +
                    self.vertices[:self.nedges, :2])

        return Polygon(vertices)

    def sample_edge(self, n, edge_id, distribution='uniform', sample_type="total", min_dist=None, return_nodes=False):
        if distribution == "uniform":
            if min_dist is None:
                if sample_type == "total":
                    nodes = np.linspace(0, 1, n + 2)[1:-1]
                else:  # n is the step
                    nodes = np.arange(0, 1, n / self.lengths[edge_id])[1:]  # might end on the last point
            else:
                relative_dist = min_dist / self.lengths[edge_id]

                if sample_type == "total":
                    nodes = np.linspace(relative_dist, 1 - relative_dist, n, endpoint=True)
                else:  # n is the step
                    nodes = np.arange(relative_dist, 1 - relative_dist, n / self.lengths[edge_id])

        else:
            nodes = chebyshev_nodes(0, 1, n)

        v1, v2 = self.vertices[edge_id], self.vertices[(edge_id + 1) % self.nedges]
        res = v1 + nodes[:, np.newaxis]*(v2 - v1)

        if return_nodes:
            return res, nodes
        else:
            return res

    def sample(self, npoints, distribution='uniform', min_dist=None, sample_type="total", return_nodes=False):
        if sample_type == 'total':
            lengths = self.lengths / np.sum(self.lengths)
            lengths = np.round(lengths * npoints).astype(int)
            lengths[-1] += npoints - np.sum(lengths)
        else:  # minimum distance between points
            n = npoints

        boundary, normals, ids, params = [], [], [], []
        for i in range(self.nedges):
            if sample_type == 'total':
                n = lengths[i]

            boundary_edge, nodes = self.sample_edge(n, i, distribution, sample_type=sample_type,
                                                    min_dist=min_dist, return_nodes=True)

            boundary.append(boundary_edge)
            npb = len(nodes)
            normals.append(np.repeat(self.normals[i].reshape([1, -1]), npb, axis=0))
            ids.append([i]*npb)

            if return_nodes:
                params.append(nodes)

        boundary, normals = np.concatenate(boundary, axis=0), np.concatenate(normals, axis=0)
        ids = np.concatenate(ids, axis=0)
        res = (boundary, normals, ids)

        if return_nodes:
            params = np.concatenate(params)
            res += (params,)

        return res

    def get_normal(self, edge_id):
        return self.normals[edge_id]

    def parametrization_to_coord(self, edge_id, param):
        """Given an array of integers edge_id and an array of floats param (comprised between 0 and 1), return the
        corresponding coordinates in 2D"""
        v1, v2 = self.vertices[edge_id], self.vertices[(edge_id + 1) % self.nedges]
        return v1 + param[:, np.newaxis] * (v2 - v1)

    def compute_angles(self, points):
        """Compute polar angles of points with respect to the vertices of the polygon, using the next edge (in
        trigonometric order) as polar axis. Return angles and vectors from vertices to points."""
        diffs = points[:, np.newaxis] - self.vertices[np.newaxis, :]  # shape (npoints, nvertices, 2)
        norms = np.linalg.norm(diffs, axis=2)

        # shape (npoints, nvertices)
        angles = mod2pi(mod2pi(np.arctan2(diffs[:, :, 1], diffs[:, :, 0])) - self.polar_angles[np.newaxis, :])

        return angles, diffs, norms

    def sample_inside(self, sep):
        points, triang = gmsh_sampling(self.vertices[:self.nedges, :], mesh_size=sep)
        return points[:, :2], triang

    def _update_edges_and_angles(self):
        self.edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.nedges = len(self.edges)
        self.lengths = np.linalg.norm(self.edges, axis=1)  # length of each edge
        renormed_edges = self.edges / self.lengths[:, np.newaxis]
        # compute inner angles
        dotprod = np.sum(-np.roll(renormed_edges, 1, axis=0) * renormed_edges, axis=1)
        sin_sign = -np.sign(np.cross(-np.roll(renormed_edges, 1, axis=0), renormed_edges))
        self.angles = np.arccos(np.clip(dotprod, -1, 1))
        self.angles = mod2pi(np.where(sin_sign < 0, 2 * np.pi - self.angles, self.angles))

    def update_param(self, new_phi, new_theta):
        self.phi = new_phi
        self.theta = mod2pi(new_theta)
        self.nedges = len(self.phi)
        self.normals = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)
        if self.origin is None:
            raise ValueError('Polygon must be initialized with an origin if given as a tuple (phi, theta)')
        renormed_normals = self.normals * self.phi[:, np.newaxis]
        rotated_normals = np.stack([-self.normals[:, 1], self.normals[:, 0]], axis=1)
        projections = self.origin + renormed_normals

        translated_projections = projections + rotated_normals
        denom = ((projections[:, 0] - translated_projections[:, 0]) *
                 np.roll(projections[:, 1] - translated_projections[:, 1], 1) -
                 (projections[:, 1] - translated_projections[:, 1]) *
                 np.roll(projections[:, 0] - translated_projections[:, 0], 1))
        num1 = (projections[:, 0] * translated_projections[:, 1] - projections[:, 1] * translated_projections[:, 0])
        num2 = (np.roll(projections[:, 0] * translated_projections[:, 1], 1) -
                np.roll(projections[:, 1] * translated_projections[:, 0], 1))
        x = ((num1 * np.roll(projections[:, 0] - translated_projections[:, 0], 1) -
              (projections[:, 0] - translated_projections[:, 0]) * num2) / denom)
        y = ((num1 * np.roll(projections[:, 1] - translated_projections[:, 1], 1) -
              (projections[:, 1] - translated_projections[:, 1]) * num2) / denom)
        self.vertices = np.stack([x, y], axis=1)
        self.polar_angles = mod2pi(self.theta + np.pi / 2)  # angles parametrizing the normals
        self._update_edges_and_angles()

    def update_vertices(self, new_vert):
        self.vertices = new_vert
        self._update_edges_and_angles()
        self.lengths = np.linalg.norm(self.edges, axis=1)  # length of each edge
        self.normals = np.stack([self.edges[:, 1], -self.edges[:, 0]], axis=1)
        self.normals /= np.linalg.norm(self.normals, axis=1)[:, np.newaxis]

        # distances of each edge to the origin
        self.phi = np.abs((np.roll(self.vertices[:, 0], -1, axis=0) - self.vertices[:, 0]) *
                          (self.vertices[:, 1] - self.origin[1]) -
                          (np.roll(self.vertices[:, 1], -1, axis=0) - self.vertices[:, 1]) *
                          (self.vertices[:, 0] - self.origin[0])) / self.lengths

        # compute polar angles of each edge with respect to the x-axis assuming origin is a vertex, in [-2, -2+2pi]
        self.polar_angles = mod2pi(np.arctan2(self.edges[:, 1], self.edges[:, 0]))

        self.theta = mod2pi(self.polar_angles - np.pi / 2)  # angles parametrizing the normals

    def is_in(self, points, min_dist=0):
        """Return an array of booleans indicating whether each point is inside the polygon."""
        # compute scalar products between each outward normal and the vector from the origin to the point
        return np.all(np.sum((np.reshape(points, (-1, 2)) - self.origin)[:, np.newaxis, :]*
                             self.normals[np.newaxis, :, :], axis=2)
                      <= self.phi[np.newaxis, :] - min_dist, axis=1)

    def plot(self):
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], 'x')
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], 'k')
        plt.plot([self.vertices[-1, 0], self.vertices[0, 0]], [self.vertices[-1, 1], self.vertices[0, 1]], 'k')
        plt.plot(self.origin[0], self.origin[1], 'o')
        projections = self.origin + self.phi[:, np.newaxis] * self.normals

        # add vertex numbers as text
        for i in range(self.nedges):
            plt.text(self.vertices[i, 0], self.vertices[i, 1], str(i))
            plt.text(projections[i, 0], projections[i, 1], str(i))

        plt.axis('equal')
        plt.show()


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


class ExtrudedPolygon(Polygon):
    def __init__(self, vertices, height):
        super().__init__(vertices)
        self.height = height
        vertices = np.zeros((2 * self.nedges, 3))
        vertices[:self.nedges, :2], vertices[self.nedges:, :2] = self.vertices.copy(), self.vertices.copy()
        vertices[self.nedges:, 2] = self.height

        normals = np.zeros((self.nedges + 2, 3))
        normals[:self.nedges, :2] = self.normals.copy()
        normals[self.nedges, 2], normals[self.nedges + 1, 2] = -1., 1.  # bottom and top normals

        self.vertices, self.normals = vertices, normals

    def sample(self, separation, distribution='uniform', min_dist=0.01, sample_type="separation",
               return_nodes=False, return_triangulation=False):  # return_nodes is for compatibility with Polygon
        if distribution != 'uniform':  # kept for compatibility with Polygon
            raise NotImplementedError("Only uniform distribution is implemented for extruded polygons.")
        if sample_type != 'separation':
            raise NotImplementedError("Only separation sampling is implemented for extruded polygons.")
        renormed_edges = self.edges / np.linalg.norm(self.edges, axis=1)[:, np.newaxis]

        boundary, bound_ids, normals, triangulations = [], [], [], []
        e3 = np.array([0, 0, 1], dtype=float)*(self.height - min_dist)
        for i in range(self.nedges):  # vertical faces
            points, triang = gmsh_sampling(np.stack([self.vertices[i] +
                                                     np.array([renormed_edges[i, 0]*min_dist,
                                                               renormed_edges[i, 1]*min_dist, min_dist]),
                                                     self.vertices[(i+1) % self.nedges] +
                                                     np.array([-renormed_edges[i, 0]*min_dist,
                                                               -renormed_edges[i, 1]*min_dist, min_dist]),
                                                     self.vertices[(i+1) % self.nedges] + e3, self.vertices[i] + e3],
                                                    axis=0),
                                           mesh_size=separation)

            boundary.append(points)
            bound_ids.append(np.full(points.shape[0], i))
            normals.append(np.full([points.shape[0], 3], self.normals[i]))
            triangulations.append(triang)
        # sample bottom and top : mesh the inside of a smaller polygon
        inside_poly = self.shift(-min_dist)
        points, triang = inside_poly.sample_inside(separation)
        boundary.append(np.concatenate([points, np.zeros([points.shape[0], 1])], axis=1))  # bottom
        boundary.append(boundary[-1].copy())  # top
        boundary[-1][:, 2] = self.height

        normals.append(np.full((points.shape[0], 3), [0., 0., -1.]))
        normals.append(np.full((points.shape[0], 3), [0., 0., 1.]))
        bound_ids.append(np.full(points.shape[0], self.nedges))
        bound_ids.append(np.full(points.shape[0], self.nedges + 1))
        triangulations.append(triang)
        triangulations.append(triangulations[-1].copy())
        res = (np.concatenate(boundary), np.concatenate(normals), np.concatenate(bound_ids))
        if return_nodes:
            res += (None,)

        if return_triangulation:
            res += (triangulations,)
        return res


def mod2pi(angle, lbound=-2.):
    """Return angle modulo 2*pi"""
    return (angle - lbound) % (2 * np.pi) + lbound


def gmsh_sampling(vertices, mesh_size):
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(vertices, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
    return mesh.points, mesh.cells_dict["triangle"]
