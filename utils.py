from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
import pygmsh
from scipy.integrate import simps
from scipy.spatial import ConvexHull
import json 

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


def in_polygon(points, vertices):
    poly = ShapelyPolygon(vertices)
    inside = np.zeros(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        pp = ShapelyPoint(p)
        inside[i] = poly.contains(pp)
    return inside 


def plot_boundary(bound, norm_vec, boundv, pos_src):
    plt.plot(bound[:, 0], bound[:, 1], '.-', label='boundary')
    plt.quiver(bound[:, 0], bound[:, 1], norm_vec[:, 0], norm_vec[:, 1])
    plt.plot(boundv[:, 0], boundv[:, 1], '.', label='virtual boundary')
    plt.plot(pos_src[:, 0], pos_src[:, 1], 'o', label='source')
    plt.axis('equal')
    plt.legend()
    plt.show()


class Polygon:
    def __init__(self, inp, origin=None, parametrization='vertices'):

        self.polar_angles, self.projs, self.distances = None, None, None
        self.vertices, self.nedges, self.lengths, self.normals = None, None, None, None

        if parametrization == "vertices":
            if origin is None:
                self.origin = np.mean(inp, axis=0)
                print('Polygon origin automatically set to {}'.format(self.origin))
            else:
                self.origin = origin
            self.update_vertices(inp)

        elif parametrization == 'half-planes':
            if origin is None:
                raise ValueError('Polygon must be initialized with an origin if given as half-plane parametrization')
            self.origin = origin
            self.update_param(inp)

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

    def get_params(self, parametrization):
        if parametrization == "vertices":
            return self.vertices 
        elif parametrization == "half-planes":
            return self.projs 
        else:
            raise ValueError("unknown parametrization, accepted values:, 'half-planes', 'vertices'")

    def compute_angles(self, points):
        """Compute polar angles of points with respect to the vertices of the polygon, using the next edge (in
        trigonometric order) as polar axis. Return angles and vectors from vertices to points."""
        diffs = points[:, np.newaxis, :2] - self.vertices[np.newaxis, :self.nedges, :2]  # shape (npoints, nvertices, 2)
        norms = np.linalg.norm(diffs, axis=2)

        # shape (npoints, nvertices)
        angles = mod2pi(np.arctan2(np.abs(diffs[:, :, 0]*self.edges[:, 1] - diffs[:, :, 1]*self.edges[:, 0]),
                                   diffs[:, :, 0]*self.edges[:, 0] + diffs[:, :, 1]*self.edges[:, 1]))

        return angles, diffs, norms

    def sample_inside(self, sep):
        points, triang = gmsh_sampling(self.vertices[:self.nedges, :], mesh_size=sep)
        return points[:, :2], triang

    def _update_edges_and_angles(self):
        self.edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.renormed_edges = self.edges / np.linalg.norm(self.edges, axis=1)[:, np.newaxis]
        self.nedges = len(self.edges)
        self.lengths = np.linalg.norm(self.edges, axis=1)  # length of each edge
        renormed_edges = self.edges / self.lengths[:, np.newaxis]
        # compute inner angles
        dotprod = np.sum(-np.roll(renormed_edges, 1, axis=0) * renormed_edges, axis=1)
        sin_sign = -np.sign(np.cross(-np.roll(renormed_edges, 1, axis=0), renormed_edges))
        self.angles = np.arccos(np.clip(dotprod, -1, 1))
        self.angles = mod2pi(np.where(sin_sign < 0, 2 * np.pi - self.angles, self.angles))
        self.translated_vertices = self.vertices - self.origin
        # compute polar angles of each edge with respect to the x-axis assuming origin is a vertex, in [-2, -2+2pi]
        self.polar_angles = mod2pi(np.arctan2(self.edges[:, 1], self.edges[:, 0]))

    def update_param(self, new_projs, new_origin=None, bounds=None, check_in=None, min_length=None):
        if new_origin is not None:  # otherwise assume old origin is still in the polygon
            self.origin = new_origin

        self.projs = new_projs
        self.distances = np.linalg.norm(new_projs - self.origin, axis=1)
        self.nedges = len(self.projs)
        self.normals = new_projs - self.origin
        self.normals /= np.linalg.norm(self.normals, axis=1)[:, np.newaxis]

        # compute every possible intersections between the lines defined by the normals
        intersections = []
        for i in range(self.nedges):
            tmp = compute_intersections(new_projs[i], self.normals[i], new_projs[i+1:], self.normals[i+1:])
            intersections.append(tmp[np.isfinite(tmp[:, 0])])

        intersections = np.concatenate(intersections, axis=0)

        self.vertices = intersections[self.is_in(intersections, tol=1e-6)]
        self.distances = np.linalg.norm(self.vertices - self.origin, axis=1)
        if len(self.vertices) > self.nedges:
            raise ValueError("The number of vertices should not exceed the number of half-planes")
        elif len(self.vertices) < self.nedges:
            print("Warning: some half-planes do not intersect, {} < {} edges found.".format(len(self.vertices),
                                                                                            self.nedges))

        success = self.update_vertices(self.vertices, sort=True, min_length=min_length, bounds=bounds)  # to ensure indexation is correct

        if success and (check_in is not None):  # check if the points are in (after ensuring correct update of vertices) 
            if not np.all(self.is_in(check_in)):
                return False 

        return success
        
    def sort_vertices(self):
        """Sort vertices in trigonometric order"""
        angles = mod2pi(np.arctan2(self.vertices[:, 1] - self.origin[1], self.vertices[:, 0] - self.origin[0]))
        self.vertices = self.vertices[np.argsort(angles)]

    def update_vertices(self, new_vert, sort=False, bounds=None, check_in=None, convex_proj=False, update_origin=False, min_length=None):
        if bounds is not None:
            if not np.all((bounds[0] < new_vert) & (new_vert < bounds[1])):
                return False 
        if convex_proj:
            new_vert = new_vert[ConvexHull(new_vert).vertices]

        self.vertices = new_vert.copy()

        if min_length is not None: # merge vertices
            while np.min(self.lengths) < min_length:
                self._update_edges_and_angles()

                indmin = np.argmin(self.lengths)
                v1, v2 = self.vertices[indmin], self.vertices[(indmin + 1) % self.nedges]
                vmid = (v1 + v2) / 2.
                self.vertices[indmin] = vmid 
                self.vertices = np.concatenate([self.vertices[:indmin+1], self.vertices[indmin+2:]], axis=0)
        
        if sort:
            self.sort_vertices()
        if len(self.vertices) < 3:
            print("Number of vertices should exceed 2")
            return False

        if update_origin:
            self.origin = np.mean(self.vertices, axis=0)

        self._update_edges_and_angles()
        
        self.normals = np.stack([self.edges[:, 1], -self.edges[:, 0]], axis=1)
        self.normals /= np.linalg.norm(self.normals, axis=1)[:, np.newaxis]

        # distances of each edge to the origin
        self.distances = np.abs((np.roll(self.vertices[:, 0], -1, axis=0) - self.vertices[:, 0]) *
                                (self.vertices[:, 1] - self.origin[1]) -
                                (np.roll(self.vertices[:, 1], -1, axis=0) - self.vertices[:, 1]) *
                                (self.vertices[:, 0] - self.origin[0])) / self.lengths
        self.projs = self.origin + self.normals * self.distances[:, np.newaxis]

        if check_in is not None:
            if not np.all(self.is_in(check_in)):
                return False 
       
        return True

    def get_edge_distances(self, points):
        return np.sum((np.reshape(points, (-1, 2)) - self.origin)[:, np.newaxis, :] *
                      self.normals[np.newaxis, :, :], axis=2)

    def is_in(self, points, min_dist=0., tol=1e-12):
        """Return an array of booleans indicating whether each point is inside the (convex) polygon."""
        # compute scalar products between each outward normal and the vector from the origin to the point
        return np.all(self.get_edge_distances(points)
                      <= self.distances[np.newaxis, :] - min_dist + tol, axis=1)

    def random_point_in_poly(self, npoints, min_dist=0):
        minx, miny, maxx, maxy = np.min(self.vertices[:,0]), np.min(self.vertices[:,1]), np.max(self.vertices[:,0]), np.max(self.vertices[:,1])
        points = []
        while len(points) < npoints:
            pnt = np.array([[np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)]])
            if self.is_in(pnt, min_dist=min_dist)[0]:
                points.append(pnt)
        return np.concatenate(points, axis=0)

    def integrate_edge(self, vals, quad_nodes, edge_id):
        """Integrate the values vals over the edge with id edge_id, using the quadrature nodes quad_nodes and the
        simpson rule."""
        return simps(vals, quad_nodes, axis=0) * self.lengths[edge_id]

    def integrate(self, vals, quad_nodes, quad_ids):
        """Integrate the values vals over the polygon, using the quadrature nodes quad_nodes and the simpson rule."""
        res = 0.
        for i in range(self.nedges):
            mask = quad_ids == i
            res += self.integrate_edge(vals[mask], quad_nodes[mask], i)
        return res

    def convex_projection(self, update_origin=False):
        """Update the polygon to its convex hull."""
        self.update_vertices(self.vertices[ConvexHull(self.vertices).vertices], update_origin=update_origin)

    def plot(self):
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], 'x')
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], 'k')
        plt.plot([self.vertices[-1, 0], self.vertices[0, 0]], [self.vertices[-1, 1], self.vertices[0, 1]], 'k')
        plt.plot(self.origin[0], self.origin[1], 'o')
        projections = self.projs

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
        e3 = np.array([0, 0, 1], dtype=float)*(self.height - 2*min_dist)
        for i in range(self.nedges):  # vertical faces
            shifted1 = self.vertices[i] + np.array([renormed_edges[i, 0] * min_dist,
                                                    renormed_edges[i, 1] * min_dist, min_dist])
            shifted2 = self.vertices[(i + 1) % self.nedges] + np.array([-renormed_edges[i, 0] * min_dist,
                                                                        -renormed_edges[i, 1] * min_dist, min_dist])
            points, triang = gmsh_sampling(np.stack([shifted1, shifted2, shifted2 + e3, shifted1 + e3], axis=0),
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


class Cuboid(Polygon):
    def __init__(self, lx, ly, lz):
        self.lx, self.ly, self.lz = lx, ly, lz

        vertices = np.array([[0., 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
                             [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]])

        super().__init__(vertices[:4, :2])

        self.vertices = vertices
        normals = np.zeros((6, 3))
        normals[:4, :2] = self.normals.copy()
        normals[4, 2], normals[5, 2] = -1., 1.  # bottom and top normals
        self.normals = normals
        # self.normals = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.nedges = 4  # for compatibility with ExtrudedPolygon

    def sample(self, param, distribution='chebyshev', min_dist=0.01, sample_type="total", return_nodes=False):
        # param is a 3-tuple giving the number of points in each direction

        if sample_type == "separation":
            # compute the corresponding number of points for each direction
            param = (int(np.ceil(self.lx / param)),
                     int(np.ceil(self.ly / param)),
                     int(np.ceil(self.lz / param)))
        nodesx = chebyshev_nodes(0, self.lx, param[0])
        nodesy = chebyshev_nodes(0, self.ly, param[1])
        nodesz = chebyshev_nodes(0, self.lz, param[2])

        # first vertical face
        nodesxy = np.stack(np.meshgrid(nodesx, nodesy), axis=-1).reshape(-1, 2)
        nodesyz = np.stack(np.meshgrid(nodesy, nodesz), axis=-1).reshape(-1, 2)
        nodesxz = np.stack(np.meshgrid(nodesx, nodesz), axis=-1).reshape(-1, 2)

        boundary = [np.zeros([len(nodesxz), 3]), np.zeros([len(nodesyz), 3]), np.zeros([len(nodesxz), 3]),
                    np.zeros([len(nodesyz), 3]), np.zeros([len(nodesxy), 3]), np.zeros([len(nodesxy), 3])]
        boundary[0][:, 1], boundary[0][:, 0], boundary[0][:, 2] = 0, nodesxz[:, 0], nodesxz[:, 1]
        boundary[1][:, 0], boundary[1][:, 1], boundary[1][:, 2] = self.lx, nodesyz[:, 0], nodesyz[:, 1]
        boundary[2][:, 1], boundary[2][:, 0], boundary[2][:, 2] = self.ly, nodesxz[:, 0], nodesxz[:, 1]
        boundary[3][:, 0], boundary[3][:, 1], boundary[3][:, 2] = 0, nodesyz[:, 0], nodesyz[:, 1]
        boundary[4][:, :2], boundary[4][:, 2] = nodesxy, 0
        boundary[5][:, :2], boundary[5][:, 2] = nodesxy, self.lz

        bound_ids = [np.full(len(boundary[i]), i) for i in range(6)]
        normals = [np.zeros([len(boundary[i]), 3]) for i in range(6)]
        normals[0][:, 1], normals[2][:, 1] = -1, 1
        normals[1][:, 0], normals[3][:, 0] = 1, -1
        normals[4][:, 2], normals[5][:, 2] = -1, 1
        bound_ids, boundary, normals = np.concatenate(bound_ids), np.concatenate(boundary), np.concatenate(normals)

        return boundary, normals, bound_ids

    def is_in(self, points, min_dist=0., tol=1e-6):
        return ((-tol + min_dist <= points[:, 0]) & (self.lx - min_dist + tol >= points[:, 0]) &
                (-tol + min_dist <= points[:, 1]) & (self.ly - min_dist + tol >= points[:, 1]) &
                (-tol + min_dist <= points[:, 2]) & (self.lz - min_dist + tol >= points[:, 2]))


def mod2pi(angle):
    """Return angle modulo 2*pi"""
    return np.where(angle < 0, angle + 2*np.pi, angle)


def angle_distance(angle1, angle2):
    """Return the distance between two angles in [0, 2*pi]"""
    return np.minimum(np.abs(angle1 - angle2), 2*np.pi - np.abs(angle1 - angle2))


def gmsh_sampling(vertices, mesh_size):
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(vertices, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
    return mesh.points, mesh.cells_dict["triangle"]


def compute_intersections(p1, vector1, p2, vectors2, tol=1e-6):
    """Compute the intersection between a line defined by a point and a vector and a set of lines defined by points
    and vectors. Return the intersection points."""
    dotprod1 = np.sum(p1*vector1)
    dotprods2 = np.sum(p2*vectors2, axis=1)  # shape (n, 2)
    crossprod = vector1[0]*vectors2[:, 1] - vector1[1]*vectors2[:, 0]  # shape (n,), determinant of the matrix
    indnan = np.abs(crossprod) < tol
    indfinite = np.logical_not(indnan)
    crossprod = np.where(indnan, np.nan, crossprod)
    res = np.zeros([len(p2), 2])
    res[indnan] = np.nan
    res[indfinite, 0] = (dotprod1*vectors2[indfinite, 1] - dotprods2[indfinite]*vector1[1]) / crossprod[indfinite]
    res[indfinite, 1] = (-dotprod1*vectors2[indfinite, 0] + dotprods2[indfinite]*vector1[0]) / crossprod[indfinite]
    return res


def rotate(points, angle, center):
    """Rotate points by an angle around a center. Points and center are numpy arrays."""
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (points - center) @ rot + center


def random_poly(bounds, nvertices, in_constraints=None, min_length=None):
    if type(bounds) == float:
        gen_fun = lambda: random_point_disk(bounds, nvertices)
    else:
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        gen_fun = lambda: np.random.rand(nvertices, 2)*np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])  

    while True:
        vertices = gen_fun()
        vertices = vertices[ConvexHull(vertices).vertices]
        if len(vertices) == nvertices:
            poly = Polygon(vertices)
            if min_length is not None and np.min(poly.lengths) < min_length:
                continue
            if in_constraints is not None:
                if np.all(poly.is_in(in_constraints)):
                    return poly
            else:
                return poly


def random_point_disk(r, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, n)
    r = np.sqrt(np.random.uniform(0, r**2, n))
    return np.stack([r*np.cos(theta), r*np.sin(theta)], axis=1)


def array_to_list(arr):
    """Converts arr to a list if arr is an ndarray, leave as it is otherwise."""
    if type(arr) == np.ndarray:
        return arr.tolist()
    else:
        return arr


def dict_to_json(dico, path):
    dd = dico.copy()
    for key in dd:
        dd[key] = array_to_list(dd[key])
    fd = open(path, 'w')
    json.dump(dd, fd)
    fd.close()
