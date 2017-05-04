import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import matplotlib.path
import uuid
from collections import deque
import itertools
import materials

# Define global coordinate system unit vectors
xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])
unit_vecs = np.array([xhat, yhat, zhat])

def get_local_plane_coordinate_system(vector):
    """
    Define a coordinate system with the z-axis oriented in the direction of the normal vector,
    and x,y oriented on the orthogonal plane.
    The x and y directions are assigned in a deterministic way.
    """
    min_index = min(enumerate(vector), key=lambda x: abs(x[1]))[0]
    y_dir = unit_vecs[min_index] - vector*np.dot(unit_vecs[min_index], vector)
    y_dir = y_dir / np.linalg.norm(y_dir)
    x_dir = np.cross(y_dir, vector)
    x_dir = x_dir / np.linalg.norm(x_dir)
    local_unit_vecs = np.array([x_dir, y_dir, vector])
    return local_unit_vecs

class Plane:
    """
    Class representing a Euclidean plane
    """
    def __init__(self, point, vector):
        self.point = point
        self.vector = vector / np.linalg.norm(vector)
        self.unit_vecs = get_local_plane_coordinate_system(self.vector)
        self.p0 = self.point + self.unit_vecs[0]
        self.p1 = self.point + self.unit_vecs[1]
        self.p2 = self.point

    def __eq__(self, other):
        if (self.point != other.point).any():
            return False
        if (self.vector != other.vector).any():
            return False
        return True

    def distance(self, p):
        """
        Compute the distance beteen a point and the plane
        """
        t = np.dot(self.vector, (self.point - p))
        return t

    def project(self, p):
        """
        Project a point onto the plane
        """
        return p + self.vector * self.distance(p)

    def ray_intersection(self, ray):
        new_int = self.new_ray_intersection(ray)
        old_int = self.old_ray_intersection(ray)
        if np.any(new_int != old_int) and np.any(abs(new_int - old_int)/abs(new_int) > 1e-5):
            print 'new_int', new_int
            print 'old_int', old_int
            raise ValueError('Ray intersection does not match')
        return new_int

    def new_ray_intersection(self, ray):
        M = np.array([
                [-ray.d[0], self.unit_vecs[0][0], self.unit_vecs[1][0]],
                [-ray.d[1], self.unit_vecs[0][1], self.unit_vecs[1][1]],
                [-ray.d[2], self.unit_vecs[0][2], self.unit_vecs[1][2]],
            ])
        if np.linalg.det(M) == 0:
            return None
        vv = ray.p - self.point
        t, u, v = np.dot(np.linalg.inv(M), vv)
        return ray.p + ray.d*t

    def old_ray_intersection(self, ray):
        """
        Compute the intersection point of a ray with the plane
        """
        # p = v*t+p0
        # f(x) = n*(x - x0) = 0
        # f(v*t+p0) = n*(v*t+p0 - x0) = 0
        # (n*v)*t + n*(p0 - x0) = 0
        # t = n*(x0 - p0)/(n*v)
        # pint = v*(n*(x0 - p0)/(n*v)) + p0
        # Find the intersection of voxel edges with the plane
        n = self.vector
        x0 = self.point
        v = ray.d
        p0 = ray.p
        ndv = np.dot(n, v)
        if ndv == 0:
            return None
        return v*(np.dot(n, x0-p0)/ndv) + p0

def fit_plane(points):
    """
    Compute the best fit plane to a set of pointd in 3D
    """
    center = sum(points) / len(points)
    x = np.array(points - center)
    M = np.dot(np.transpose(x),x)
    normal = np.linalg.svd(M)[0][:,-1]
    return center, normal

def is_left(p0, p1, p2, normal=None):
    """
    Input: three points p0, p1, and p2
    Return: >0 for p2 left of the line through p0 to p1
            =0 for p2 on the line
            <0 for p2 right of the line
    """
    if normal is None:
        center, normal = fit_plane(np.array([p0, p1, p2]))
    return np.dot(np.cross(p1 - p0, p2 - p1), normal)

class Edge:
    """
    Class representing the edge of a polygon
    """
    def __init__(self, p0, p1):
        p0 = list(p0)
        p1 = list(p1)
        points = sorted([p0, p1])
        self.p0 = np.array(points[0])
        self.p1 = np.array(points[1])

    def points(self):
        return np.array([self.p0, self.p1])
    def __eq__(self, other):
        if (self.p0 != other.p0).any():
            return False
        if (self.p1 != other.p1).any():
            return False
        return True
    def __repr__(self):
        return '['+repr(self.p0)+', '+repr(self.p1)+']'
    def __hash__(self):
        return hash(repr(self.p0) + repr(self.p1))

class Ray:
    """
    Class representing a ray
    """
    def __init__(self, p, d):
        self.p = p
        self.d = d / np.linalg.norm(d)
    def pca_to_point(self, point):
        """
        Compute the ray's closest approach to a point
        """
        return self.p + self.d*np.dot(point - self.p, self.d)

class Polygon:
    """
    Class representing a polygon
    """
    def __init__(self, plane, points, strict=True):
        self.plane = plane
        self.points = points
        self.edges = []
        # Check that the points lie on the plane to within some tolerance
        for i, p in enumerate(self.points):
            t = self.plane.distance(p)
            if abs(t) >= 1e-5:
                if strict:
                    raise ValueError('Point not on plane')
                print 'Projecting point', p, 'onto plane'
                self.points[i] = self.plane.project()
        for i in xrange(len(self.points)):
            self.edges.append(Edge(self.points[i], self.points[(i+1) % len(self.points)]))

        local_points = [np.dot(self.plane.unit_vecs[:2],p) for p in self.points]
        self.path = matplotlib.path.Path(local_points)

    def get_edges(self):
        return self.edges

    def winding_number(self, point):
        """
        Compute the winding number of a point with respect to the polygon
        The winding number is representative of the number of times the polygon edge wraps
            counterclockwise around a point
        Can be used to check if a point is contained in a polygon
        """
        point = point - self.plane.point
        ray = self.plane.unit_vecs[0]
        ray_perp = self.plane.unit_vecs[1]
        wn = 0
        for i in xrange(len(self.points)):
            v0 = self.points[i] - self.plane.point
            v1 = self.points[(i+1)%len(self.points)] - self.plane.point
            vertical = np.dot(v1-v0, ray) == 0
            horizontal = np.dot(v1-v0, ray_perp) == 0
            isLeft = is_left(v0, v1, point, normal=self.plane.vector)
            if horizontal:
                if isLeft == 0:
                    return int(np.dot(v1 - v0, ray) > 0)
                continue
            v0_perp = np.dot((point-v0),ray_perp)
            v1_perp = np.dot((point-v1),ray_perp)

            if isLeft > 0:
                if v0_perp >= 0 and v1_perp < 0:
                    wn += 1
            elif isLeft < 0:
                if v0_perp < 0 and v1_perp >= 0:
                    wn -= 1
            elif isLeft == 0:
                return int(np.dot(v1 - v0, ray_perp) < 0)
        return wn

    def contains(self, point):
        """
        Compute whether a point is inside a polygon or not
        A non zero winding number implies that the point is contained
        """
        wn_contains = self.winding_number(point) != 0
        #numpy_contains = self.path.contains_point(np.dot(self.plane.unit_vecs,point)[:2])
        
        #if wn_contains != numpy_contains:
        #    print 'WN:', wn_contains
        #    print 'NP:', numpy_contains
        #    print 'points', self.points
        #    print 'point', point
        #    print 'path', self.path
        #    print 'proj point', np.dot(self.plane.unit_vecs, point)[:2]
        #    raise ValueError('Bad winding number result')
        return wn_contains

    def ray_intersection(self, ray):
        """
        Compute the intersection of a ray with the polygon if it exists
        """
        pint = self.plane.ray_intersection(ray)
        if pint is not None and self.contains(pint):
            return pint
        else:
            return None

class Surface:
    """
    Class representing a surface
    """
    def __init__(self, polygons):
        self.polygons = polygons
        raw_edges = [e for p in self.polygons for e in p.get_edges()]
        edge_counts = Counter(raw_edges)
        naked_edges = []
        point_dict = dict()
        # Check that each edge is shared by only two polygons
        for e, c in edge_counts.items():
            if c == 1:
                for p in e.points():
                    if repr(p) not in point_dict:
                        point_dict[repr(p)] = []
                    point_dict[repr(p)].append(e)
                naked_edges.append(e)
            elif c > 2:
                raise ValueError('Edge shared by too many polygons')
        # If some edges are not shared by more than one polygon
        # the surface is considered to be open.
        # Attempt to construct a single polygon to close the surface.
        if len(naked_edges) > 0:
            # Require that each point belongs to two lines
            for p, edges in point_dict.items():
                if len(edges) != 2:
                    raise ValueError('Polygon not well connected')
            last_edge = naked_edges[0]
            last_point = last_edge.points()[0]
            ordered_edges = [last_edge]
            ordered_points = [last_point]
            # Sort the edges end to end
            while True:
                last_edge = ordered_edges[-1]
                last_point = ordered_points[-1]
                edges = point_dict[repr(last_point)]
                new_edge = [e for e in edges if e != last_edge][0]
                ordered_edges.append(new_edge)
                new_point = [p for p in new_edge.points() if (p != last_point).any()][0]
                if (new_point == ordered_points[0]).all():
                    break
                ordered_points.append(new_point)
            # Construct and add te new polygon
            center = sum(ordered_points) / len(ordered_points)
            normal = np.linalg.svd(np.transpose(np.array(ordered_points - center)))[0][-1]
            new_plane = Plane(center, normal)
            new_polygon = Polygon(new_plane, ordered_points)
            self.polygons.append(new_polygon)

class Voxel:
    """
    Class representing a voxel
    """
    def __init__(self, coord, scale, vol=False):
        # The vol specifies that the voxel is not a part of a grid of voxels
        self.vol = vol
        if self.vol:
            self.center = coord
        else:
            self.coord = coord

        # Set the size of the voxel
        if not hasattr(scale, '__len__'):
            scale = [scale, scale, scale]
        self.scale = np.array(scale)
        self.reinit()

    def reinit(self):
        """
        Re-initializes relevant quantities after the voxel coordinates/scale have been changed
        """
        self.half_scale = self.scale/2.0
        if not self.vol:
            self.center = self.coord * self.scale
        self.min_planes = self.center - self.half_scale
        self.max_planes = self.center + self.half_scale
        self.planes = [self.min_planes, self.max_planes]

    def contains_point(self, point, include_boundaries=True, tol=1e-6):
        """
        Returns true if point is inside voxel or on boundary
        """
        if include_boundaries:
            return np.all(point+tol >= self.min_planes) and np.all(point-tol <= self.max_planes)
        else:
            return np.all(point > self.min_planes) and np.all(point < self.max_planes)

    def get_corners(self):
        """
        Return the coordinates of the voxel coorners
        """
        # I'm pretty sure this doesn't work because of some bit shifting nastiness
        # Caused infinite loops when I tries to use the ouput
        corners = []
        for i in xrange(0, 8):
            corner = np.zeros(3)
            for d in [0, 1, 2]:
                corner[d] = self.planes[i >> d & 0x1][d]
            corners.append(corners)
        return corners

    def get_faces(self):
        """
        Return polygon objects that represent the rectangular faces of the voxel
        """
        faces = []
        for d in xrange(3):
            for m in [0, 1]:
                corner = np.zeros(3)
                corner[d] = self.planes[m][d]
                vector = np.zeros(3)
                vector[d] = -1+2*m
                corners = []
                for bit_mask in [0,1,3,2]:
                    xd = (d+1)%3
                    xm = (bit_mask >> 0) & 0x1
                    corner[xd] = self.planes[xm][xd]
                    yd = (d+2)%3
                    ym = (bit_mask >> 1) & 0x1
                    corner[yd] = self.planes[ym][yd]
                    corners.append(corner.copy())
                center = sum(corners)/4.
                plane = Plane(center, vector)
                face = Polygon(plane, corners)
                faces.append(face)
        return faces

    def in_voxel(self, voxel, count_edge_contact=False):
        """
        Check if this voxel is inside another voxel
        """
        corner = np.zeros(3)
        n_inside = 0
        for corner in self.get_corners():
            if voxel.contains_point(corner, include_boundaries=count_edge_contact):
                return True
            if not count_edge_contact:
                if voxel.contains_point(corner, include_boundaries=True):
                    n_inside += 1
        if n_inside == 8:
            return True
        return False

    def plane_t(self, dim, ray, minmax=0):
        """
        Compute the distance between a ray and one of the six planes
        dim: specifies the dimension of the plane's normal vector
        minmax: specifies whether it is the minimum or maximum plane in that dimension
        """
        p = ray.p
        d = ray.d
        if (d[dim] == 0):
            t = np.inf
        elif minmax:
            t = (self.max_planes[dim] - p[dim])/d[dim]
        else:
            t = (self.min_planes[dim] - p[dim])/d[dim]
        return t

    def exit(self, ray):
        """
        Compute the distance to a ray's exit from the voxel and the plane through which it will exit
        """
        t_infos = []
        for d in xrange(3):
            t0 = self.plane_t(d, ray, minmax=0)
            t1 = self.plane_t(d, ray, minmax=1)
            minmax, t = max(enumerate([t0, t1]), key=lambda x: x[1])
            t_infos.append((t, d, minmax))
        t, d, minmax = min(t_infos, key=lambda x: x[0])
        return (unit_vecs[d] * (minmax*2 - 1), t)

    def entry(self, ray):
        """
        Compute the distance to a ray's entry to the voxel and the plane through which it will enter
        """
        t_infos = []
        for d in xrange(3):
            t0 = self.plane_t(d, ray, minmax=0)
            t1 = self.plane_t(d, ray, minmax=1)
            minmax, t = min(enumerate([t0, t1]), key=lambda x: x[1])
            t_infos.append((t, d, minmax))
        t, d, minmax = min(t_infos, key=lambda x: x[0])
        return (unit_vecs[d] * (minmax*2 - 1), t)

    def exit_plane(self, ray):
        return self.exit(ray)[0]

    def entry_plane(self, ray):
        return self.entry(ray)[0]

    def exit_t(self, ray):
        return self.exit(ray)[1]

    def entry_t(self, ray):
        return self.entry(ray)[1]

    def next_voxel(self, ray):
        """
        Re-initialize voxel as the next voxel a ray will pass through
        """
        plane = self.exit_plane(ray)
        self.coord += plane
        self.reinit()

    def ray_intersects(self, ray):
        """
        Check if a ray intersects the voxel
        """
        t = self.entry_t(ray)
        p = ray.p + ray.d*t
        return self.contains_point(p, include_boundaries=True, tol=1e-6)

    def poly_intersects(self, polygon):
        """
        Check if a polygon intersects with the voxel
        """
        # Return True is any polygon points are in the voxel
        for p in polygon.points:
            if self.contains_point(p):
                return True

        # If the projection of the voxel center onto the plane of the polygon is not inside the voxel,
        # then no point on the polygon is inside the voxel
        if not self.contains_point(polygon.plane.project(self.center)):
            return False

        n = polygon.plane.vector
        x0 = polygon.plane.point

        # p = v*t+p0
        # f(x) = n*(x - x0) = 0
        # f(v*t+p0) = n*(v*t+p0 - x0) = 0
        # (n*v)*t + n*(p0 - x0) = 0
        # t = n*(x0 - p0)/(n*v)
        # pint = v*(n*(x0 - p0)/(n*v)) + p0
        # Find the intersection of voxel edges with the plane
        for d in [0, 1, 2]: # d = unit direction of edge
            v = unit_vecs[d]
            d0 = (d + 1) % 3 # other dimention 0
            d1 = (d + 2) % 3 # other dimension 1
            for dm0, dm1 in [(0,0), (0,1), (1,0), (1,1)]:
                db0 = self.planes[dm0][d0] # coordinate 0 of edge
                db1 = self.planes[dm1][d1] # coordinate 1 of edge
                p0 = self.center.copy() # Point on edge
                p0[d0] = db0
                p0[d1] = db1
                ndv = np.dot(n, v)
                if ndv == 0: # Don't want to divide by zero
                    continue
                pint = v*(np.dot(n, x0-p0)/np.dot(n, v)) + p0 # compute the intersection point
                if np.linalg.norm(pint - p0) <= self.half_scale[d]: # check if the point is in the voxel
                    if polygon.contains(pint): # check if the point is in the polygon
                        return True
        return False

class PolyNode:
    """
    Class to store information about a polygon,
    and the surface it belongs to in the spacial data structure
    """
    def __init__(self, poly, geo_node):
        self.poly = poly
        self.geo_node = geo_node
        self.uid = uuid.uuid4()

class GeometryNode:
    """
    Class for hierarchically organizing surfaces in the geomtry
    as well as specifying the properties of the volume within that surface
    """
    def __init__(self, surface, properties=None, name='', parent=None):
        self.name = name
        self.surface = surface
        self.properties = properties
        self.uid = uuid.uuid4()
        self.parent = parent
        if parent is not None:
            parent.add_child(self)
        self.children = []
        self.level = 0
    def add_child(self, c):
        c.parent = self
        self.children.append(c)

class Geometry:
    """
    Class representing the geometry of the simulation environment
    Organizes polygons of surfaces into spacial data structure for fast ray tracing
    Performs ray tracing
    Determines where points in the volume are (what voxel, what surface)
    """
    def __init__(self, root_nodes, voxel_scale=2, default_properties=materials.get_properties(7)):
        self.root_nodes = root_nodes
        self.default_properties = default_properties # default properties are those of nitrogen
        self.default_properties['root'] = True
        self.nodes = []

        # Traverse geometry nodes to collect polygons
        point_dict = dict()
        self.poly_nodes = []
        traversed = set()
        untraversed = deque()
        untraversed.extend(root_nodes)
        while len(untraversed) > 0:
            next_node = untraversed.popleft()
            self.nodes.append(next_node)
            for child in next_node.children:
                if child.uid in traversed:
                    pass
                else:
                    child.level = next_node.level + 1
                    untraversed.append(child)
                    traversed.add(child.uid)
            for poly in next_node.surface.polygons:
                poly_node = PolyNode(poly, next_node)
                self.poly_nodes.append(poly_node)
                for p in poly.points:
                    point_dict[repr(p)] = p

        # Compute the bounds of the simulation volume
        points = np.array([p for p in point_dict.values()])
        min_x, max_x = min(points[:,0]), max(points[:,0])
        min_y, max_y = min(points[:,1]), max(points[:,1])
        min_z, max_z = min(points[:,2]), max(points[:,2])
        min_planes = np.array([min_x, min_y, min_z])
        max_planes = np.array([max_x, max_y, max_z])
        min_v = np.floor(min_planes/voxel_scale)
        max_v = np.ceil(max_planes/voxel_scale)
        lspaces = []
        for i in [0, 1, 2]:
            lspaces.append((min_v[i], max_v[i], max_v[i]-min_v[i]+1))

        # Construct the volume voxel and store boundaries as polygons
        volume_coord = (max_v + min_v) / 2.0 * voxel_scale
        volume_scale = (max_v - min_v+1) * voxel_scale
        self.volume = Voxel(volume_coord, volume_scale, vol=True)
        volume_faces = self.volume.get_faces()
        volume_surface = Surface(volume_faces)
        volume_node = GeometryNode(volume_surface, self.default_properties)
        volume_node.level = -1
        volume_pnodes = [PolyNode(p, volume_node) for p in volume_faces]
        for node in self.root_nodes:
            volume_node.add_child(node)
        self.poly_nodes.extend(volume_pnodes)

        # Iterate over all voxels and store polygons that itersect appropriately in the spatial data structure
        self.voxel_dict = dict()
        grid = np.meshgrid(np.linspace(*lspaces[0]), np.linspace(*lspaces[1]), np.linspace(*lspaces[2]))
        voxel_coords = (np.array([x,y,z]) for x, y, z in itertools.izip(grid[0].flatten(), grid[1].flatten(), grid[2].flatten()))
        voxel = Voxel(np.zeros(3), voxel_scale)
        self.voxel_scale = voxel.scale
        mx = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]
        for vc in voxel_coords:
            k = repr(vc)
            self.voxel_dict[k] = []
            voxel.coord = vc
            voxel.reinit()
            for i in xrange(3):
                mx[i][0] = min(mx[i][0], vc[i])
                mx[i][1] = max(mx[i][1], vc[i])
            for pn in self.poly_nodes:
                if voxel.poly_intersects(pn.poly):
                    self.voxel_dict[k].append(pn)

    def get_polygon_nodes(self, voxel):
        if np.all(voxel.scale == self.voxel_scale):
            return self.voxel_dict[repr(voxel.coord)]
        else:
            raise ValueError("Voxel scale mismatch")

    def get_containing_voxel(self, point):
        voxel_coord = np.floor(point/self.voxel_scale + 0.5)
        return voxel_coord

    def get_t_until_next_voxel(self, ray):
        current_voxel = self.get_containing_voxel(ray.p)
        voxel = Voxel(current_voxel, self.voxel_scale)
        return voxel.exit_t(ray)

    def ray_trace(self, ray, all_intersections=False):
        checked = dict()
        intersections = dict()
        for poly_node in self.poly_nodes:
            if poly_node.uid in checked:
                continue
            pint = poly_node.poly.ray_intersection(ray)
            if pint is not None:
                intersections[poly_node.uid] = (poly_node, pint)
            checked[poly_node.uid] = poly_node
        
        intersections = [(pn, pint, np.dot(pint - ray.p, ray.d)) for pn, pint in intersections.values()]
        intersections = [(pn, pint, d) for pn, pint, d in intersections if d>=0]
        intersections = sorted(intersections, key=lambda x: x[2])
        intersections = [(pn, pint) for pn, pint, d in intersections]
        return intersections

    def bad_ray_trace(self, ray, all_intersections=False):
        """
        Compute the intersections of a ray with polygons in the volume
        Uses information about which polygons intersect with a voxel
            to reduce computation time
        """
        intersections = dict()
        checked = dict()
        # Check if the ray is already in the volume
        if self.volume.contains_point(ray.p):
            # Find the voxel containing the ray
            voxel_coord = np.floor(ray.p/self.voxel_scale + 0.5)
            voxel = Voxel(voxel_coord, self.voxel_scale)
            vxid = repr(voxel.coord) in self.voxel_dict
            if not vxid:
                voxel.next_voxel(ray)
                if not repr(voxel.coord) in self.voxel_dict:
                    return []
        else:
            # Find the intersection with the volume
            if not self.volume.ray_intersects(ray):
                return []
            entry_t = self.volume.entry_t(ray)
            if entry_t is None:
                voxel = None
            else:
                pint = ray.p+ray.d*entry_t
                voxel_coord = np.floor(pint/self.voxel_scale + 0.5)
                voxel = Voxel(voxel_coord, self.voxel_scale)
                vxid = repr(voxel.coord) in self.voxel_dict
                if not vxid:
                    voxel.next_voxel(ray)
                    if not repr(voxel.coord) in self.voxel_dict:
                        return []

        # Iterate through voxels
        working_ray = Ray(ray.p.copy(), ray.d.copy())
        while voxel is not None:
            # Get the polygons that intersect with the voxel
            poly_nodes = self.get_polygon_nodes(voxel)
            # Check for intersections with the polygon nodes
            for poly_node in poly_nodes:
                if poly_node.uid in checked:
                    continue
                pint = poly_node.poly.ray_intersection(working_ray)
                if pint is not None:
                    intersections[poly_node.uid] = (poly_node, pint)
                checked[poly_node.uid] = poly_node
            if not all_intersections and len(intersections) > 0:
                break
            # Move to the next voxel
            voxel.next_voxel(working_ray)
            exit_plane, exit_t = voxel.exit(working_ray)
            working_ray.p = working_ray.p + working_ray.d*exit_t
            # Quit if the voxel is not in the volume
            if (not self.volume.contains_point(working_ray.p)) or repr(voxel.coord) not in self.voxel_dict:
                voxel = None

        intersections = [(pn, pint, np.linalg.norm(pint - ray.p)) for pn, pint in intersections.values()]
        intersections = sorted(intersections, key=lambda x: x[2])
        return intersections

    def get_containing_surface(self, ray, intersections=None, return_intersections=False):
        """
        Compute the surface that contains the ray
        """
        if intersections is None:
            intersections = self.ray_trace(ray, all_intersections=True)
        n = len(intersections)
        if not self.volume.contains_point(ray.p) or n == 0:
            if return_intersections:
                print 'Setting node to None'
                return None, intersections
            else:
                return None
        last = -2
        geo_nodes = [None]
        current = -2
        for i in xrange(n):
            i = n-1 - i
            node = intersections[i][0].geo_node
            next_l = node.level
            mod = False
            if next_l > last:
                current += 1
                geo_nodes.append(node)
                mod = False
            elif next_l == last:
                if mod:
                    current += 1
                    geo_nodes.append(node)
                else:
                    current -= 1
                    geo_nodes.pop()
                mod = not mod
            else:
                current -= 1
                geo_nodes.pop()
                mod = False
        if return_intersections:
            return geo_nodes[-1], intersections
        else:
            return geo_nodes[-1]

def rotate(v, euler_angles):
    """
    Rotate a vector according to euler angles
    """
    s1, s2, s3 = np.sin(euler_angles)
    c1, c2, c3 = np.cos(euler_angles)
    M = [
            [c1*c3 - c2*s1*s3,  -c1*s3-c2*c3*s1,    s1*s2   ],
            [c3*s1+c1*c2*s3,    c1*c2*c3-s1*s3,     -c1*s2  ],
            [s2*s3,             c3*s2,              c2      ],
        ]
    return np.dot(M, v)

def deflect_vector(v, theta, phi, preserve_mag=False):
    """
    Deflect a vector by a polar and azimuthal angle
    Uses an arbitrary coordinate system to specify the phi=0 angle
    """
    if preserve_mag:
        mag = np.linalg.norm(v)
    else:
        mag = 1.0
    unit_vecs = get_local_plane_coordinate_system(v)
    local_vector = np.dot(unit_vecs, v)
    local_vector = local_vector / np.linalg.norm(local_vector)
    local_vector = rotate(local_vector, np.array([phi, theta, 0]))
    local_vector = local_vector / np.linalg.norm(local_vector)
    v1 = np.dot(unit_vecs.T, local_vector)
    v1 = v1 / np.linalg.norm(v1)
    return v1*mag

def get_projected_angles(v0, v1):
    """
    Decompose the angle between v0, v1 into two parts
    Uses an arbitrary coordinate system to do the decomposition
    """
    unit_vecs = get_local_plane_coordinate_system(v0)
    local_vector = np.dot(unit_vecs, v1)
    thetax = np.arcsin(local_vector[0]/np.sqrt(local_vector[0]**2+local_vector[2]**2))
    thetay = np.arcsin(local_vector[1]/np.sqrt(local_vector[1]**2+local_vector[2]**2))
    return thetax, thetay

def regular_prism_surface(r=1., l=1., n=6, center=np.array([0,0,0]), rotation=None):
    """
    Construct prism with a regular polygon cross section
    """
    if rotation is None:
        rot = lambda x, a: x
    else:
        rot = rotate
    center = np.array(center)
    rad = 2*np.pi
    z = l/2
    upper_points = np.zeros((n, 3))
    lower_points = np.zeros((n, 3))
    for i in xrange(n):
        theta = rad * i/n
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        upper_points[i] = rot(np.array([x, y, z]), rotation) + center
        lower_points[i] = rot(np.array([x, y, -z]), rotation) + center
    sides = [[upper_points[i], upper_points[(i+1)%n], lower_points[(i+1)%n], lower_points[i]] for i in xrange(n)]
    polygons = [upper_points, lower_points] + sides
    polygons = [Polygon(Plane(*fit_plane(p)), p) for p in polygons]
    surface = Surface(polygons)
    return surface

def hex_surface(r=1., l=1.):
    """
    Construct a hexagonal prism
    """
    r32 = np.sqrt(3.0)/2.0
    hex_points = np.array([
        [1,0,0],
        [0.5,r32,0],
        [-0.5,r32,0],
        [-1,0,0],
        [-0.5,-r32,0],
        [0.5,-r32,0],
        ])
    upper_hex = hex_points * r + zhat*l/2
    lower_hex = hex_points * r - zhat*l/2
    sides = [[upper_hex[i], upper_hex[(i+1)%len(hex_points)], lower_hex[(i+1)%len(hex_points)], lower_hex[i]] for i in xrange(len(hex_points))]
    polygons = [upper_hex] + sides
    polygons = [Polygon(Plane(*fit_plane(p)), p) for p in polygons]
    surface = Surface(polygons)
    return surface

