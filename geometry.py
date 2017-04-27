import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import uuid
from collections import deque
import itertools

xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])
unit_vecs = np.array([xhat, yhat, zhat])

class Plane:
    def __init__(self, point, vector):
        self.point = point
        self.vector = vector / np.linalg.norm(vector)

        min_index = min(enumerate(self.vector), key=lambda x: x[1])[0]
        y_dir = unit_vecs[min_index] - self.vector*np.dot(unit_vecs[min_index], self.vector)
        y_dir = y_dir / np.linalg.norm(y_dir)
        x_dir = np.cross(y_dir, self.vector)
        x_dir = x_dir / np.linalg.norm(x_dir)

        self.unit_vecs = np.array([x_dir, y_dir, self.vector])

    def __eq__(self, other):
        if (self.point != other.point).any():
            return False
        if (self.vector != other.vector).any():
            return False
        return True

    def distance(self, p):
        t = np.dot(self.vector, (self.point - p))
        return t

    def project(self, p):
        return p + self.vector * self.distance(p)

    def ray_intersection(self, ray):
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
    def __init__(self, p, d):
        self.p = p
        self.d = d / np.linalg.norm(d)
    def pca_to_point(self, point):
        return self.p + self.d*np.dot(point - self.p, self.d)

class Polygon:
    def __init__(self, plane, points, strict=True):
        self.plane = plane
        self.points = points
        self.edges = []
        for i, p in enumerate(self.points):
            t = self.plane.distance(p)
            if abs(t) >= 1e-5:
                if strict:
                    raise ValueError('Point not on plane')
                print 'Projecting point', p, 'onto plane'
                self.points[i] = self.plane.project()
        for i in xrange(len(self.points)):
            self.edges.append(Edge(self.points[i], self.points[(i+1) % len(self.points)]))

    def get_edges(self):
        return self.edges

    def winding_number(self, point):
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
        return self.winding_number(point) != 0

    def ray_intersection(self, ray):
        pint = self.plane.ray_intersection(ray)
        if pint is not None and self.contains(pint):
            return pint
        else:
            return None

class Surface:
    def __init__(self, polygons):
        self.polygons = polygons
        raw_edges = [e for p in self.polygons for e in p.get_edges()]
        edge_counts = Counter(raw_edges)
        naked_edges = []
        point_dict = dict()
        for e, c in edge_counts.items():
            if c == 1:
                for p in e.points():
                    if repr(p) not in point_dict:
                        point_dict[repr(p)] = []
                    point_dict[repr(p)].append(e)
                naked_edges.append(e)
            elif c > 2:
                raise ValueError('Edge shared by too many polygons')
        if len(naked_edges) > 0:
            for p, edges in point_dict.items():
                if len(edges) != 2:
                    raise ValueError('Polygon not well connected')
            last_edge = naked_edges[0]
            last_point = last_edge.points()[0]
            ordered_edges = [last_edge]
            ordered_points = [last_point]
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
            center = sum(ordered_points) / len(ordered_points)
            normal = np.linalg.svd(np.transpose(np.array(ordered_points - center)))[0][-1]
            new_plane = Plane(center, normal)
            new_polygon = Polygon(new_plane, ordered_points)
            self.polygons.append(new_polygon)

class Voxel:
    def __init__(self, coord, scale, vol=False):
        self.vol = vol
        if self.vol:
            self.center = coord
        else:
            self.coord = coord

        if not hasattr(scale, '__len__'):
            scale = [scale, scale, scale]
        self.scale = np.array(scale)
        self.reinit()

    def reinit(self):
        self.half_scale = self.scale/2.0
        if not self.vol:
            self.center = self.coord * self.scale
        self.min_planes = self.center - self.half_scale
        self.max_planes = self.center + self.half_scale
        self.planes = [self.min_planes, self.max_planes]
        #print self.planes

    def contains_point(self, point, include_boundaries=True):
        """
        Returns true if point is inside voxel or on boundary
        """
        if include_boundaries:
            return np.all(point >= self.min_planes) and np.all(point <= self.max_planes)
        else:
            return np.all(point > self.min_planes) and np.all(point < self.max_planes)

    def in_voxel(self, voxel):
        corner = np.zeros(3)
        n_inside = 0
        for i in xrange(0, 8):
            for d in [0, 1, 2]:
                corner[d] = self.planes[i >> d & 0x1][d]
            if voxel.contains_point(corner, include_boundaries=False):
                return True
            if voxel.contains_point(corner, include_boundaries=True):
                n_inside += 1
        if n_inside == 8:
            return True
        return False

    def plane_t(self, dim, ray, invert=False):
        p = ray.p
        d = ray.d
        if (d[dim] > 0) ^ invert:
            t = ((self.max_planes[dim] - p[dim])/d[dim], 1, dim)
        elif (d[dim] < 0) ^ invert:
            t = ((self.min_planes[dim] - p[dim])/d[dim], 0, dim)
        else:
            t = (np.inf, None)
        return t

    def exit(self, ray, invert=False):
        if self.contains_point(ray.p) ^ invert:
            plane_vals = [self.plane_t(dim, ray, invert) for dim in xrange(3)]
            min_val = min([val for val in plane_vals if val[0] > 0], key=lambda x: x[0])
            return (unit_vecs[min_val[2]] * (min_val[1]*2 - 1), min_val[0])
        else:
            return (np.zeros(3), None)

    def entry(self, ray):
        return self.exit(ray, invert=True)

    def exit_plane(self, ray, invert=False):
        return self.exit(ray, invert)[0]

    def entry_plane(self, ray):
        return self.entry(ray)[0]

    def exit_t(self, ray, invert=False):
        return self.exit(ray, invert)[1]

    def entry_t(self, ray):
        return self.entry(ray)[1]

    def next_voxel(self, ray):
        self.coord += self.exit_plane(ray)
        self.reinit()

    def ray_intersects(self, ray):
        return np.any(self.exit_plane(ray) != np.zeros(3))

    def poly_intersects(self, polygon):
        for p in polygon.points:
            if self.contains_point(p):
                #print 'in voxel'
                return True
        #print polygon.points
        #print 'projection:', polygon.plane.project(self.center)
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
        for d in [0, 1, 2]:
            v = unit_vecs[d]
            d0 = (d + 1) % 3
            d1 = (d + 2) % 3
            for dm0, dm1 in [(0,0), (0,1), (1,0), (1,1)]:
                db0 = self.planes[dm0][d0]
                db1 = self.planes[dm1][d1]
                p0 = self.center.copy()
                p0[d0] = db0
                p0[d1] = db1
                ndv = np.dot(n, v)
                if ndv == 0:
                    continue
                pint = v*(np.dot(n, x0-p0)/np.dot(n, v)) + p0
                #print pint
                if np.linalg.norm(pint - p0) <= self.half_scale[d]:
                    if polygon.contains(pint):
                        return True
        return False

class PolyNode:
    def __init__(self, poly, surface):
        self.poly = poly
        self.surface = surface
        self.uid = uuid.uuid4()

class GeometryNode:
    def __init__(self, surface, properties=None, parent=None, children=[]):
        self.surface = surface
        self.properties = properties
        self.uid = uuid.uuid4()
        self.parent = parent
        self.children = children
    def add_child(self, child):
        self.children.append(child)
        child.parent = self

class Geometry:
    def __init__(self, root=None, voxel_scale=2):
        self.root = root
        self.nodes = []
        point_dict = dict()
        poly_nodes = []
        untraversed = deque()
        if root is not None:
            untraversed.append(root)
        while len(untraversed) > 0:
            next_node = untraversed.popleft()
            self.nodes.append(next_node)
            if next_node.children is not None:
                untraversed.extend(next_node.children)
            for poly in next_node.surface.polygons:
                poly_node = PolyNode(poly, next_node)
                poly_nodes.append(poly_node)
                for p in poly.points:
                    point_dict[repr(p)] = p
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

        volume_coord = (max_v + min_v) / 2.0 * voxel_scale
        volume_scale = (max_v - min_v+1) * voxel_scale
        self.volume = Voxel(volume_coord, volume_scale, vol=True)

        self.voxel_dict = dict()
        grid = np.meshgrid(np.linspace(*lspaces[0]), np.linspace(*lspaces[1]), np.linspace(*lspaces[2]))
        voxel_coords = (np.array([x,y,z]) for x, y, z in itertools.izip(grid[0].flatten(), grid[1].flatten(), grid[2].flatten()))
        voxel = Voxel(np.zeros(3), voxel_scale)
        self.voxel_scale = voxel.scale
        #print lspaces
        for vc in voxel_coords:
            k = repr(vc)
            self.voxel_dict[k] = []
            voxel.coord = vc
            voxel.reinit()
            for pn in poly_nodes:
                if voxel.poly_intersects(pn.poly):
                    #print 'intersection!'
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
        current_voxel = get_containing_voxel(ray.p)
        voxel = Voxel(current_voxel, self.voxel_scale)
        return voxel.exit_t(ray)

    def ray_trace(self, ray, all_intersections=False):
        intersections = dict()
        checked = dict()
        # Check if the ray is already in the volume
        if self.volume.contains_point(ray.p):
            print 'Start inside the volume'
            # Find the voxel containing the ray
            voxel_coord = np.floor(ray.p/self.voxel_scale + 0.5)
            voxel = Voxel(voxel_coord, self.voxel_scale)
        else:
            print 'Start outside the volume'
            # Find the intersection with the volume
            entry_t = self.volume.entry_t(ray)
            print 'Entry time:', entry_t
            if entry_t is None:
                print 'No entry point'
                voxel = None
            else:
                pint = ray.p+ray.d*entry_t
                print 'Entry point:', pint
                voxel_coord = np.floor(pint/self.voxel_scale + 0.5)
                voxel = Voxel(voxel_coord, self.voxel_scale)
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
            voxel.next_voxel(working_ray)
            working_ray.p = ray.pca_to_point(voxel.center)
            if not voxel.in_voxel(self.volume):
                voxel = None

        intersections = [(pn, pint, np.linalg.norm(pint - ray.p)) for pn, pint in intersections.values()]
        intersections = sorted(intersections, key=lambda x: x[2])
        return intersections

def rotate(v, euler_angles):
    s1, s2, s3 = np.sin(euler_angles)
    c1, c2, c3 = np.cos(euler_angles)
    M = [
            [c1*c3 - c2*s1*s3,  -c1*s3-c2*c3*s1,    s1*s2   ],
            [c3*s1+c1*c2*s3,    c1*c2*c3-s1*s3,     -c1*s2  ],
            [s2*s3,             c3*s2,              c2      ],
        ]
    return np.dot(M, v)


def regular_prism_surface(r=1., l=1., n=6, center=np.array([0,0,0]), rotation=None):
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
    # Construct hexagonal prism geometry
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

