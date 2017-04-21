import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import uuid
from collections import deque

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
        self.d = d

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

class PolyNode:
    def __init__(self, poly, surface):
        self.poly = poly
        self.surface = surface

class Voxel:
    def __init__(self, coord, scale):
        self.coord = coord
        self.scale = scale
        self.half_scale = scale/2.0
        self.reinit()
    
    def reinit(self)
        self.center = coord * scale
        self.min_planes = self.center - self.half_scale
        self.max_planes = self.center + self.half_scale

    def in_voxel(self, point):
        """
        Returns true if point is inside voxel or on boundary
        """
        return np.all(point >= self.min_planes) and np.all(point <= self.max_planes)

    def plane_t(self, dim, ray):
        p = ray.p
        d = ray.d
        if d[dim] > 0:
            t = ((self.max_planes[dim] - p[dim])/d[dim], 1, dim)
        elif d[dim] < 0:
            t = ((self.min_planes[dim] - p[dim])/d[dim], 0, dim)
        else:
            t = (np.inf, None)
        return t

    def exit_plane(self, ray):
        if self.in_voxel(ray.p):
            plane_vals = [self.plane_t(dim, ray) for dim in xrange(3)]
            min_val = min([val for val in plane_vals if val[0] > 0], key=lambda x: x[0])
            return unit_vecs[min_val[2]] * (min_val[1]*2 - 1)
        else:
            return np.zeros(3)

    def next_voxel(self, ray):
        self.coord += self.exit_plane()
        self.reinit()

class PolyContainer:
    def __init__(self, surfaces):


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
    def __init__(self, root=None):
        self.root = root
        self.nodes = []
        untraversed = deque()
        if root is not None:
            untraversed.append(root)
        while len(untraversed) > 0:
            next_node = untraversed.popleft()
            self.nodes.append(next_node)
            if next_node.children is not None:
                untraversed.extend(next_node.children)

        


r32 = np.sqrt(3.0)/2.0
zhat= np.array([0,0,1])
hex_points = np.array([ 
        [1,0,0],
        [0.5,r32,0],
        [-0.5,r32,0],
        [-1,0,0],
        [-0.5,-r32,0],
        [0.5,-r32,0],
        ])
r = 10
l = 10
upper_hex = hex_points * r + zhat*l/2
lower_hex = hex_points * r - zhat*l/2
sides = [[upper_hex[i], upper_hex[(i+1)%len(hex_points)], lower_hex[(i+1)%len(hex_points)], lower_hex[i]] for i in xrange(len(hex_points))]
polygons = [upper_hex] + sides
polygons = [Polygon(Plane(*fit_plane(p)), p) for p in polygons]

surface = Surface(polygons)

N = 10000

points = np.zeros((N, 3)) + 5
points[:,0:2] = np.random.uniform(-10, 10, (N, 2))

#points = np.array([[9,9,10]])

w = []

for p in points:
    w.append(polygons[0].winding_number(np.array(p)))

w = np.array(w)

plt.plot(points[w!=0,0], points[w!=0,1], color='b', marker='.', linestyle='')
plt.plot(points[w==0,0], points[w==0,1], color='r', marker='.', linestyle='')
upper_hex = np.array(list(upper_hex) + [list(upper_hex[0])])
plt.plot(upper_hex[:,0], upper_hex[:,1], color='m')
plt.show()
