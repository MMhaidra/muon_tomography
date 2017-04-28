import numpy as np
from matplotlib import pyplot as plt
import geometry as geo
import materials
import tracking


def test_surface():
    # Test surface
    N = 10000
    surface = geo.hex_surface()
    v_scale = 2.0
    v = geo.Voxel(np.array([0, 0, 3]), v_scale)
    i = 2
    points = np.random.uniform(-v_scale*2, v_scale*2, (N, 3)) + np.array([0, 0, 3]) * v_scale
    points[:,i] = v.center[i]
    w = np.array([v.contains_point(p) for p in points])

    plt.plot(points[w,(i+1)%3], points[w,(i+2)%3], color='b', marker='.', linestyle='')
    plt.plot(points[w == False,(i+1)%3], points[w == False,(i+2)%3], color='r', marker='.', linestyle='')
    plt.show()

def test_geo():
    # Test geometry
    surface = geo.hex_surface()
    root_node = geo.GeometryNode(surface)
    g = geo.Geometry([root_node], voxel_scale=2.0)
    #print len(geo.voxel_dict.values())
    #print len([val for val in geo.voxel_dict.values() if len(val) > 0])
    ray = geo.Ray(np.array([0, 0, -20]), np.array([0, 0, 1]))
    print g.ray_trace(ray)
    print g.ray_trace(ray, all_intersections=True)

def test_propagation():
    surface = geo.hex_surface()
    root_node = geo.GeometryNode(surface)
    root_node.properties = materials.get_properties(92)
    g = geo.Geometry([root_node], voxel_scale=2.0)
    position = np.random.uniform(size=3)
    direction = -position / np.linalg.norm(position)
    position *= 2
    #print position, direction
    #position, direction = (np.array([0, 0, -20]), np.array([0, 0, 1]))
    mass = tracking.muon_mass
    energy = 1
    track = tracking.Track(position, direction, mass, energy, g)
    #print track.ray.p, track.ray.d

    while True:
        prop = tracking.Propagator(track)
        result = prop.propagate_step()
        #print track.ray.p, track.ray.d
        if not track.geometry.volume.contains_point(track.ray.p) or not result:
            break

#test_surface()
#test_geo()
test_propagation()
