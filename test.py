import numpy as np
from matplotlib import pyplot as plt
import geometry as geo
import materials
import tracking
import pickle
import uuid

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

    prop = tracking.Propagator(track)
    while True:
        result = prop.propagate_step()
        #print track.ray.p, track.ray.d
        if not track.geometry.volume.contains_point(track.ray.p) or not result:
            break

def test_detection(n = 1):

    tank_node = geo.GeometryNode(geo.regular_prism_surface(r=.92, l=2, n=12, center=np.array([0,0,0]), rotation=None))
    tank_node.properties = materials.get_properties(13)

    water_node = geo.GeometryNode(geo.regular_prism_surface(r=.915, l=1.98, n=12, center=np.array([0,0,0]), rotation=None), parent=tank_node)
    water_node.properties = materials.get_properties('water')

    rod_node = geo.GeometryNode(geo.regular_prism_surface(r=.064, l=.98, n=6, center=np.array([0,0,0]), rotation=None), parent=water_node)
    rod_node.properties = materials.get_properties(13)

    U_node = geo.GeometryNode(geo.regular_prism_surface(r=.06, l=1.5, n=6, center=np.array([0,0,0]), rotation=None), parent=rod_node)
    U_node.properties = materials.get_properties(92)

    scale = np.array([3., .1, 3.])
    upper = np.array([0., -1., 1.])
    lower = np.array([0., 1., -1.])
    upper_det_surface = geo.Surface(geo.Voxel(upper, scale, vol=True).get_faces())
    lower_det_surface = geo.Surface(geo.Voxel(lower, scale, vol=True).get_faces())
    upper_node = geo.GeometryNode(upper_det_surface)
    lower_node = geo.GeometryNode(lower_det_surface)
    upper_node.properties = materials.get_properties(6)
    lower_node.properties = materials.get_properties(6)
    upper_node.properties['det'] = 'upper'
    lower_node.properties['det'] = 'lower'
    g = geo.Geometry([tank_node, upper_node, lower_node], voxel_scale=1)

    logs = []
    for i in xrange(n):
        phi = 2*np.pi*np.random.uniform()
        theta = np.arccos(0.5*np.random.uniform()+0.5)
        position = geo.deflect_vector(upper, theta, phi)*10
        direction = -position/np.linalg.norm(position)
        #print position, direction
        #position, direction = (np.array([0, 0, -20]), np.array([0, 0, 1]))
        mass = tracking.muon_mass
        n = -2.
        E0 = 0.5
        E1 = 100.
        y = np.random.uniform()
        energy = ((E1**(n+1)-E0**(n+1))*y + E0**(n+1))**(1/(n+1))
        track = tracking.Track(position, direction, mass, energy, g)
        #print track.ray.p, track.ray.d

        prop = tracking.Propagator(track)
        while True:
            result = prop.propagate_step()
            #print track.ray.p, track.ray.d
            if not track.geometry.volume.contains_point(track.ray.p) or not result:
                break
        log = prop.log
        if len(log) > 0:
            upper_l = [l for l in log if l[3] == 'upper']
            lower_l = [l for l in log if l[3] == 'lower']
            if len(upper_l) > 0 and len(lower_l) > 0:
                ul = min(upper_l, key=lambda x: np.linalg.norm(x[1]))
                ll = min(lower_l, key=lambda x: np.linalg.norm(x[1]))
                log = [[ul[0], ul[1], ul[2]], [ll[0], ll[1], ll[2]]]
                logs.append(log)
    return logs

def run():
    uid = str(uuid.uuid4())
    i = 0
    while True:
        logs = test_detection(100000)
        print 'Done with', i
        f = open(uid+'_'+str(i)+'.pkl', 'wb')
        pickle.dump(logs, f)
        f.close()
        del logs
        i = i + 1

#test_surface()
#test_geo()
#test_propagation()
#print test_detection(10000)
