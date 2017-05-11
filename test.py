import numpy as np
from matplotlib import pyplot as plt
import geometry as geo
import materials
import tracking
import pickle
import uuid
import time
import itertools
import collections

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

    tank_node = geo.GeometryNode(geo.regular_prism_surface(r=.92, l=2, n=12, center=np.array([0,0,0]), rotation=None), name='tank')
    tank_node.properties = materials.get_properties(13)

    water_node = geo.GeometryNode(geo.regular_prism_surface(r=.915, l=1.98, n=12, center=np.array([0,0,0]), rotation=None), parent=tank_node, name='water')
    water_node.properties = materials.get_properties('water')

    rod_node = geo.GeometryNode(geo.regular_prism_surface(r=.064, l=1.96, n=6, center=np.array([0,0,0]), rotation=None), parent=water_node, name='metal_rod')
    rod_node.properties = materials.get_properties(13)

    U_node = geo.GeometryNode(geo.regular_prism_surface(r=.06, l=1.5, n=6, center=np.array([0,0,0]), rotation=None), parent=rod_node, name='U')
    U_node.properties = materials.get_properties(92)

    scale = np.array([3., .1, 3.])
    big_scale = np.array([6., 6., 6.])
    big = np.array([0.,0.,0.])
    upper = np.array([0., -1., 1.])
    lower = np.array([0., 1., -1.])
    big_det_surface = geo.Surface(geo.Voxel(big, big_scale, vol=True).get_faces())
    #upper_det_surface = geo.Surface(geo.Voxel(upper, scale, vol=True).get_faces())
    #lower_det_surface = geo.Surface(geo.Voxel(lower, scale, vol=True).get_faces())
    #upper_node = geo.GeometryNode(upper_det_surface)
    #lower_node = geo.GeometryNode(lower_det_surface)
    big_node = geo.GeometryNode(big_det_surface, name='big_detector')
    big_node.properties = materials.get_properties(7)
    #upper_node.properties = materials.get_properties(6)
    #lower_node.properties = materials.get_properties(6)
    big_node.properties['det'] = 'big'
    #upper_node.properties['det'] = 'upper'
    #lower_node.properties['det'] = 'lower'
    #g = geo.Geometry([tank_node, upper_node, lower_node], voxel_scale=1)
    big_node.add_child(tank_node)
    g = geo.Geometry([big_node], voxel_scale=1)

    node = big_node
    while node is not None:
        print node.level, ':', node.properties['Z']
        if len(node.children) == 0:
            node = None
        else:
            node = node.children[0]

    sim_scale = np.array([12., 12., 12.])
    sim = np.array([0.,0.,0.])
    sim_v = geo.Voxel(sim, sim_scale, vol=True)

    # Define energy spectrum parameters
    gamma = -2.
    E0 = 10.
    E1 = 100.

    logs = []
    maxZs = []
    for i in xrange(n):
        if i%100 == 0:
            print i
            print collections.Counter(maxZs)
        phi = 2*np.pi*np.random.uniform()
        theta = np.arccos(1.-2.*np.random.uniform())
        position = geo.deflect_vector(np.array([0.,0.,5.5]), theta, phi, preserve_mag=True)
        direction = np.array([0,0,0]) - position
        direction = direction/np.linalg.norm(direction)
        phi = 2*np.pi*np.random.uniform()
        theta = np.arccos(1.-0.03*np.random.uniform())
        direction = geo.deflect_vector(direction, theta, phi, preserve_mag=False)
        mass = tracking.muon_mass
        y = np.random.uniform()
        energy = ((E1**(gamma+1)-E0**(gamma+1))*y + E0**(gamma+1))**(1/(gamma+1))
        track = tracking.Track(position, direction, mass, energy, g)

        prop = tracking.Propagator(track)
        start = time.time()
        while True:
            try:
                result = prop.propagate_step()
            except KeyboardInterrupt:
                raise
            outside_sim_v = not sim_v.contains_point(prop.track.ray.p)
            hit_detector = len(track.detector_log) >= 2
            if outside_sim_v or hit_detector:
                #print 'hit:', hit_detector, ', outside:', outside_sim_v, ', point:', prop.track.ray.p
                #print track.detector_log
                prop.finish()
                break
        end = time.time()
        duration = end - start
        log = track.detector_log
        if len(log) == 2:
            #log = [[log[0][0], log[0][1]], [log[1][0], log[1][1]]]
            maxZ = max(prop.material_log, key=lambda x: x[1])[1]
            maxZs.append(maxZ)
            logs.append(log)
    return logs, maxZs

def run():
    uid = str(uuid.uuid4())
    i = 0
    while True:
        logs, maxZs = test_detection(2000)
        print 'Done with', i
        f = open('./pkl/'+uid+'_'+str(i)+'.pkl', 'wb')
        pickle.dump(logs, f)
        f.close()
        #print logs
        del logs
        i = i + 1

#test_surface()
#test_geo()
#test_propagation()
#print test_detection(10000)
