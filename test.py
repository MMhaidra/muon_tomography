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

    prop = tracking.Propagator(track, detailed_logging=True)
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

    sim_scale = np.array([11., 11., 11.])
    sim = np.array([0.,0.,0.])
    sim_v = geo.Voxel(sim, sim_scale, vol=True)

    logs = []
    maxZs = []
    for i in xrange(n):
        #if i%1000 == 0:
        print i
        print collections.Counter(maxZs)
        phi = 2*np.pi*np.random.uniform()
        theta = np.arccos(1.-2.*np.random.uniform())
        position = geo.deflect_vector(np.array([3.,3.,3.1]), theta, phi, preserve_mag=True)
        print position, sim_v.contains_point(position), g.volume.contains_point(position)
        direction = np.array([0,0,0]) - position
        direction = direction/np.linalg.norm(direction)
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

        prop = tracking.Propagator(track, detailed_logging=True)
        start = time.time()
        while True:
            try:
                result = prop.propagate_step()
            except KeyboardInterrupt:
                print prop.detailed_log
                raise
            #print track.ray.p, track.ray.d
            outside_sim_v = not sim_v.contains_point(prop.track.ray.p)
            hit_detector = len(prop.detector_log) >= 2
            if outside_sim_v or hit_detector:
                print 'hit:', hit_detector, ', outside:', outside_sim_v, ', point:', prop.track.ray.p
                prop.finish()
                break
        end = time.time()
        duration = end - start
        log = prop.detector_log
        if duration > 1 or len(log) == 1:
            print 'Muon propagation took more than 1s!'
            print 'Duration:', duration
            print 'Maximum Z:', max(prop.material_log, key=lambda x: x[1])[1]
            print 'Number of steps:', len(prop.detailed_log)
            print 'Maximum r:', max(enumerate([np.linalg.norm(l[0]) for l in prop.detailed_log]), key=lambda x: x[1])
            p0 = prop.detailed_log[0][0]
            v0 = prop.detailed_log[0][1]
            total_displacement = [np.linalg.norm(p-(p0 + np.dot(p-p0, v0)*v0)) for p,v in prop.detailed_log]
            total_deflection = [np.arccos(np.dot(v0, v)) for p,v in prop.detailed_log]
            deflection = [np.arccos(np.dot(pv1[1],pv2[1])) for pv1,pv2 in itertools.izip(prop.detailed_log[:-1], prop.detailed_log[1:])]
            displacement = [np.linalg.norm(pv2[0]-(pv1[0] + np.dot(pv2[0]-pv1[0], pv1[1])*pv1[1])) for pv1,pv2 in itertools.izip(prop.detailed_log[:-1], prop.detailed_log[1:])]
            print 'Maximum deflection:', max(deflection)
            print 'Maximum displacement:', max(displacement)
            print 'Maximum total_deflection:', max(total_deflection)
            print 'Maximum total_displacement:', max(total_displacement)
            if len(log) == 1:
                print log
                for l in prop.detailed_log:
                    print l, sim_v.contains_point(l[0]), g.volume.contains_point(l[0])
                raise ValueError('Only 1 hit')
        if len(log) < 2:
            print 'Min r:', min(enumerate([np.linalg.norm(l[0]) for l in prop.detailed_log]), key=lambda x: x[1])
        if len(log) > 0:
            #upper_l = [l for l in log if l[3] == 'upper']
            #lower_l = [l for l in log if l[3] == 'lower']
            #if len(upper_l) > 0 and len(lower_l) > 0:
            if len(log) == 2:
                #ul = min(upper_l, key=lambda x: np.linalg.norm(x[1]))
                #ll = min(lower_l, key=lambda x: np.linalg.norm(x[1]))
                #log = [[ul[0], ul[1], ul[2]], [ll[0], ll[1], ll[2]]]
                log = [[log[0][0], log[0][1]], [log[1][0], log[1][1]]]
                maxZ = max(prop.material_log, key=lambda x: x[1])[1]
                maxZs.append(maxZ)
                print log
                logs.append(log)
    return logs, maxZs

def run():
    uid = str(uuid.uuid4())
    i = 0
    while True:
        logs, maxZs = test_detection(100000)
        print 'Done with', i
        f = open(uid+'_'+str(i)+'.pkl', 'wb')
        pickle.dump(logs, f)
        f.close()
        print logs
        del logs
        i = i + 1
        break

#test_surface()
#test_geo()
#test_propagation()
#print test_detection(10000)
