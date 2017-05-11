import numpy as np
import geometry as geo
import tracking
import pickle
import itertools
import glob

#### Data format of output files ####
# *.pkl -> list(data_points)
# data_point -> [upper_data, lower_data]
# upper/lower_data -> [energy, position, direction]
####

def get_entries(data_point, mass):
    upper = data_point[0]
    lower = data_point[1]
    avg_E = (upper[0] + lower[0])
    momentum = np.sqrt(avg_E**2. - mass**2.)
    thetax, thetay = geo.get_projected_angles(upper[2], lower[2])
    return abs(momentum*thetax), abs(momentum*thetay)

def get_Dr(point, ray0, ray1):
    return max(np.linalg.norm(point - ray0.pca_to_point(point)), np.linalg.norm(point -ray1.pca_to_point(point)))

def make_voxel_structure(max_points, min_points, scale=0.05):
    n_voxels = np.ceil((max_points-min_points)/scale)
    center = (max_points + min_points) / 2.0
    max_p = center + n_voxels/2.0*scale
    min_p = center - n_voxels/2.0*scale

    lspaces = []
    for i in [0, 1, 2]:
        lspaces.append((min_p[i], max_p[i], n_voxels[i]+1))
    grid = np.meshgrid(np.linspace(*lspaces[0]), np.linspace(*lspaces[1]), np.linspace(*lspaces[2]))
    
    #voxel_coords = [np.array([x,y,z]) for x, y, z in itertools.izip(grid[0].flatten(), grid[1].flatten(), grid[2].flatten())]
    return grid

def fill_list(ray0, ray1, entries, coords, data_list, dh):
    for i,p in enumerate(coords):
        if get_Dr(p, ray0, ray1) <= dh:
            data_list[i].extend(entries)

def toca(r0, r1):
    w0 = r1.p - r0.p
    u = r0.d
    v = r1.d
    
    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w0)
    e = np.dot(v,w0)

    denom = a*c-b**2
    if denom == 0:
        return None
    s = (b*e-c*d)/denom
    t = (a*e-b*d)/denom
    return (s, t)

def poca(r0, r1):
    t = toca(r0, r1)
    if t is None:
        return t
    s, t = t
    p0 = r0.p + s*r0.d
    p1 = r1.p + t*r1.d
    return (p0, p1)

def doca(r0, r1):
    p = poca(r0, r1)
    if p is None:
        return p
    p0, p1 = p
    return np.linalg.norm(p0-p1)

def get_data(fname, scale, max_p, min_p):
    f = open(fname, 'rb')
    data = pickle.load(f)
    f.close()

    min_v = np.floor(min_p/scale)
    max_v = np.ceil(max_p/scale)
    volume_coord = (max_v + min_v) / 2.0 * scale
    volume_scale = (max_v - min_v+1) * scale
    volume = geo.Voxel(volume_coord, volume_scale, vol=True)
    maxiter = int(max(max_p - min_p)*np.sqrt(3)/(scale)*8+5)
    lspaces = []
    for i in [0, 1, 2]:
        lspaces.append((min_v[i], max_v[i], max_v[i]-min_v[i]+1))

    get_state = lambda p: np.all(p >= min_p) and np.all(p <= max_p)

    dh = scale*np.sqrt(3)

    actual_hits = []
    for d in data:
        #print d
        r0 = geo.Ray(d[0][1], d[0][2])
        r1 = geo.Ray(d[1][1], -d[1][2])
        #if not (volume.ray_intersects(r0) and volume.ray_intersects(r1)):
        #    continue
        entries = get_entries(d, tracking.muon_mass)
        hit_sets = []
        for i, ray in enumerate([r0, r1]):
            #print ray.p, ray.d
            voxel_coord = np.floor(ray.p/scale + 0.5)
            #print voxel_coord
            voxel = geo.Voxel(voxel_coord, scale)
            #print voxel.coord
            voxel_hits = []
            state = get_state(voxel.center)
            flipper = int(state)
            it = 0
            while flipper < 2 and it <= maxiter:
                it += 1
                voxel_hits.append(voxel.coord.copy())
                #for bm in xrange(6):
                #    dim = bm/2
                #    m = bm%2
                #    vc = voxel.coord + (-1)**m * geo.unit_vecs[dim]
                #    p = vc*scale
                #    if np.linalg.norm(p - ray.pca_to_point(p)) <= dh:
                #        voxel_hits.append(vc)
                voxel.next_voxel(ray)
                #print voxel.center, flipper, state
                #exit_plane, exit_t = voxel.exit(ray)
                #ray.p = ray.p + ray.d*exit_t
                new_state = get_state(voxel.center)
                if new_state != state:
                    flipper += 1
                    state = new_state
            vh_set = [tuple([int(xx) for xx in x]) for x in voxel_hits]
            #print vh_set
            hit_sets.append(vh_set)

        hs0 = set(hit_sets[0])
        hs1 = set(hit_sets[1])
        #print doca(r0, r1)
        for vc in hs0:
            if vc in hs1:
                actual_hits.extend([(vc, entries[0]), (vc, entries[1])])
    #print actual_hits
    return actual_hits

def run_reco(files='./*.pkl', output='./reco_out.pkl'):
    data_dict = {}
    i = 0
    for fname in glob.glob(files):
        hits = get_data(fname, 0.1, np.array([2,2,2]), np.array([-2,-2,-2]))
        for h in hits:
            if not h[0] in data_dict:
                data_dict[h[0]] = []
            data_dict[h[0]].append(h[1])
        print i
        i += 1

    f = open(output, 'wb')
    pickle.dump(data_dict, f)
    f.close()
