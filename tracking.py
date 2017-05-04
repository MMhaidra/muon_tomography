import numpy as np
import geometry as geo

hb = 6.582119514e-16 # (GeV*ns)
c = .299792458 # m/ns
c2 = c**2 # m/ns
a0 = 5.2917721067e-11 # m
e2_epsilon0 = 1.8095128e-17 # (m*GeV)
muon_mass = .105658374 # (GeV)

def gamma(energy, mass):
    return energy/(mass)

def beta(gamma):
    return 1.-1./gamma**2.

def screening_parameter(mass, gamma, beta, Z):
    return 0.25*(hb/(mass*gamma*beta/c))**2*(0.885*Z**(-1./3.)*a0)**(-2.)

def G1W(A):
    return 2*A*((1+A)*np.log((1+A)/A)-1)

def G2W(A):
    return 6*A*(1+A)*((1+2*A)*np.log((1+A)/A)-2)

def sigmaW(A, Z, mass, gamma, beta):
    return (Z*e2_epsilon0)**2/(mass * gamma * beta**2)**2*np.pi/(A*(1+A))

def lambdaW(N, sigmaW):
    return 1/(N*sigmaW)

def lambda1W(lambdaW, G1W):
    return lambdaW / G1W

def lambda2W(lambdaW, G2W):
    return lambdaW / G2W

def lambdah(lambdaW, Cs, lambda1W):
    #print 'setting lambdah'
    return max(lambdaW, Cs*lambda1W)

def mu_cut(A, lambdah, lambdaW):
    return A*(lambdah - lambdaW)/(A*lambdah+lambdaW)

def mu_hard(A, mu_cut):
    xi = np.random.random()
    return mu_cut + (A+mu_cut)*xi*(1.-mu_cut)/((A+1.)-xi*(1.-mu_cut))

def lambda1Ws(A, lambdaW, mu_cut):
    return 1./((2./lambdaW) * A * (1.+A)*(np.log((mu_cut+A)/A)-mu_cut/(mu_cut+A)))

def lambda2Ws(A, lambdaW, mu_cut):
    return 1./((6./lambdaW)*A*(1.+A)*((1.+2.*A)*np.log((mu_cut+A)/A)-(1.+2.*A+mu_cut)*mu_cut/(mu_cut+A)))

def mu_soft_mean(t, lambda1Ws):
    return 0.5*(1.-np.exp(-t/lambda1Ws))

def mu2_soft_mean(t, mu_soft_mean, lambda2Ws):
    return mu_soft_mean - 1./6.*(1.-np.exp(-t/lambda2Ws))

def a(mu_soft_mean, mu2_soft_mean):
    return (mu2_soft_mean-mu_soft_mean**2.)/(mu_soft_mean*(1.-mu_soft_mean))

def alpha(a):
    return (3.*a+(a*(a+8.))**0.5)/(4.*(1.-a))

def mu_soft(mu_soft_mean, alpha):
    xi = np.random.random()
    comp = 1.-mu_soft_mean
    if xi < comp:
        return mu_soft_mean*(xi/(comp))**alpha
    else:
        return 1.-comp*((1.-xi)/mu_soft_mean)**alpha

class Track:
    def __init__(self, position, direction, mass, energy, geometry, detailed_logging = True):
        self.ray = geo.Ray(position, direction)
        self.mass = mass
        self.energy = energy
        self.geo_node, self.intersections = geometry.get_containing_surface(self.ray, return_intersections=True)
        self.geometry = geometry
        self.on_boundary = False
        self.last_geo_node = None
        self.next_geo_node = None
        self.surface_exit_vector = None
        self.boundary = None
        self.detailed_logging = detailed_logging
        self.detailed_log = []
        self.detailed_log.append((self.ray.p.copy(), self.ray.d.copy(), self.get_current_level()))
        self.detector_log = []
        #self.intersections = []
        #self.update()
    def go_to_interface(self):
        #print 'Go to interface'
        self.intersections = self.geometry.ray_trace(self.ray)
        if len(self.intersections) > 0:
            poly_node, pint = self.intersections[0]
            dint = np.dot(pint - self.ray.p, self.ray.d)
            geo_node = poly_node.geo_node
            #print 'poly geo node:', geo_node
            #print 'current geo node:', self.geo_node
                
            current_level = self.get_current_level()
            #print 'Current level', current_level
            #print 'Next level', geo_node.level

            going_in = geo_node.level > current_level
            going_out = geo_node.level == current_level
            nonsense = geo_node.level < current_level

            if nonsense:
                print self.detailed_log
                print pint
                new_intersections = self.geometry.ray_trace(self.ray)
                print self.intersections
                print new_intersections
                raise ValueError('Next geo level is larger than current geo level')
            elif going_in:
                #print 'Going in!'
                self.next_geo_node = geo_node
            elif going_out:
                #print 'Going out!'
                self.next_geo_node = geo_node.parent

            self.surface_exit_vector = np.dot(self.ray.d, poly_node.poly.plane.vector)*poly_node.poly.plane.vector
            self.surface_exit_vector /= np.linalg.norm(self.surface_exit_vector)

            self.ray.p = pint
            self.intersections = self.geometry.ray_trace(self.ray)
            if np.dot(self.intersections[0][1] - self.ray.p, self.ray.d) <= 0:
                self.intersections = self.intersections[1:]
            self.on_boundary = True
            self.boundary = poly_node

            self.last_geo_node = self.geo_node
            self.geo_node = None
            if 'det' in poly_node.geo_node.properties:
                self.detector_log.append((self.ray.p.copy(), self.ray.d.copy()))
            return dint
        return 0

    def move_forward(self, t):
        #print 'Move forward', t
        if self.on_boundary:
            self.on_boundary = False
            self.geo_node = self.next_geo_node
            self.next_geo_node = None
        if len(self.intersections):
            poly_node, pint = self.intersections[0]
            geo_lim = np.dot(pint - self.ray.p, self.ray.d)
            if geo_lim < t:
                #print 'Only going', geo_lim
                # Hit an interface!!!
                return self.go_to_interface()
        self.ray.p += self.ray.d*t
        if self.detailed_logging:
            self.detailed_log.append((self.ray.p.copy(), self.ray.d.copy(), self.get_current_level(), 'pos'))
        return t

    def change_direction(self, theta, phi):
        #print 'Change direction'
        model_change = False
        v1 = geo.deflect_vector(self.ray.d, theta, phi)
        if self.on_boundary:
            exit_component = np.dot(v1,self.surface_exit_vector)
            if exit_component < 0:
                self.last_geo_node, self.next_geo_node = self.next_geo_node, self.last_geo_node
                self.surface_exit_vector *= -1.
                model_change = True
            elif exit_component == 0:
                # Oh god why
                # Instead of handling this properly we'll just add some rounding error
                v1 += self.surface_exit_vector * 1e-10 * (np.random.uniform()-0.5)
                v1 /= np.linalg.norm(v1)
        self.ray.d = v1
        self.intersections = self.geometry.ray_trace(self.ray, all_intersections=True)
        if self.on_boundary:
            if len(self.intersections) > 0 and self.intersections[0][0].uid == self.boundary.uid:
                self.intersections = self.intersections[1:]
        if self.detailed_logging:
            self.detailed_log.append((self.ray.p.copy(), self.ray.d.copy(), self.get_current_level(), 'dir'))
        return model_change

    def get_current_level(self):
        if self.on_boundary:
            return self.next_geo_node.level
        else:
            self.intersections = self.geometry.ray_trace(self.ray)
            if len(self.intersections) == 0:
                return -2
            current_node = self.intersections[0][0].geo_node
            uid_map = dict()
            for i in self.intersections:
                uid = i[0].geo_node.uid
                if not uid in uid_map:
                    uid_map[uid] = []
                uid_map[uid].append(i)
            uid = current_node.uid
            n = len(uid_map[uid])
            inside = n%2 == 1
            if not inside:
                current_node = current_node.parent
            if current_node is None:
                return -2


    def get_current_level_old(self):
        node = self.geo_node
        if node is None:
            node = self.next_geo_node
        if node is None:
            level = -2
        else:
            level = node.level
        return level

    def get_material_properties(self):
        if self.geo_node is None:
            geo_node = self.next_geo_node
        else:
            geo_node = self.geo_node
        if geo_node is None:
            props = self.geometry.default_properties
        else:
            props = geo_node.properties
        return props

class Propagator:
    def __init__(self, track, detailed_logging=False):
        self.detailed_logging = detailed_logging
        self.track = track
        self.mass = track.mass
        self.energy = track.energy
        self.Z = 0
        self.N = 0
        self.Cs = 0.05
        self.step_log = []
        self.material_log = []
        self.energy_log = []
        self.detector_log = []
        self.detailed_log = []
        self.update_model()
        pass

    def update_model(self, material_update=True, energy_update=True):
        # Prerequisite to step 2: compute lambdah
        update = not (material_update or energy_update)
        if energy_update and 1-self.track.energy/self.energy >= 0.01:
            self.energy_log.append((self.track.ray.p, self.track.energy))
            update = True
        props = self.track.get_material_properties()
        if material_update and (self.N != props['N'] or self.Z != props['Z']):
            self.material_log.append((self.track.ray.p, props['Z']))
            update = True

        if not update:
            return

        self.energy = self.track.energy
        self.gamma = gamma(self.energy, self.mass)
        self.beta = beta(self.gamma)
        self.Z, self.N = props['Z'], props['N']
        self.A = screening_parameter(self.mass, self.gamma, self.beta, self.Z)
        self.G1W = G1W(self.A)
        self.G2W = G2W(self.A)
        self.sigmaW = sigmaW(self.A, self.Z, self.mass, self.gamma, self.beta)
        self.lambdaW = lambdaW(self.N, self.sigmaW)
        self.lambda1W = lambda1W(self.lambdaW, self.G1W)
        self.lambda2W = lambda2W(self.lambdaW, self.G2W)
        self.lambdah = lambdah(self.lambdaW, self.Cs, self.lambda1W)
        self.mu_cut = mu_cut(self.A, self.lambdah, self.lambdaW)
 
        # Prerequisite to step 5: compute soft lambdas
        self.lambda1Ws = lambda1Ws(self.A, self.lambdaW, self.mu_cut)
        self.lambda2Ws = lambda2Ws(self.A, self.lambdaW, self.mu_cut)

    def propagate_step(self):
        # 1: set the initial position and direction of the particle
        p0 = self.track.ray.p.copy()
        v0 = self.track.ray.d.copy()
        if self.detailed_logging:
            self.detailed_log.append((p0.copy(), v0.copy()))

        #print 'Tracking step starts at:', p0, 'going', v0

        # Update the MSC model quantities is the energy has changed by 1% or more
        if 1-self.track.energy/self.energy >= 0.01:
            self.update_model(material_update=False, energy_update=True)

        #print np.arccos(1-2*self.mu_cut)/np.pi, self.A, self.lambdah, self.lambdaW, self.sigmaW

        # 2: sample the length t of the step for the hard scatter using t=-lambdah*ln(xi)
        xi = np.random.random()
        self.t = -self.lambdah*np.log(xi)

        #print 'Hard scatter after', self.t, 'm'

        # 3: advance the particle tau=t*xi
        xi = np.random.random()
        self.tau = self.t*xi
        #print 'Soft scatter after', self.tau, 'm'
        #p1 = p0 + v0*self.tau
        propagation_length = self.track.move_forward(self.tau)
        
        # 4: check if the track has crossed an interface
        if propagation_length < self.tau:
            # Crossed an interface
            self.update_model(material_update=True)
            return
        

        """
        # 4: check if the track has crossed an interface
        #intersections = self.track.geometry.ray_trace(self.track.ray, all_intersections=False)
        intersections = track.intersections
        crossed = False
        if len(intersections) > 0:
            geo_lim = intersections[0][2]
            if geo_lim <= self.tau:
                # If the track crosses an interface, stop it at the interface
                track.go_to_interface()
                self.track.ray.p = p0 + v0*geo_lim
                if 'det' in intersections[0][0].geo_node.properties:
                    self.detector_log.append((self.track.energy, self.track.ray.p, self.track.ray.d, intersections[0][0].geo_node.properties['det']))
                self.detailed_log.append((self.track.ray.p.copy(), self.track.ray.d.copy()))
                self.track.update()
                self.update_model(material_update=True, energy_update=False)
                return not ((self.track.geo_node is None or self.track.geo_node.level >= -1) and len(self.track.intersections) == 0)
        self.track.ray.p = p1
        """

        # 5: simulate the soft scattering
        self.mu_soft_mean = mu_soft_mean(self.t, self.lambda1Ws)
        self.mu2_soft_mean = mu2_soft_mean(self.t, self.mu_soft_mean, self.lambda2Ws)
        self.a = a(self.mu_soft_mean, self.mu2_soft_mean)
        self.alpha = alpha(self.a)
        self.mu_soft = mu_soft(self.mu_soft_mean, self.alpha)
        self.theta_soft = np.arccos(1-2*self.mu_soft)
        xi = np.random.random()
        self.phi_soft = 2*np.pi*xi
        #print 'soft', self.theta_soft, self.phi_soft

        model_update = self.track.change_direction(self.theta_soft, self.phi_soft)
        if model_update:
            self.update_model(material_update=True)

        """
        # Change direction of the vector
        # Still need to implement
        # Use euler angle rotation function in geometry class, and local coordinate system defined in Plane class
        # alpha = phi, beta = theta
        # z = direction of track
        # x, y from the local "plane" coordinate system
        # Step1: project onto new coordinate system
        # Step2: perform rotation
        # Step3: project back onto original coordinate system
        v1 = geo.deflect_vector(v0, self.theta_soft, self.phi_soft)
        self.track.ray.d = v1
        self.detailed_log.append((self.track.ray.p.copy(), self.track.ray.d.copy()))
        """

        """
        # 6: advance the particle by t-tau
        #p2 = p1 + v1*(self.t - self.tau)
        """
        
        # 6: advance the particle by t-tau
        propagation_length = self.track.move_forward(self.t-self.tau)
        
        # 7: check if the track has crossed an interface
        if propagation_length < self.t-self.tau:
            # Crossed an interface
            self.update_model(material_update=True)
            return
       
        """
        # 7: check if the track has crossed an interface
        intersections = self.track.geometry.ray_trace(self.track.ray, all_intersections=False)
        crossed = False
        if len(intersections) > 0:
            geo_lim = intersections[0][2]
            if geo_lim <= (self.t - self.tau):
                #print 'Hit the geometric limit before hard scatter', geo_lim, self.t-self.tau
                # If the track crosses an interface, stop it at the interface
                #print 'Moving track to boundary before hard scatter'
                self.track.ray.p = p1 + v1*geo_lim
                if 'det' in intersections[0][0].geo_node.properties:
                    self.detector_log.append((self.track.energy, self.track.ray.p, self.track.ray.d, intersections[0][0].geo_node.properties['det']))
                self.detailed_log.append((self.track.ray.p.copy(), self.track.ray.d.copy()))
                #print 'Updating track'
                self.track.update()
                #print 'Updating model'
                self.update_model(material_update=True, energy_update=False)
                #print 'Returning'
                #return self.track.geo_node is not None and self.track.geo_node.level >= -1
                return not ((self.track.geo_node is None or self.track.geo_node.level >= -1) and len(self.track.intersections) == 0)
        self.track.ray.p = p2
        """

        # 8: simulate a hard scatter
        self.mu_hard = mu_hard(self.A, self.mu_cut)
        self.theta_hard = np.arccos(1-2*self.mu_hard)
        xi = np.random.random()
        self.phi_hard = 2*np.pi*xi
        #print 'hard', self.theta_hard, self.phi_hard

        model_update = self.track.change_direction(self.theta_hard, self.phi_hard)
        if model_update:
            self.update_model(material_update=True)
        
        """
        # Change the direction of the vector
        v2 = geo.deflect_vector(v1, self.theta_hard, self.phi_hard)
        self.track.ray.d = v2
        self.detailed_log.append((self.track.ray.p.copy(), self.track.ray.d.copy()))
        return True
        """
    def finish(self):
        if self.detailed_logging:
            self.detailed_log.append((self.track.ray.p.copy(), self.track.ray.d.copy()))
