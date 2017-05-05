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
    def __init__(self, position, direction, mass, energy, geometry):
        self.ray = geo.Ray(position, direction)
        self.mass = mass
        self.energy = energy
        self.intersections = None
        self.on_boundary = False
        self.geometry = geometry
        self.last_geo_node = None
        self.next_geo_node = None
        self.surface_exit_vector = None
        self.boundary = None
        self.detector_log = []
        self.geo_node = self.get_containing_surface()

    def go_to_interface(self):
        # Check that there is an interface to go to
        if len(self.intersections) > 0:
            # Get the interface to move to
            poly_node, pint = self.intersections[0]
            dint = np.dot(pint - self.ray.p, self.ray.d)
            geo_node = poly_node.geo_node
                
            current_level = self.get_current_level()

            going_in = geo_node.level > current_level
            going_out = geo_node.level == current_level
            nonsense = geo_node.level < current_level

            if nonsense:
                print pint
                new_intersections = self.geometry.ray_trace(self.ray)
                print self.intersections
                print new_intersections
                raise ValueError('Next geo level is larger than current geo level')
            elif going_in:
                self.next_geo_node = geo_node
            elif going_out:
                self.next_geo_node = geo_node.parent

            # Store the exit vector
            self.surface_exit_vector = np.dot(self.ray.d, poly_node.poly.plane.vector)*poly_node.poly.plane.vector
            self.surface_exit_vector /= np.linalg.norm(self.surface_exit_vector)

            # Update status information
            self.ray.p = pint
            self.intersections = self.intersections[1:]
            self.on_boundary = True
            self.boundary = poly_node
            self.last_geo_node = self.geo_node
            self.geo_node = None

            # Log if the interface is part of a detector
            if 'det' in poly_node.geo_node.properties:
                self.detector_log.append((self.ray.p.copy(), self.ray.d.copy()))
            return dint
        return 0

    def move_forward(self, t):
        if self.on_boundary:
            # Track is moving into the volume in front of it
            self.on_boundary = False
            self.geo_node = self.next_geo_node
            self.next_geo_node = None

        if len(self.intersections):
            # Check if the track hits an interface
            poly_node, pint = self.intersections[0]
            geo_lim = np.dot(pint - self.ray.p, self.ray.d)
            if geo_lim < t:
                # Hit an interface!!!
                # Move the track to the interface instead of the full length
                return self.go_to_interface()
        self.ray.p += self.ray.d*t
        return t

    def change_direction(self, theta, phi):
        model_change = False
        v1 = geo.deflect_vector(self.ray.d, theta, phi)
        if self.on_boundary:
            # If the track is on a boundary, update the next volume appropriately
            exit_component = np.dot(v1,self.surface_exit_vector)
            if exit_component < 0:
                self.last_geo_node, self.next_geo_node = self.next_geo_node, self.last_geo_node
                self.surface_exit_vector *= -1.
                model_change = True
            elif exit_component == 0:
                # Oh god why
                # The new direction is exactly in the plane of the polygon
                # Instead of handling this properly we'll just add some rounding error
                v1 += self.surface_exit_vector * 1e-10 * (np.random.uniform()-0.5)
                v1 /= np.linalg.norm(v1)

        # Update the direction and intersections
        self.ray.d = v1
        self.intersections = self.geometry.ray_trace(self.ray, all_intersections=True)
        if self.on_boundary:
            if len(self.intersections) > 0 and self.intersections[0][0].uid == self.boundary.uid:
                # Remove the first intersection if the track is on it
                self.intersections = self.intersections[1:]
        return model_change

    def get_containing_surface(self):
        if self.on_boundary:
            # The volume the track cares about is the one it is going in to
            return self.next_geo_node
        else:
            # Update the intersections if we need to
            if self.intersections is None:
                self.intersections = self.geometry.ray_trace(self.ray)
            # No intersections means the track is in empy space
            if len(self.intersections) == 0:
                return None
            current_node = self.intersections[0][0].geo_node
            uid_map = dict()
            for i in self.intersections:
                uid = i[0].geo_node.uid
                if not uid in uid_map:
                    uid_map[uid] = []
                uid_map[uid].append(i)
            uid = current_node.uid
            n = len(uid_map[uid])
            # An odd number of intersections means the track is inside the surface
            inside = n%2 == 1
            if not inside:
                # An even number of intersections means the next surface is the child of the current volume
                current_node = current_node.parent
            return current_node

    def get_current_level(self):
        containing_surface = self.get_containing_surface()
        if containing_surface is None:
            return -1
        else:
            return containing_surface.level

    def get_material_properties(self):
        containing_surface = self.get_containing_surface()
        if containing_surface is None:
            return self.geometry.default_properties
        else:
            return containing_surface.properties

class Propagator:
    def __init__(self, track):
        self.Z = 0
        self.N = 0
        self.Cs = 0.05
        self.reinit(track)

    def reinit(self, track):
        self.track = track
        self.mass = track.mass
        self.energy = track.energy
        self.step_log = []
        self.material_log = []
        self.energy_log = []
        self.detector_log = []
        self.update_model()

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

        # Update the MSC model quantities is the energy has changed by 1% or more
        if 1-self.track.energy/self.energy >= 0.01:
            self.update_model(material_update=False, energy_update=True)

        # 2: sample the length t of the step for the hard scatter using t=-lambdah*ln(xi)
        xi = np.random.random()
        self.t = -self.lambdah*np.log(xi)

        # 3: advance the particle tau=t*xi
        xi = np.random.random()
        self.tau = self.t*xi
        propagation_length = self.track.move_forward(self.tau)
        
        # 4: check if the track has crossed an interface
        if propagation_length < self.tau:
            # Crossed an interface
            self.update_model(material_update=True)
            return

        # 5: simulate the soft scattering
        self.mu_soft_mean = mu_soft_mean(self.t, self.lambda1Ws)
        self.mu2_soft_mean = mu2_soft_mean(self.t, self.mu_soft_mean, self.lambda2Ws)
        self.a = a(self.mu_soft_mean, self.mu2_soft_mean)
        self.alpha = alpha(self.a)
        self.mu_soft = mu_soft(self.mu_soft_mean, self.alpha)
        self.theta_soft = np.arccos(1-2*self.mu_soft)
        xi = np.random.random()
        self.phi_soft = 2*np.pi*xi

        model_update = self.track.change_direction(self.theta_soft, self.phi_soft)
        if model_update:
            self.update_model(material_update=True)

        # 6: advance the particle by t-tau
        propagation_length = self.track.move_forward(self.t-self.tau)
        
        # 7: check if the track has crossed an interface
        if propagation_length < self.t-self.tau:
            # Crossed an interface
            self.update_model(material_update=True)
            return

        # 8: simulate a hard scatter
        self.mu_hard = mu_hard(self.A, self.mu_cut)
        self.theta_hard = np.arccos(1-2*self.mu_hard)
        xi = np.random.random()
        self.phi_hard = 2*np.pi*xi

        model_update = self.track.change_direction(self.theta_hard, self.phi_hard)
        if model_update:
            self.update_model(material_update=True)

    def finish(self):
        pass
