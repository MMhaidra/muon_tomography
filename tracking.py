import numpy as np
import geometry as geo

hb = 6.582119514e-16 # (GeV*ns)
c = .299792458 # m/ns
c2 = c**2 # m/ns
a0 = 5.2917721067e-11 # m
e2_epsilon0 = 1.8095128e-17 # (m*GeV)
muon_mass_energy = .105658374 # (GeV)
muon_mass = .105658374 / c2 # (GeV/(m/s)^2)

def gamma(energy, mass):
    return energy/(mass*c2)

def beta(gamma):
    return 1-1/gamma**2

def screening_parameter(mass, gamma, beta, Z):
    return 0.25*(hb/(mass*gamma*beta*c))**2/(0.885*Z**(1./3.)*a0)**2

def G1W(A):
    return 2*A*((1+A)*np.log((1+A)/A)-1)

def G2W(A):
    return 6*A*(1+A)*((1+2*A)*np.log((1+A)/A)-2)

def sigmaW(A, Z, mass, gamma, beta):
    return (Z*e2_epsilon0)**2/(mass * gamma * beta**2 * c2)*np.pi/(A*(1+A))

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
    return mu_cut + (A+mu_cut)*xi*(1-mu_cut)/((A+1)-xi*(1-mu_cut))

def lambda1Ws(A, lambdaW, mu_cut):
    return 2/lambdaW * A * (1+A)*(np.log((mu_cut+A)/A)-mu_cut/(mu_cut+A))

def lambda2Ws(A, lambdaW, mu_cut):
    return 6/lambdaW*A*(1+A)*((1+2*A)*np.log((mu_cut+A)/A)-(1+2*A+mu_cut)*mu_cut/(mu_cut+A))

def mu_soft_mean(t, lambda1Ws):
    return 0.5*(1-np.exp(-t/lambda1Ws))

def mu2_soft_mean(t, mu_soft_mean, lambda2Ws):
    return mu_soft_mean - 1/6*(1-np.exp(-t/lambda2Ws))

def a(mu_soft_mean, mu2_soft_mean):
    return (mu2_soft_mean-mu_soft_mean**2)/(mu_soft_mean*(1-mu_soft_mean))

def alpha(a):
    return (3*a+(a*(a+8))**0.5)/(4*(1-a))

def mu_soft(mu_soft_mean, alpha):
    xi = np.random.random()
    comp = 1-mu_soft_mean
    if xi < comp:
        return mu_soft_mean*(xi/(comp))**alpha
    else:
        return 1-comp*((1-xi)/mu_soft_mean)**alpha

class Track:
    def __init__(self, position, direction, mass, energy, geometry):
        self.ray = geo.Ray(position, direction)
        self.mass = mass
        self.energy = energy
        self.geo_node = None
        self.geometry = geometry
        self.update()
    def update(self, intersections=None):
        self.geo_node, self.intersections = self.geometry.get_containing_surface(self.ray, intersections, return_intersections=True)
        if intersections is not None and len(intersections) > 0 and self.intersections[0][2] < 1e-6:
            self.ray.p = self.ray.p + self.ray.v + (self.intersections[0][2]+1e-9)
            self.update(intersections=self.intersections[1:])
    def get_material_properties(self):
        if self.geo_node is None:
            props = self.geometry.default_properties
        else:
            props = self.geo_node.properties
        return props

class Propagator:
    def __init__(self, track):
        self.track = track
        self.mass = track.mass
        self.energy = track.energy
        self.Z = 0
        self.N = 0
        self.Cs = 0.05
        self.update_model()
        pass

    def update_model(self, material_update=True, energy_update=True):
        # Prerequisite to step 2: compute lambdah
        update = not (material_update or energy_update)
        if energy_update and 1-self.track.energy/self.energy >= 0.01:
            update = True
        props = self.track.get_material_properties()
        if material_update and (self.N != props['N'] or self.Z != props['Z']):
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
        v0 = self.track.ray.v.copy()

        # Update the MSC model quantities is the energy has changed by 1% or more
        if 1-self.track.energy/self.energy >= 0.01:
            self.update_model(material_update=False, energy_update=True)

        # 2: sample the length t of the step for the hard scatter using t=-lambdah*ln(xi)
        xi = np.random.random()
        self.t = -self.lambdah*np.log(xi)

        # 3: advance the particle tau=t*xi
        xi = np.random.random()
        self.tau = self.t*xi
        p1 = p0 + v0*self.tau

        # 4: check if the track has crossed an interface
        intersections = geometry.ray_trace(ray, all_intersections=False)
        crossed = False
        if len(intersections) > 0:
            geo_lim = intersections[0][2]
            if geo_lim <= self.tau and geo_lim > 1e-9:
                # If the track crosses an interface, stop it at the interface
                self.track.ray.p = p0 + v0*geo_lim
                self.track.update()
                self.update_model(material_update=True, energy_update=False)
                return
        self.track.ray.p = p1

        # 5: simulate the soft scattering
        self.mu_soft_mean = mu_soft_mean(self.t, self.lambda1Ws)
        self.mu2_soft_mean = mu2_soft_mean(self.t, self.mu_soft_mean, self.lambda2Ws)
        self.a = a(self.mu_soft_mean, self.mu2_soft_mean)
        self.alpha = alpha(a)
        self.mu_soft = mu_hard(self.A, self.mu_cut)
        self.theta_soft = np.arccos(1-2*self.mu_soft)
        xi = np.random.random()
        self.phi_soft = 2*np.pi*xi

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
        self.track.ray.v = v1

        # 6: advance the particle by t-tau
        p2 = p1 + v1*(self.t - self.tau)
        
        # 7: check if the track has crossed an interface
        intersections = geometry.ray_trace(self.track.ray, all_intersections=False)
        crossed = False
        if len(intersections) > 0:
            geo_lim = intersections[0][2]
            if geo_lim <= (self.t - self.tau) and geo_lim > 1e-9:
                # If the track crosses an interface, stop it at the interface
                self.track.ray.p = p1 + v1*geo_lim
                self.track.update()
                self.update_model(material_update=True, energy_update=False)
                return
        self.track.ray.p = p2

        # 8: simulate a hard scatter
        self.mu_hard = mu_hard(self.A, self.mu_cut)
        self.theta_hard = np.arccos(1-2*self.mu_hard)
        xi = np.random.random()
        self.phi_hard = 2*np.pi*xi

        # Change the direction of the vector
        v2 = geo.deflect_vector(v1, self.theta_hard, self.phi_hard)
        self.track.ray.v = v2
