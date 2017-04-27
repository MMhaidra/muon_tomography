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

class Propagator:
    def __init__(self):
        self.Cs = 0.05
        pass

    def propagate_step(self, track, geometry):
        self.Z, self.N = get_material_properties(track, geometry) # needs to be implemented
        # 1: set the initial position and direction of the particle
        p0 = track.ray.p.copy()
        v0 = track.ray.v.copy()

        # prereq 2: compute lambdah
        self.energy = track.energy
        self.mass = track.mass
        self.gamma = gamma(self.energy, self.mass)
        self.beta = beta(self.gamma)
        self.A = screening_parameter(self.mass, self.gamma, self.beta, self.Z)
        self.G1W = G1W(self.A)
        self.G2W = G2W(self.A)
        self.sigmaW = sigmaW(self.A, self.Z, self.mass, self.gamma, self.beta)
        self.lambdaW = lambdaW(self.N, self.sigmaW)
        self.lambda1W = lambda1W(self.lambdaW, self.G1W)
        self.lambda2W = lambda2W(self.lambdaW, self.G2W)
        self.lambdah = lambdah(self.lambdaW, self.Cs, self.lambda1W)
        self.mu_cut = mu_cut(self.A, self.lambdah, self.lambdaW)

        # 2: sample the length t of the step for the hard scatter using t=-lambdah*ln(xi)
        xi = np.random.random()
        self.t = -self.lambdah*np.log(xi)

        # 3: advance the particle tau=t*xi
        xi = np.random.random()
        self.tau = self.t*xi
        p1 = p0 + v0*self.tau
        track.ray.p = p1

        # 4: check if the track has crossed an interface
        intersections = geometry.ray_trace(ray, all_intersections=False)
        crossed = False
        if len(intersections) > 0:
            geo_lim = intersections[0][2]
            if geo_lim <= self.tau:
                track.ray.p = p0 + v0*geo_lim
                return track

        # prereq 5
        self.lambda1Ws = lambda1Ws(self.A, self.lambdaW, self.mu_cut)
        self.lambda2Ws = lambda2Ws(self.A, self.lambdaW, self.mu_cut)
        
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
        v1 = None
        track.ray.v = v1

        # 6: advance the particle by t-tau
        p2 = p1 + v1*(self.t - self.tau)
        track.ray.p = p2
        
        # 7: check if the track has crossed an interface
        intersections = geometry.ray_trace(track.ray, all_intersections=False)
        crossed = False
        if len(intersections) > 0:
            geo_lim = intersections[0][2]
            if geo_lim <= (self.t - self.tau):
                track.ray.p = p1 + v1*geo_lim
                return track

        # 8: simulate a hard scatter
        self.mu_hard = mu_hard(self.A, self.mu_cut)
        self.theta_hard = np.arccos(1-2*self.mu_hard)
        xi = np.random.random()
        self.phi_hard = 2*np.pi*xi

        # Change the direction of the vector
        # Still need to implement
        v2 = None
        track.ray.v = v2
        
        return track
