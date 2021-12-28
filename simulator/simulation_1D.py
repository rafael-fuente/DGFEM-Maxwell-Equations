import numpy as np
from .polynomials.gll import gll
from .polynomials.lagrange1st import lagrange1st
import matplotlib.pyplot as plt
from .current_sources import CurrentSources

from .time_integrators import euler_integrator, rk2_integrator, lserk4_integrator


class Simulation1D:
    def __init__(self, zpos_vertex, Np , ε, 𝜇 , σ, boundaries = 'perfect_conductor_boundaries', detectors = [] , current_sources = CurrentSources([],[],[],[])):
        
        self.Np = Np #polynomial order
        self.Vx = zpos_vertex # z position of the vertices of the elements
        self.Nv = len(self.Vx) # Number of vertices
        self.K = self.Nv - 1 #Number of elements
        
        # physical parameters of the medium
        self.ε, self.𝜇 , self.σ = ε, 𝜇 , σ 
        self.Z = np.sqrt(self.𝜇/self.ε) # impedance
        self.Y = 1/self.Z # admittance


        dz_k = np.diff(self.Vx) #length of each element
        
        
        z_gll,w_gll = np.array(gll(self.Np))     # z, Np+1 coordinates [-1 1] of GLL points
                                                # w Integration weights at GLL points
            
        Dz_lj_zi = lagrange1st(self.Np)        #first derivative of lagrange node at GLL points (diff( l_i(z_j) , z))
        self.Dr =  Dz_lj_zi.T # Differentiation matrix (Dr). Currently, Dr remplaces Stiffness matrix for faster time iterations
        
        self.J = dz_k/2 #Jacobian of each element
        self.J_inv = 1/self.J  #Jacobian inverse of each element

        z_center = self.Vx[0:self.K] + dz_k/2 # z center position of each element
        z_local = np.outer(z_gll, np.ones(self.K))  * self.J # local coordinates of lagrange nodes.
        
        self.z_center = z_center
        self.z_local = z_local

        #global coordinates of lagrange nodes. Columns-> k element index, Rows-> lagrange node index
        self.z =   np.outer( np.ones(Np+1), z_center)  +  z_local   

        #can be used for easily plotting the fields
        self.z_grid = (self.z.T).flatten()
        
        
        # -----------------------------------------------------------------

        
        # Initialization of system matrices

        # Elemental reduced Mass matrix (M)
        self.M = np.zeros((self.Np+1, self.Np+1))
        for i in range(0, self.Np+1):
            self.M[i, i] = w_gll[i]

        # Inverse matrix of M (M is diagonal)
        self.Minv = np.identity(self.Np+1)

        for i in range(0, self.Np+1):
            self.Minv[i,i] = 1. / self.M[i,i]

            
        
        self.S = np.zeros((self.Np+1, self.Np+1))

        # Elemental Stiffness Matrix (S) Currenly unused. I use instead the Differentiation matrix Dr
        for i in range(0, self.Np+1):
            for j in range(0, self.Np+1):
                self.S[i,j] =  w_gll[i] * Dz_lj_zi[j, i]


        # -----------------------------------------------------------------

        # Boundaries

        #for readability in the C++ version boundary index should be a enumeration, but for numba jit compiler isn't an equivalent, so we use integers.

        implemented_boundaries = ('absorbing_boundaries', 'perfect_conductor_boundaries', 'periodic_boundaries')


        if boundaries == 'absorbing_boundaries':
            self.boundary_index = 0

        elif boundaries == 'perfect_conductor_boundaries':
            self.boundary_index = 1

        elif boundaries == 'periodic_boundaries':
            self.boundary_index = 2
        else:
            raise NotImplementedError(
                f"{implemented_boundaries} has not been implemented. Use one of {implemented_boundaries}")

        # -----------------------------------------------------------------

        # Detectors
        
        self.detectors = detectors

        # -----------------------------------------------------------------

        # Sources

        self.current_sources = current_sources
        self.current_sources.init_sources(self)

        # total time simulated
        self.t = 0


    def set_initial_conditions(self, initial_fields_function):

        self.Ex , self.Hy = initial_fields_function(self.z)

                
    def run(self, total_time, dt, time_integrator = 'LSERK4'):

        # Time extrapolation Nt steps

        implemented_integrators = ('Euler', 'RK2', 'LSERK4')


        if time_integrator == 'Euler':
            euler_integrator(self, total_time, dt, self.boundary_index)
        elif time_integrator == 'RK2':
            rk2_integrator(self, total_time, dt, self.boundary_index)
        elif time_integrator == 'LSERK4': 
            lserk4_integrator(self, total_time, dt, self.boundary_index)
        else:
            raise NotImplementedError(
                f"{time_integrator} has not been implemented. Use one of {implemented_integrators}")

    from .plotting import plot_fields 