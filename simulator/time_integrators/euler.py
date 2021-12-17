import numpy as np
from ..numerical_flux import flux

"""
Euler integrator use is discouraged, as it's outclassed by RK integrators in DGFEM both by the stability of the time step and performance, 
especially when the integrator is run in parallel 
"""

def euler_integrator(sim, total_time, dt, boundary_index):
    # Time extrapolation Nt steps

    Nt = int(total_time / dt)
    
    Ex_new = np.zeros([sim.Np+1, sim.K])  
    Hy_new = np.zeros([sim.Np+1, sim.K])  
    
    ε_ = np.outer(np.ones(sim.Np+1), sim.ε)
    σ_ = np.outer(np.ones(sim.Np+1), sim.σ)

    𝜇_ = np.outer(np.ones(sim.Np+1), sim.𝜇)
    J_ = np.outer(np.ones(sim.Np+1), sim.J)


    for detector in sim.detectors:
        detector.init_detector(Nt, total_time)

    for iter in range(Nt):            
        Flux = flux(sim.Ex, sim.Hy, sim.Np, sim.K, sim.Z, sim.Y, boundary_index)        
        # Extrapolate each element using flux F 
        
        tmp = ( - sim.Dr @ sim.Hy + sim.Minv @ (Flux[:,:,0] - σ_*sim.Ex))
        Ex_new = dt /(ε_ * J_) * (tmp) + sim.Ex

        tmp = ( - sim.Dr @ sim.Ex + sim.Minv @ Flux[:,:,1])
        Hy_new = dt /(𝜇_ * J_) * ( tmp) + sim.Hy              

        

        sim.Ex = Ex_new 
        sim.Hy = Hy_new
        

        for detector in sim.detectors:
            detector.measure_fields(sim.Ex, sim.Hy, iter)


        # This is the expensive loop that can be parallelized and how the compiled version should be implemented. 
        # The evaluation of each element k is independent
        # Currently being tested
        
        """
        for k in range(0,self.K):
                            
            tmp =  - matrix_mult(self.Dr , self.Hy[:,k]) + matrix_mult(self.Minv , Flux[:,k,0])
            Ex_new[:,k] = dt *1 /(self.ε[k] * self.J[k]) * tmp + self.Ex[:,k]
            
            tmp =  - matrix_mult(self.Dr , self.Ex[:,k]) + matrix_mult(self.Minv , Flux[:,k,1])
            Hy_new[:,k] = dt *1 /(self.𝜇[k] * self.J[k]) * tmp + self.Hy[:,k]                

        self.Ex = Ex_new 
        self.Hy = Hy_new
        """
        
        
        # Only stored for debugging
        # self.Flux = Flux
