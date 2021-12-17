import numpy as np
from ..numerical_flux import flux

def rk2_integrator(sim, total_time, dt, boundary_index):
    # Time extrapolation Nt steps

    Nt = int(total_time / dt)
        
    ε_ = np.outer(np.ones(sim.Np+1), sim.ε)
    σ_ = np.outer(np.ones(sim.Np+1), sim.σ)

    𝜇_ = np.outer(np.ones(sim.Np+1), sim.𝜇)
    J_ = np.outer(np.ones(sim.Np+1), sim.J)


    k1_Ex = np.zeros([sim.Np+1, sim.K])
    k1_Hy = np.zeros([sim.Np+1, sim.K])

    k2_Ex = np.zeros([sim.Np+1, sim.K])
    k2_Hy = np.zeros([sim.Np+1, sim.K])

    for detector in sim.detectors:
        detector.init_detector(Nt, total_time)


    for iter in range(Nt):            
        Flux = flux(sim.Ex, sim.Hy, sim.Np, sim.K, sim.Z, sim.Y, boundary_index)        
        # Extrapolate each element using flux F 
        
        k1_Ex = dt /(ε_ * J_) * ( - sim.Dr @ sim.Hy + sim.Minv @ (Flux[:,:,0] - σ_*sim.Ex))
        k1_Hy = dt /(𝜇_ * J_) * ( - sim.Dr @ sim.Ex + sim.Minv @ Flux[:,:,1])             


        Ex_plus_k1 = sim.Ex + k1_Ex
        Hy_plus_k1 = sim.Hy + k1_Hy


        Flux = flux(Ex_plus_k1, Hy_plus_k1, sim.Np, sim.K, sim.Z, sim.Y, boundary_index)  

        k2_Ex = dt /(ε_ * J_) * ( - sim.Dr @ Hy_plus_k1 + sim.Minv @ (Flux[:,:,0] - σ_*Ex_plus_k1))
        k2_Hy = dt /(𝜇_ * J_) * ( - sim.Dr @ Ex_plus_k1 + sim.Minv @ Flux[:,:,1])             

        sim.Ex += (k1_Ex + k2_Ex)/2. 
        sim.Hy += (k1_Hy + k2_Hy)/2. 

        
        for detector in sim.detectors:
            detector.measure_fields(sim.Ex, sim.Hy, iter)
