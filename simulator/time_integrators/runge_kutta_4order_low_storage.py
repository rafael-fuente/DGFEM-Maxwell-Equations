import numpy as np
from ..numerical_flux import flux
from ..util.LSERK4_coefficients import rk4a , rk4b , rk4c
from ..detector import Detector

"""
The main difference of LSRK4 with respect to the classical RK4 is that only one additional storage level for the k variables is required, thus reducing
the memory usage significantly. On the other hand, this comes at the price of
an additional function evaluation, as the low-storage version has five stages. The added cost in the low-storage RK is offset by
allowing a larger stable timestep, ∆t, very relevant regarding the GPU / parallel 2D / 3D implementation, as fewer synchronization steps are required
to achieve the same precision.

Reference: https://ntrs.nasa.gov/citations/19940028444

"""

def lserk4_integrator(sim, total_time, dt, boundary_index):
    # Time extrapolation Nt steps

    Nt = int(total_time / dt)
        
    ε_ = np.outer(np.ones(sim.Np+1), sim.ε)
    σ_ = np.outer(np.ones(sim.Np+1), sim.σ)

    𝜇_ = np.outer(np.ones(sim.Np+1), sim.𝜇)
    J_ = np.outer(np.ones(sim.Np+1), sim.J)

    t = sim.t

    k_Ex = np.zeros([sim.Np+1, sim.K])
    k_Hy = np.zeros([sim.Np+1, sim.K])

    for detector in sim.detectors:
        detector.init_detector(Nt, total_time)

    for iter in range(Nt):            
               
        for i in range(5):
            # evaluate the five stages of lserk4

            Flux = flux(sim.Ex, sim.Hy, sim.Np, sim.K, sim.Z, sim.Y, boundary_index) 
            k_Ex = rk4a[i]*k_Ex + dt /(ε_ * J_) * ( - sim.Dr @ sim.Hy + sim.Minv @ (Flux[:,:,0] - σ_*sim.Ex  - sim.current_sources.get_current(t + rk4c[i]*dt)))
            k_Hy = rk4a[i]*k_Hy + dt /(𝜇_ * J_) * ( - sim.Dr @ sim.Ex + sim.Minv @ Flux[:,:,1])
              
            sim.Ex = sim.Ex+rk4b[i]*k_Ex
            sim.Hy = sim.Hy+rk4b[i]*k_Hy


        for detector in sim.detectors:
            detector.measure_fields(sim.Ex, sim.Hy, iter)

        t += dt

    sim.t = t
