import numpy as np
from simulator import Simulation1D, Detector, CurrentSources, um, fs


# very simple test to measure the non-interference transmittance of a dielectric film and test that the boundary conditions are correctly implemented




K = 200  #Number of elements
zpos_vertex = np.linspace(0,20*um,K+1) # z position of the vertices of the elements
dz = 20*um / K

Np = 4 #polynomial order

#medium parameters
ε = np.ones(K)
𝜇 = np.ones(K) 
σ = np.zeros(K)

sim = Simulation1D(zpos_vertex, Np ,  ε, 𝜇 , σ,
				   boundaries = 'absorbing_boundaries', 
				   current_sources = CurrentSources(element_indexes = [int(5.*um /dz ),    int(15.*um /dz )], 
				   								    frequencies = [1. / (1.*um) ,         1. / (0.5*um)], 
				   								    phases = [0.,  0.], 
				   								    J0 = [0.2 ,  0.2]                        ))
				   





#Set initial conditions
def intial_fields(z):
	Ex = np.zeros_like(z)
	Hy = np.zeros_like(z)
	return Ex, Hy


sim.set_initial_conditions(intial_fields)


#plot fields at t = 0 (time unit is 1um /c seconds ≈ 3.33 fs)
sim.plot_fields(title = "t = 0")



# run simulation 
total_time   = 1.0
nt = int(500 * total_time)
dt = total_time/nt
sim.run(total_time, dt , time_integrator = 'LSERK4')
sim.plot_fields(title = "t = 1.0")



# run simulation 
total_time   = 1.5
nt = int(500 * total_time)
dt = total_time/nt
sim.run(total_time, dt , time_integrator = 'LSERK4')
sim.plot_fields(title = "t = 2.5")
