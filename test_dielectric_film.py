import numpy as np
from simulator import Simulation1D, um


K = 200  #Number of elements
zpos_vertex = np.linspace(0,10*um,K+1) # z position of the vertices of the elements
Np = 4 #polynomial order

#medium parameters
ε = np.ones(K)
𝜇 = np.ones(K) 
σ = np.zeros(K)

# dielectric film
ε[100:150]= 4

sim = Simulation1D(zpos_vertex, Np ,  ε, 𝜇 , σ,
				   boundaries = 'absorbing_boundaries')





#Set initial conditions
def intial_fields(z):
	sig    = 0.5*um
	z0 = 2.0*um

	Ex = np.exp(-1/sig**2*((z -z0))**2)
	Hy = np.exp(-1/sig**2*((z -z0))**2)
	return Ex, Hy


sim.set_initial_conditions(intial_fields)


#plot fields at t = 0 (time unit is 1um /c seconds ≈ 3.33 fs)
sim.plot_fields(title = "t = 0")



# run simulation 
total_time   = 4
nt = int(3000 * total_time)
dt = total_time/nt
sim.run(total_time, dt)
sim.plot_fields(title = "t = 4")




# run simulation 
total_time   = 5
nt = int(3000 * total_time)
dt = total_time/nt
sim.run(total_time, dt)
sim.plot_fields(title = "t = 9")