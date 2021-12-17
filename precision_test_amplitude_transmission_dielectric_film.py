import numpy as np
from simulator import Simulation1D, Detector, um, fs


# very simple test to measure the non-interference transmittance of a dielectric film and test that the boundary conditions are correctly implemented




K = 200  #Number of elements
zpos_vertex = np.linspace(0,12*um,K+1) # z position of the vertices of the elements
dz = 12*um / K

Np = 4 #polynomial order

#medium parameters
ε = np.ones(K)
𝜇 = np.ones(K) 
σ = np.zeros(K)

# dielectric film

ε_film = 4
n_film = np.sqrt(ε_film)
ε[100:150]= ε_film

sim = Simulation1D(zpos_vertex, Np ,  ε, 𝜇 , σ,
				   boundaries = 'absorbing_boundaries',
				   detectors = [Detector(element_index = int(3*um /dz )), Detector(element_index = int(10*um /dz ))])





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
total_time   = 12
nt = int(500 * total_time)
dt = total_time/nt
sim.run(total_time, dt)
sim.plot_fields(title = "t = 4")




#######################################################################################
# plot the measured field by the detectors

import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.plot(sim.detectors[0].t, sim.detectors[0].Ex, label='Detector at 3 um')
ax1.plot(sim.detectors[1].t, sim.detectors[1].Ex, label='Detector at 10 um')
ax1.legend()
ax1.set_xlabel(' t [um /c]')
ax1.set_ylabel('$E_x$')
ax1.set_title('Detectors measurements')
ax1.grid()

# print the transmittance of the dielectric film

n1 = 1

print(u'analytical transmittance: {}'.format("%.5f"  %  (2*n1/ (n1 + n_film)    *    2*n_film/ (n1 + n_film))      ))
print(u'numerical transmittance {}'.format("%.5f"  %  (np.amax(sim.detectors[1].Ex)/np.amax(sim.detectors[0].Ex))  ))

plt.show()
