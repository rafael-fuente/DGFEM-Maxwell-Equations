import numpy as np
from simulator import Simulation1D, Detector, um, fs, c


#==========================================================================================================================================================================
# Measurement of the transmittance spectra of a thin film
#
# The spectral transmittance is computed by taking the FFT of the incident and transmitted pulse. Simulation results are compared with the analytical expression.
#==========================================================================================================================================================================


K = 200  #Number of elements
zpos_vertex = np.linspace(0,10*um,K+1) # z position of the vertices of the elements
dz = 10*um / K

Np = 4 #polynomial order



#medium parameters
蔚 = np.ones(K)
饾渿 = np.ones(K) 
蟽 = np.zeros(K)


# dielectric film
n_film = 1.5
蔚_film = n_film**2
width_film = dz * 10
蔚[100 : (100 + 10)] = 蔚_film


# set up the simulation
sim = Simulation1D(zpos_vertex, Np ,  蔚, 饾渿 , 蟽,
				   boundaries = 'absorbing_boundaries',
				   detectors = [Detector(element_index = int(9*um /dz ))])



#Set initial conditions (we set an initial pulse with a bandwidth 螖位 and center wavelength 位0)

螖位 = 2.*um
螖f = c/螖位

位0 = 0.8*um
f0 = c/位0

def intial_fields(z):
	z0 = 3.0*um
	Ex = np.exp(-2*np.pi**2*(z-z0)**2*螖f**2)*np.cos(2*np.pi*f0*(z-z0))
	Hy = np.exp(-2*np.pi**2*(z-z0)**2*螖f**2)*np.cos(2*np.pi*f0*(z-z0))
	return Ex, Hy

sim.set_initial_conditions(intial_fields)


#plot fields at t = 0 (time unit is 1um /c seconds 鈮? 3.33 fs)
sim.plot_fields(title = "t = 0")


# run simulation 
total_time   = 15
nt = int(500 * total_time)
dt = total_time/nt
sim.run(total_time, dt)




########################################################################################

# compute the transmittance spectra   


#transmitted pulse Ex spectrum measured at z = 9*um
f , spectrum_transmitted_pulse = sim.detectors[0].get_Ex_spectrum()

#incident pulse Ex spectrum (Fourier transform of the incident pulse)
spectrum_incident_pulse = np.abs(np.exp(- 1/2 * ((f - f0)/ 螖f)**2)  / (2*螖f * np.sqrt(2*np.pi))  +   np.exp(- 1/2 * ((-f - f0)/ 螖f)**2)  / (2*螖f * np.sqrt(2*np.pi)))


# get the bandwidth we are interested (otherwise we get a 0 division error when computing T)
mask = (f< c/(0.2*um)) & (f> c/(1.5*um))

f = np.extract(mask, f)
spectrum_incident_pulse = np.extract(mask, spectrum_incident_pulse)
spectrum_transmitted_pulse = np.extract(mask, spectrum_transmitted_pulse)

位 = 1/f
T = (np.abs(spectrum_transmitted_pulse/spectrum_incident_pulse))**2





# plot the transmittance spectra   

import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot()

n1 = 1
ax1.plot(位, 1/((n1/n_film - n_film/n1)**2*np.sin(width_film*n_film*2*np.pi/位)**2/4 + 1), label = 'analytical transmittance')
ax1.plot(位,  T , "--", label = 'simulation transmittance')

ax1.set_ylim([0.0, 1])
ax1.set_xlim([0.3*um, 1.3*um])
ax1.set_xlabel('位 [um]')
ax1.set_ylabel('T(位)')

ax1.legend()
ax1.grid()
plt.show()
