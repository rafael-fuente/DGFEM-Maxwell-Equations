import numpy as np
import numba

@numba.jit("(f8[:,:], f8[:,:], i8, i8, f8[:], f8[:], f8[:,:,:])", nopython=True, nogil=True)
def absorbing_boundaries_flux(Ex, Hy, Np, K, ε, 𝜇, out_flux):

	#left boundary

	k = 0
	E_plus = Ex[0,k]
	H_plus = Hy[0,k]

	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]

	Z_plus = np.sqrt(𝜇_plus/ε_plus)

	H_star = -E_plus/Z_plus + H_plus
	E_star = 0
	out_flux[0,k,0] = -(H_plus-H_star)
	out_flux[0,k,1] = -(E_plus-E_star)



	#right boundary

	E_plus = Ex[Np,k]
	H_plus = Hy[Np,k]


	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]


	Z_plus = np.sqrt(𝜇_plus/ε_plus)

	H_star = E_plus/Z_plus + H_plus
	E_star = 0

	out_flux[Np,k,0] = (H_plus-H_star)
	out_flux[Np,k,1] = (E_plus-E_star)




	#left boundary

	k = K-1
	E_plus = Ex[0,k]
	H_plus = Hy[0,k]

	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]

	Z_plus = np.sqrt(𝜇_plus/ε_plus)

	H_star = -E_plus/Z_plus + H_plus
	E_star = 0
	out_flux[0,k,0] = -(H_plus-H_star)
	out_flux[0,k,1] = -(E_plus-E_star)



	#right boundary

	E_plus = Ex[Np,k]
	H_plus = Hy[Np,k]


	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]


	Z_plus = np.sqrt(𝜇_plus/ε_plus)

	H_star = E_plus/Z_plus + H_plus
	E_star = 0

	out_flux[Np,k,0] = (H_plus-H_star)
	out_flux[Np,k,1] = (E_plus-E_star)
