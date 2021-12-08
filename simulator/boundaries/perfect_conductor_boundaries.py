import numpy as np
import numba

@numba.jit("(f8[:,:], f8[:,:], i8, i8, f8[:], f8[:], f8[:,:,:])", nopython=True, nogil=True)
def perfect_conductor_boundaries_flux(Ex, Hy, Np, K, ε, 𝜇, out_flux):

	k = 0
	#left boundary of the element

	E_plus = Ex[0,k]
	H_plus = Hy[0,k]

	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]

	Z_plus = np.sqrt(𝜇_plus/ε_plus)

	H_star = -E_plus/Z_plus + H_plus
	E_star = 0
	out_flux[0,k,0] = -(H_plus-H_star)
	out_flux[0,k,1] = -(E_plus-E_star)



	#right boundary of the element

	E_plus = Ex[Np,k]
	H_plus = Hy[Np,k]


	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]

	E_minus = Ex[0, k+1]
	H_minus = Hy[0, k+1]

	ε_minus = ε[k+1] 
	𝜇_minus = 𝜇[k+1]

	Z_minus = np.sqrt(𝜇_minus /ε_minus )
	Z_plus = np.sqrt(𝜇_plus/ε_plus)
	Y_minus = np.sqrt(ε_minus/𝜇_minus)
	Y_plus = np.sqrt(ε_plus/𝜇_plus)

	H_star = -E_minus/(Z_minus + Z_plus) + E_plus/(Z_minus + Z_plus) + H_minus*Z_minus/(Z_minus + Z_plus) + H_plus*Z_plus/(Z_minus + Z_plus)
	E_star = E_minus*Y_minus/(Y_minus + Y_plus) + E_plus*Y_plus/(Y_minus + Y_plus) - H_minus/(Y_minus + Y_plus) + H_plus/(Y_minus + Y_plus)

	out_flux[Np,k,0] = (H_plus-H_star)
	out_flux[Np,k,1] = (E_plus-E_star)



	k = K - 1
	#left boundary of the element

	E_plus = Ex[0,k]
	H_plus = Hy[0,k]

	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]

	E_minus = Ex[Np,k-1]
	H_minus = Hy[Np,k-1]

	ε_minus = ε[k-1] 
	𝜇_minus = 𝜇[k-1]

	Z_minus = np.sqrt(𝜇_minus /ε_minus )
	Z_plus = np.sqrt(𝜇_plus/ε_plus)
	Y_minus = np.sqrt(ε_minus/𝜇_minus)
	Y_plus = np.sqrt(ε_plus/𝜇_plus)

	H_star = E_minus/(Z_minus + Z_plus) - E_plus/(Z_minus + Z_plus) + H_minus*Z_minus/(Z_minus + Z_plus) + H_plus*Z_plus/(Z_minus + Z_plus)
	E_star = E_minus*Y_minus/(Y_minus + Y_plus) + E_plus*Y_plus/(Y_minus + Y_plus) + H_minus/(Y_minus + Y_plus) - H_plus/(Y_minus + Y_plus)
	out_flux[0,k,0] = -(H_plus-H_star)
	out_flux[0,k,1] = -(E_plus-E_star)



	#right boundary of the element

	E_plus = Ex[Np,k]
	H_plus = Hy[Np,k]


	ε_plus = ε[k] 
	𝜇_plus = 𝜇[k]


	Z_plus = np.sqrt(𝜇_plus/ε_plus)

	H_star = E_plus/Z_plus + H_plus
	E_star = 0

	out_flux[Np,k,0] = (H_plus-H_star)
	out_flux[Np,k,1] = (E_plus-E_star)
