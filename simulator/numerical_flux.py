import numpy as np
import numba

from .boundaries import absorbing_boundaries_flux, perfect_conductor_boundaries_flux, periodic_boundaries_flux


@numba.jit("f8[:,:,:](f8[:,:], f8[:,:], i8, i8, f8[:], f8[:], i8)", nopython=True, nogil=True)
def flux(Ex, Hy, Np, K, ε, 𝜇, boundary_index):
    """
    calculates the flux betwçeen two boundary sides of 
    connected elements for element k
    """
    # for every element we have 2 faces to other elements (left and right)
    out_flux = np.zeros((Np+1,K,2)) 


    # Calculate Fluxes inside domain
    for k in range(1, K-1):
        
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
        
        
    #for readability in the C++ version boundary index should be a enumeration, but for numba jit compiler isn't an equivalent, so we use integers.
    if boundary_index == 0:
        absorbing_boundaries_flux(Ex, Hy, Np, K, ε, 𝜇, out_flux)

    elif boundary_index == 1:
        perfect_conductor_boundaries_flux(Ex, Hy, Np, K, ε, 𝜇, out_flux)

    elif boundary_index == 2:
        periodic_boundaries_flux(Ex, Hy, Np, K, ε, 𝜇, out_flux)
    
    
    return out_flux