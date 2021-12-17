import numpy as np
import numba

@numba.jit("(f8[:,:], f8[:,:], i8, i8, f8[:], f8[:], f8[:,:,:])", nopython=True, nogil=True)
def periodic_boundaries_flux(Ex, Hy, Np, K, Z,Y, out_flux):

    k = 0
    #left boundary of the element

    E_plus = Ex[0,k]
    H_plus = Hy[0,k]

    Z_plus = Z[k]
    Y_plus = Y[k]

    E_minus = Ex[Np,k-1]
    H_minus = Hy[Np,k-1]

    Z_minus = Z[k-1]
    Y_minus = Y[k-1]

    H_star = E_minus/(Z_minus + Z_plus) - E_plus/(Z_minus + Z_plus) + H_minus*Z_minus/(Z_minus + Z_plus) + H_plus*Z_plus/(Z_minus + Z_plus)
    E_star = E_minus*Y_minus/(Y_minus + Y_plus) + E_plus*Y_plus/(Y_minus + Y_plus) + H_minus/(Y_minus + Y_plus) - H_plus/(Y_minus + Y_plus)
    out_flux[0,k,0] = -(H_plus-H_star)
    out_flux[0,k,1] = -(E_plus-E_star)



    #right boundary of the element

    E_plus = Ex[Np,k]
    H_plus = Hy[Np,k]


    Z_plus = Z[k]
    Y_plus = Y[k]

    E_minus = Ex[0, k+1]
    H_minus = Hy[0, k+1]

    Z_minus = Z[k+1]
    Y_minus = Y[k+1]

    H_star = -E_minus/(Z_minus + Z_plus) + E_plus/(Z_minus + Z_plus) + H_minus*Z_minus/(Z_minus + Z_plus) + H_plus*Z_plus/(Z_minus + Z_plus)
    E_star = E_minus*Y_minus/(Y_minus + Y_plus) + E_plus*Y_plus/(Y_minus + Y_plus) - H_minus/(Y_minus + Y_plus) + H_plus/(Y_minus + Y_plus)

    out_flux[Np,k,0] = (H_plus-H_star)
    out_flux[Np,k,1] = (E_plus-E_star)



    k = K - 1
    #left boundary of the element

    E_plus = Ex[0,k]
    H_plus = Hy[0,k]

    Z_plus = Z[k]
    Y_plus = Y[k]

    E_minus = Ex[Np,k-1]
    H_minus = Hy[Np,k-1]

    Z_minus = Z[k-1]
    Y_minus = Y[k-1]

    H_star = E_minus/(Z_minus + Z_plus) - E_plus/(Z_minus + Z_plus) + H_minus*Z_minus/(Z_minus + Z_plus) + H_plus*Z_plus/(Z_minus + Z_plus)
    E_star = E_minus*Y_minus/(Y_minus + Y_plus) + E_plus*Y_plus/(Y_minus + Y_plus) + H_minus/(Y_minus + Y_plus) - H_plus/(Y_minus + Y_plus)
    out_flux[0,k,0] = -(H_plus-H_star)
    out_flux[0,k,1] = -(E_plus-E_star)



    #right boundary of the element

    E_plus = Ex[Np,k]
    H_plus = Hy[Np,k]


    Z_plus = Z[k]
    Y_plus = Y[k]

    E_minus = Ex[0, k+1]
    H_minus = Hy[0, k+1]

    Z_minus = Z[k+1]
    Y_minus = Y[k+1]

    H_star = -E_minus/(Z_minus + Z_plus) + E_plus/(Z_minus + Z_plus) + H_minus*Z_minus/(Z_minus + Z_plus) + H_plus*Z_plus/(Z_minus + Z_plus)
    E_star = E_minus*Y_minus/(Y_minus + Y_plus) + E_plus*Y_plus/(Y_minus + Y_plus) - H_minus/(Y_minus + Y_plus) + H_plus/(Y_minus + Y_plus)

    out_flux[Np,k,0] = (H_plus-H_star)
    out_flux[Np,k,1] = (E_plus-E_star)
