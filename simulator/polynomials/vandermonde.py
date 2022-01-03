import numpy as np
from jacobi import jacobiP

def Vandermonde1D(N,r):
    # Purpose : Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i);

    V1D = np.zeros((len(r),N+1))
    
    for j in range(0,N+1):
        V1D[:,j] = jacobiP(r, 0, 0, j)
    return V1D
