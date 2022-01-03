import numpy as np
import scipy.special

def jacobiP(x,alpha,beta,N): 
    # Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
    #          (alpha+beta <> -1) at points x for order N
    # Note   : They are normalized to be orthonormal.
    
    xp = x

    PL = np.zeros((N + 1,len(xp)))
    # Initial values P_0(x) and P_1(x)
    gamma0 = 2 ** (alpha + beta + 1) / (alpha + beta + 1) * scipy.special.gamma(alpha + 1) * scipy.special.gamma(beta + 1) / scipy.special.gamma(alpha + beta + 1)
    PL[0,:] = 1.0 / np.sqrt(gamma0)

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1,:] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)


    # Repeat value in recurrence.
    aold = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        bnew = - (alpha ** 2 - beta ** 2) / h1 / (h1 + 2)
        PL[i + 1,:] = 1 / anew * (- aold * PL[i - 1,:] + np.multiply((xp - bnew),PL[i + 0,:]))

        aold = anew

    return np.transpose(PL[N,:])


