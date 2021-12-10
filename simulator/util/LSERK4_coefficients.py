#Low storage Runge-Kutta 4 coefficients:

# Carpenter, Mark H. and Christopher A. Kennedy. “Fourth-order 2N-storage Runge-Kutta schemes.” (1994):
# https://ntrs.nasa.gov/citations/19940028444

import numpy as np

rk4a = np.array([0.0 ,
				-567301805773.0/1357537059087.0 ,
				-2404267990393.0/2016746695238.0 ,
				-3550918686646.0/2091501179385.0  ,
				-1275806237668.0/842570457699.0])

rk4b = np.array([1432997174477.0/9575080441755.0 ,
				 5161836677717.0/13612068292357.0 ,
                 1720146321549.0/2090206949498.0  ,
				 3134564353537.0/4481467310338.0  ,
				 2277821191437.0/14882151754819.0])
rk4c = np.array([ 0.0,
				 1432997174477.0/9575080441755.0 ,
				 2526269341429.0/6820363962896.0 ,
				 2006345519317.0/3224310063776.0 ,
				 2802321613138.0/2924317926251.0])