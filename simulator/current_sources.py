import numpy as np

"""
CurrentSources class is used for modeling harmonic current sources placed at each finite element
"""


class CurrentSources:
    def __init__(self, element_indexes, frequencies, phases, J0):
        """
        element_indexes, frequencies, phases and J0 arguments are specified as lists.  
        J0 specifies the amplitude of the sources.
        Each element of the list, correspond to the properties of the current source at the finite element with the index specified
        """

        self.element_indexes = element_indexes
        self.frequencies = np.array(frequencies)
        self.phases = np.array(phases)
        self.J0 = np.array(J0)

    def init_sources(self, sim):

        self.Np = sim.Np
        self.K = sim.K

    def get_current(self, t):

        J = np.zeros((self.Np+1,self.K))
        J[:,self.element_indexes] = self.J0*np.sin(2*np.pi*self.frequencies * t + self.phases) 
        return J





