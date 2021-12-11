import numpy as np

"""
Detector class is used for storing the time evolution of the fields at the left vertex of a specified element k
"""
class Detector:
    def __init__(self, element_index):
        self.element_index = element_index

    def init_detector(self, Nt, total_time):

        self.Ex = np.zeros([Nt])
        self.Hy = np.zeros([Nt])
        self.total_time = total_time
        self.dt = total_time/Nt
        self.Nt = Nt

        self.t = np.arange(0, total_time, self.dt) + self.dt

    def measure_fields(self, Ex, Hy, time_step_index):

        self.Ex[time_step_index] =  Ex[0, self.element_index]
        self.Hy[time_step_index] =  Hy[0, self.element_index]



    def get_Ex_spectrum(self):

        Ex_spectrum = np.fft.fftshift(np.fft.fft(  self.Ex   ))*self.dt /( np.exp(np.pi*1j * (np.arange(self.Nt)-self.Nt//2)))
        f = np.fft.fftshift(np.fft.fftfreq(self.Nt, d = self.dt))

        return f, Ex_spectrum




