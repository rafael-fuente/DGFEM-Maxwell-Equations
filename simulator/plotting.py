import matplotlib.pyplot as plt
import numpy as np

def plot_fields(self, title = None, **kwargs):
    # plot the electric and magnetic field

    fig = plt.figure(figsize=(10,6))



    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    if title != None:
        ax1.set_title(title)

    line1 = ax1.plot(self.z_grid, self.Ex.T.flatten(), 'r' ,lw=1.5 , **kwargs)
    ax1.set_ylim([-1.5,1.5])
    ax1.grid()
    ax1.set_ylabel('$E_x$')

    line2 = ax2.plot(self.z_grid, self.Hy.T.flatten(), 'b', lw=1.5 , **kwargs) 
    ax2.set_ylim([-1.5,1.5])
    ax2.set_ylabel('$H_y$')
    ax2.grid()
    
    ax2.set_xlabel(' z [μm]')


    
    ax1.plot(self.z_center, self.ε / np.amax(self.ε), label = "ε")
    ax2.plot(self.z_center, self.ε / np.amax(self.ε), label = "ε")
    ax1.legend()
    ax2.legend()
    
    plt.show()

def plot_flux(self):
    # plot the last iteration flux. Only used for debugging
    
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    line1 = ax1.plot(self.z_grid, self.Flux[:,:,0].T.flatten(), 'ro' ,lw=1.5)
    ax1.grid()

    line2 = ax2.plot(self.z_grid, self.Flux[:,:,1].T.flatten(), 'bo', lw=1.5) 

    ax1.set_ylabel('$Flux E_x$')
    ax2.set_ylabel('$Flux H_y$')
    ax2.set_xlabel(' z [μm]')
    ax2.grid()
    # 

    plt.show()

