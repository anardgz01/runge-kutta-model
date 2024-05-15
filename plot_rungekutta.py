import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

earth_radius = 6378160
moon_earth_distance = 3.844e8
earth_radius_plot = earth_radius/moon_earth_distance

def myplot():
    r = np.load('resultados/r.npy')
    phi = np.load('resultados/phi.npy')
    hamiltonian_prime = np.load('resultados/hamiltonian_prime.npy')
    moon_position = np.load('resultados/moon_positions.npy')

    fig, axs = plt.subplots(ncols=2,figsize=(10,5))

    rocket, = axs[0].plot([], [], lw=2, color='green', label='Ship')
    rocket_marker, = axs[0].plot([], [], marker='*', color='green')
    moon, = axs[0].plot([], [], label='Moon', color='orange')
    moon_marker, = axs[0].plot([], [], marker='o', color='orange')

    axs[0].set_title('Moon and ship trajectories')
    axs[0].set_xlabel('x (Earth-moon distances)')
    axs[0].set_ylabel('y (Earth-moon distances)')
    axs[0].scatter(0,0, label='Earth', c='blue')
    axs[0].set_xlim(-1.5, 1.5)
    axs[0].set_ylim(-1.5, 1.5)

    # axs[0].plot(moon_position[:,0],moon_position[:,1], label='Moon')
    # axs[0].plot(0,0,'o',label='Earth', markersize=earth_radius_plot*2*72)

    # axs[0].plot(r*np.cos(phi),r*np.sin(phi),label='Ship')
    axs[0].set_aspect('equal')
    axs[0].legend()

    axs[1].set_title('Hamiltonian prime vs time')
    axs[1].plot(range(len(hamiltonian_prime)), hamiltonian_prime)
    axs[1].set_xlabel('Time (minutes)')
    axs[1].set_ylabel('H\'')
    hprime_avg = np.mean(hamiltonian_prime)
    axs[1].set_xlim(0, len(hamiltonian_prime))
    axs[1].set_ylim(hprime_avg-1e9,hprime_avg+1e9)

    def animate(i):
        rocket.set_data(r[:i*100]*np.cos(phi[:i*100]), r[:i*100]*np.sin(phi[:i*100]))
        rocket_marker.set_data(r[i*100]*np.cos(phi[i*100]), r[i*100]*np.sin(phi[i*100]))
        moon.set_data(moon_position[:i*100, 0], moon_position[:i*100, 1])
        moon_marker.set_data(moon_position[i*100, 0], moon_position[i*100, 1])
        return rocket, moon, rocket_marker, moon_marker

    ani = FuncAnimation(fig, animate, frames=len(r)//100, interval=20, blit=True)


    plt.show()
    
if __name__ == '__main__':
    myplot()    