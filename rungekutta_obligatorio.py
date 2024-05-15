'''Compute a program that resolves the movement equations in polar coordinates of a spaceship
in a gravitational field using the Runge-Kutta method of 4th order.
Show the trajectory of the spaceship and check that H' = H -\omega*p_{phi} is a constant of motion.
'''

import numpy as np
import plot_rungekutta as myplot

# Constants
G = 6.674e-11
M_EARTH = 5.9736e24
M_MOON = 7.349e22
R_EARTH = 6378160
R_MOON = 1737400
DISTANCE_EARTHMOON = 3.844e8
OMEGA = 2.6617e-6
DELTA = G*M_EARTH/DISTANCE_EARTHMOON**3
MU = M_MOON/M_EARTH
INITIAL_VELOCITY = 1.12e4
H=1
ITERATIONS = 100000
INITIAL_LATITUDE = np.pi/2
INITIAL_THETA = np.pi/8

#I want to resolve r_tilde=r/DISTANCE_EARTHMOON, phi_tilde=phi, p_r_tilde=p_r/(m*DISTANCE_EARTHMOON), p_phi_tilde=p_phi/(m*DISTANCE_EARTHMOON**2)
#Create the arrays
r = np.zeros(ITERATIONS)
phi = np.zeros(ITERATIONS)
p_r = np.zeros(ITERATIONS)
p_phi = np.zeros(ITERATIONS)
k_r = np.zeros(4)
k_phi = np.zeros(4)
k_p_r = np.zeros(4)
k_p_phi = np.zeros(4)
moon_positions = np.zeros((ITERATIONS, 2))

#First I initialize their values at t=0, normalizing them
r[0] = R_EARTH/DISTANCE_EARTHMOON
phi[0] = INITIAL_LATITUDE
p_r[0] = INITIAL_VELOCITY/DISTANCE_EARTHMOON*np.cos(INITIAL_THETA-INITIAL_LATITUDE)
p_phi[0] = r[0]*INITIAL_VELOCITY/DISTANCE_EARTHMOON*np.sin(INITIAL_THETA-INITIAL_LATITUDE)

#Now I define the moon's position
def calc_moon_position(t):
    x = np.cos(OMEGA*t*60)
    y = np.sin(OMEGA*t*60)
    return x,y

moon_positions[:,0], moon_positions[:,1] = calc_moon_position(np.arange(ITERATIONS))

#I define the funcions i want to resolve
def f_r(p_r):
    return p_r

def f_phi(p_phi,r):
    return p_phi/(r**2)

def f_p_r(r,phi,p_phi,t):
    r_prime =np.sqrt(1 + r**2 - 2*r*np.cos(phi-OMEGA*t*60))
    return p_phi**2/r**3 - DELTA*(1/r**2+MU/r_prime**3*(r-np.cos(phi-OMEGA*t*60)))

def f_p_phi(r,phi,t):
    r_prime =np.sqrt(1 + r**2 - 2*r*np.cos(phi-OMEGA*t*60))
    return -DELTA*MU*r/r_prime**3*np.sin(phi-OMEGA*t*60)

#Now I define the Runge-Kutta method
for i in range(0,ITERATIONS-1):
    #First I calculate the k's
    #k1
    k_r[0] = f_r(p_r[i])
    k_phi[0] = f_phi(p_phi[i],r[i])
    k_p_r[0] = f_p_r(r[i],phi[i],p_phi[i],i)
    k_p_phi[0] = f_p_phi(r[i],phi[i],i)

    #Define the intermediate values
    r1 = r[i] + k_r[0]/2
    phi1 = phi[i] + k_phi[0]/2
    p_r1 = p_r[i] + k_p_r[0]/2
    p_phi1 = p_phi[i] + k_p_phi[0]/2
    t_prime = i + H/2

    #k2
    k_r[1] = f_r(p_r1)
    k_phi[1] = f_phi(p_phi1,r1)
    k_p_r[1] = f_p_r(r1,phi1,p_phi1,t_prime)
    k_p_phi[1] = f_p_phi(r1,phi1,t_prime)

    #Define the intermediate values
    r2 = r[i] + k_r[1]/2
    phi2 = phi[i] + k_phi[1]/2
    p_r2 = p_r[i] + k_p_r[1]/2
    p_phi2 = p_phi[i] + k_p_phi[1]/2

    #k3
    k_r[2] = f_r(p_r2)
    k_phi[2] = f_phi(p_phi2,r2)
    k_p_r[2] = f_p_r(r2,phi2,p_phi2,t_prime)
    k_p_phi[2] = f_p_phi(r2,phi2,t_prime)

    #Define the intermediate values
    r3 = r[i] + k_r[2]
    phi3 = phi[i] + k_phi[2]
    p_r3 = p_r[i] + k_p_r[2]
    p_phi3 = p_phi[i] + k_p_phi[2]
    t_prime = i + H

    #k4
    k_r[3] = f_r(p_r3)
    k_phi[3] = f_phi(p_phi3,r3)
    k_p_r[3] = f_p_r(r3,phi3,p_phi3,t_prime)
    k_p_phi[3] = f_p_phi(r3,phi3,t_prime)

    #Now I calculate the new values
    r[i+1] = r[i] + 1/6*(k_r[0] + 2*k_r[1] + 2*k_r[2] + k_r[3])
    phi[i+1] = phi[i] + 1/6*(k_phi[0] + 2*k_phi[1] + 2*k_phi[2] + k_phi[3])
    p_r[i+1] = p_r[i] + 1/6*(k_p_r[0] + 2*k_p_r[1] + 2*k_p_r[2] + k_p_r[3])
    p_phi[i+1] = p_phi[i] + 1/6*(k_p_phi[0] + 2*k_p_phi[1] + 2*k_p_phi[2] + k_p_phi[3])

#Now I check that H' is a constant of motion
r_unscaled = r*DISTANCE_EARTHMOON
p_r_unscaled = p_r*DISTANCE_EARTHMOON
p_phi_unscaled = p_phi*DISTANCE_EARTHMOON**2
distance_shipmoon = np.sqrt(r_unscaled**2 + DISTANCE_EARTHMOON**2 - 2*r_unscaled*DISTANCE_EARTHMOON*np.cos(phi-OMEGA*np.arange(ITERATIONS)))
hamiltonian = p_r_unscaled**2/2 + p_phi_unscaled**2/(2*r_unscaled**2) - G*M_EARTH/r_unscaled - G*M_MOON/distance_shipmoon
hamiltonian_prime = hamiltonian - OMEGA*p_phi_unscaled
dhamiltonian_prime = np.gradient(hamiltonian_prime)

print(f'std in H\' = {np.std(hamiltonian_prime)}, average is {np.mean(hamiltonian_prime)}, max and min derivative of H\' = {np.max(dhamiltonian_prime)} and {np.min(dhamiltonian_prime)}, average of derivative of H\' = {np.mean(dhamiltonian_prime)}')

#Now I save the data
np.save('resultados/r.npy',r)
np.save('resultados/phi.npy',phi)
np.save('resultados/p_r.npy',p_r)
np.save('resultados/p_phi.npy',p_phi)
np.save('resultados/hamiltonian_prime.npy',hamiltonian_prime)
np.savetxt('resultados/hamiltonian_prime.txt',hamiltonian_prime)
np.savetxt('resultados/dhamiltonian_prime.txt',dhamiltonian_prime)
np.save('resultados/moon_positions.npy',moon_positions)

myplot.myplot()
