'''Compute a program that resolves the movement equations in polar coordinates of a spaceship
in a gravitational field using the Runge-Kutta method of 4th order.
Show the trajectory of the spaceship and check that H' = H -\omega*p_{phi} is a constant of motion.
'''

import numpy as np

# Constants
G = 6.674e-11
M_EARTH = 5.9736e24
M_MOON = 7.349e22
R_EARTH = 6378160
R_MOON = 1737400
DISTANCE_EARTHMOON = 3.844e8
OMEGA = 2.6617e-6*60
INITIAL_VELOCITY = 1.2e4
H=1
ITERATIONS = 10000
INITIAL_LATITUDE = 0
INITIAL_THETA = np.pi/2

#I want to resolve r'=r/DISTANCE_EARTHMOON, phi'=phi, p_r'=p_r/(m*DISTANCE_EARTHMOON), p_phi'=p_phi/(m*DISTANCE_EARTHMOON**2)
#Create the arrays
r = np.zeros(ITERATIONS)
phi = np.zeros(ITERATIONS)
p_r = np.zeros(ITERATIONS)
p_phi = np.zeros(ITERATIONS)

#First I initialize their values at t=0
r[0] = R_EARTH/DISTANCE_EARTHMOON
phi[0] = INITIAL_LATITUDE
p_r[0] = INITIAL_VELOCITY/DISTANCE_EARTHMOON*np.cos(INITIAL_THETA-INITIAL_LATITUDE)
p_phi[0] = r[0]*INITIAL_VELOCITY/DISTANCE_EARTHMOON*np.sin(INITIAL_THETA-INITIAL_LATITUDE)

#Now I define the moon's position
def moon_position(t):
    x = DISTANCE_EARTHMOON*np.cos(OMEGA*t)
    y = DISTANCE_EARTHMOON*np.sin(OMEGA*t)
    return x,y

