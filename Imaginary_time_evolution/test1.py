import circuit_qubit
import circuit_qubit_update 

#define the total time, time step, number of steps
N_t = 
dt = 
T_tot = N_t * dt 

# compute the derivative function
# in general calculate d|\phi(\tau)> / d\theta_i 
# whether the derivative should be calculated analytically or numerically
# do not need to construct the total unitary
def derivative_V():

def A_cal():

def C_cal():

def theta_derivative():


# update theta function
# theta is a vector of the circuit varying parameters
# follow the formula theta(t+dt) = theta(t) + dt * (d theta / dt) = theta(t) + A^(-1) * C * dt 
def theta_update(theta_t0, dt, A, C):
    theta_t1 = theta_t0 + dt * theta_derivative(A, C)