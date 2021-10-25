import numpy as np

def rk4(f, xk, uk, dt):
    """
    Numerical Runge-Kutta4 method for propogating linear system dynamics:
    Inputs:
        xk -> state at time k
        uk -> control input at time k
        f  -> system dynamics : xkp1 = f(xk, uk)
        dt -> discrete time step deltat
    Output:
        xkp1 -> state at time k+1
    """
    k1 = f(xk, uk) * dt
    k2 = f(xk + k1/2, uk)*dt
    k3 = f(xk + k2/2, uk)*dt
    k4 = f(xk + k3, uk)*dt
    xkp1 = xk + k1/6 + k2/3 + k3/3 + k4/6

    return xkp1

def LinearSys(xk, uk, paramdict):
	Ak = paramdict["Ak"]
	Bk = paramdict["Bk"]
	f = Ak @ xk + Bk @ uk
	return f

def integratorDyn(xk, uk):
	# x : xk[0], y : xk[1], xdot : xk[2], ydot : xk[3]
	assert xk.shape[1] == 1 and xk.shape[0] == 4
	assert uk.shape[1] == 1 and uk.shape[0] == 2


	A = np.array([[0., 0., 1., 0.],
				  [0., 0., 0., 1.],
				  [0., 0., 0., 0.],
				  [0., 0., 0., 0.]])
	B = np.array([[0., 0.],
				  [0., 0.],
				  [1., 0.],
				  [0., 1.]])

	f = A@xk + B@uk
	return f

def CarDyn(xk, uk, paramdict):
	raise NotImplementedError

	return

if __name__ == "__main__" :
	pass
