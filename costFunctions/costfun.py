# General Cost Functions File
import numpy as np

def EllipseCost(xk):
	"""
	Running Cost function for elliptical track
	"""
	px, py, vx, theta = xk[0], xk[1], xk[2], xk[3]

	d = np.abs((px/13)**2 + (py/6)**2 - 1)

	return (100*(d**2)) + (vx - 7.0)**2

def GenLinSysCost(xk, xtrack=None, Qk=None):
	"""
	Running Cost function for Linear System example
	"""
	if xtrack is None:
		xtrack = np.zeros(xk.shape)
	if Qk is None:
		Qk = np.eye(len(xk))

	xdiff = xk - xtrack
	cost = xdiff.T @ Qk @ xdiff
	return cost

def EllipseLinCost(xk):

	px, py, vx, vy = xk[0], xk[1], xk[2], xk[3]
	d = np.abs((px/13.)**2 + (py/6.)**2 - 1.)

	speed = np.sqrt(vx**2 + vy**2)

	return (100*(d**2)) + (speed - 7.0)**2

def LinBaselineCost(xk, vdes=3.0, R=2.0, h=0.125):
	px, py, vx, vy = xk[0], xk[1], xk[2], xk[3]
	speedcost = (np.sqrt(vx**2 + vy**2) - vdes)**2
	d = np.sqrt(px**2 + py**2)
	angular_momentum = px * vy - py * vx
	ang_moment_cost = abs(angular_momentum - R * vdes)
	poscost = 0 if (R-h<=d<=R+h) else 1000
	return speedcost + poscost + ang_moment_cost

def LinBaselineSoftCost(xk, vdes=7.0, R=2.0, h=0.125,
						wpos=100., wvel=1., wang=2.):
	px, py, vx, vy = xk[0], xk[1], xk[2], xk[3]
	speedcost = (np.sqrt(vx**2 + vy**2) - vdes)**2
	d = np.sqrt(px**2 + py**2)
	poscost = (d-R)**2
	angular_momentum = px * vy - py * vx
	ang_moment_cost = abs(angular_momentum - R * vdes)
	return wpos*poscost + wvel*speedcost + wang*ang_moment_cost

def QuadHardCost(xk, dt, pdes, obstacles=[]):
	"""
	2d quadrotor cost function for obstacle avoidance.
	Obstacles are given as a list of obstacle descriptions. Each description is
	a tuple of 2 item. First one is a tuple of position and the second one is
	the radius.
	"""
	px, py, vx, vy = xk[0], xk[1], xk[2], xk[3]
	pxdes, pydes = pdes[0], pdes[1]

	poscost = 20*((px-pxdes)**2 + (py-pydes)**2)
	obscost = 0.
	for obs in obstacles:
		pos_obs = obs[0]
		R_obs = obs[1]
		diff_obs = (px-pos_obs[0])**2 + (py-pos_obs[1])**2
		diff_obs = np.sqrt(diff_obs)
		if diff_obs <= R_obs:
			obscost += 100000.
		else:
			pass

	return poscost + obscost

def QuadSoftCost(xk, dt, pdes, obstacles=[]):
	"""
	2d quadrotor soft cost function for obstacle avoidance.
	Obstacles are given as a list of obstacle descriptions. Each description is
	a tuple of 2 item. First one is a tuple of position and the second one is
	the radius.
	The difference of QuadSoftCost from the QuadObsCost is that the cost func-
	tion does not have indicator functions.
	"""
	px, py, vx, vy = xk[0,0], xk[1,0], xk[2,0], xk[3,0]
	pxdes, pydes = pdes[0], pdes[1]
	p_vec = np.array([[px], [py]])

	poscost = 20*((px-pxdes)**2 + (py-pydes)**2)
	obscost = 0.
	for obs in obstacles:
		pos_obs = obs[0]
		R_obs = obs[1]
		diff_obs = (px-pos_obs[0])**2 + (py-pos_obs[1])**2
		diff_obs = np.sqrt(diff_obs)
		obscost += np.exp(-(1/2)*(1/R_obs**2)*diff_obs**2)

	obscost *= 2000.

	return poscost + obscost

def QuadSoftCost2(xk, dt, pdes, obstacles=[]):
	"""
	2d quadrotor soft cost function for obstacle avoidance.
	Obstacles are given as a list of obstacle descriptions. Each description is
	a tuple of 2 item. First one is a tuple of position and the second one is
	the radius.
	The difference of QuadSoftCost from the QuadObsCost is that the cost func-
	tion does not have indicator functions.
	"""
	px, py, vx, vy = xk[0,0], xk[1,0], xk[2,0], xk[3,0]
	pxdes, pydes = pdes[0], pdes[1]
	p_vec = np.array([[px], [py]])

	poscost = 20*((px-pxdes)**2 + (py-pydes)**2)
	obscost = 0.
	for obs in obstacles:
		pos_obs = obs[0]
		R_obs = obs[1]
		diff_obs = (px-pos_obs[0])**2 + (py-pos_obs[1])**2
		diff_obs = np.sqrt(diff_obs)
		obscost += np.exp(-(1/2)*(1/R_obs**2)*diff_obs**2)

	obscost *= 100.

	return poscost + obscost

def QuadPosCost(xk, dt, pdes, C=20.):
	"""
	Isolated Obstacle Cost function which utilizes the exponential function.
	"""
	px, py, vx, vy = xk[0,0], xk[1,0], xk[2,0], xk[3,0]
	pxdes, pydes = pdes[0], pdes[1]

	poscost = C*((px-pxdes)**2 + (py-pydes)**2)

	return poscost

def QuadObsCost(xk, dt, obstacles=[], C=100., l=0.5):
	"""
	Isolated desired position cost function.
	"""
	px, py, vx, vy = xk[0,0], xk[1,0], xk[2,0], xk[3,0]

	obscost = 0.
	for obs in obstacles:
		pos_obs, R_obs = obs
		diff_obs = (px-pos_obs[0])**2 + (py-pos_obs[1])**2
		diff_obs = np.sqrt(diff_obs)
		obscost += np.exp(-(1/l*2)*(1/R_obs**2)*diff_obs**2)

	obscost *= C

	return obscost


if __name__ == "__main__":
	import pdb; pdb.set_trace()
	pass
