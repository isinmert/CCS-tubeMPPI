import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.stats import norm as stats_norm

import pdb

def linCovSteer(Alist, Blist, dlist, Wlist,
				mu_0, Sigma_0,
				# mu_des, Sigma_des,
				min_var_bound=5e0,
				prob_type="type-1", solver="MOSEK",
				Qlist=None, Rlist=None, Xref=None, Uref=None,
				ObsAvoidConstr=[],
				DELTA = 0.01,
				gamma = 0):
	"""
	Linear Covariance Steering Function
	"""
	nx, nu = Alist[0].shape[1], Blist[0].shape[1]
	N = len(Alist)

	if Qlist is None:
		Qlist = [np.eye(nx) for i in range(N+1)]
	if Rlist is None:
		Rlist = [np.eye(nu) for i in range(N)]

	if Xref is None:
		Xref = [np.zeros((nx, 1)) for i in range(N+1)]
	if Uref is None:
		Uref = [np.zeros((nu, 1)) for i in range(N)]


	# Get Necessary Matrices:
	G0, Gu, Gw, W, D, uffvar, Kvar, Lvar = getMatrices(Alist, Blist,
														dlist, Wlist,
														gamma=gamma)

	PTp1 = np.zeros((nx, nx*(N+1)))
	PTp1[:, -nx:] = np.eye(nx)

	# S = Rs @ Rs.T
	S = block_diag(Sigma_0, W)
	u, s, vh = np.linalg.svd(S, hermitian=True)
	Rs = u @ np.diag(np.sqrt(s))

	# Cov(X) = halfcovX @ halfcovX.T
	halfcovX = cp.hstack([G0 + Gu@Lvar, Gw + Gu@Kvar]) @ Rs
	# Cov(Xfinal) = zetaK @ zetaK.T
	zetaK = (PTp1 @ halfcovX)
	# Cov(U) = halfcovU @ halfcovU.T
	halfcovU = cp.hstack([Lvar, Kvar]) @ Rs
	# E[X] = fUbar
	fUbar = (Gu @ uffvar) + (G0 @ mu_0) + (Gw @ D)

	u, s, vh = np.linalg.svd(W, hermitian=True)
	Rw = u @ np.diag(np.sqrt(s))

	u, s, vh = np.linalg.svd(Sigma_0, hermitian=True)
	Rsigma0 = u @ np.diag(np.sqrt(s))

	# Qbig, Rbig = np.array([]), np.array([])
	for i in range(N):
		if i == 0:
			Qbig = Qlist[i]
			Rbig = Rlist[i]
		else:
			Qbig = block_diag(Qbig, Qlist[i])
			Rbig = block_diag(Rbig, Rlist[i])

	Qbig = block_diag(Qbig, Qlist[N])

	# Qbig = RQbig@RQbig.T;		Rbig = RRbig@RRbig.T
	u, s, vh = np.linalg.svd(Qbig, hermitian=True)
	RQbig = u @ np.diag(np.sqrt(s))

	u, s, vh = np.linalg.svd(Rbig, hermitian=True)
	RRbig = u @ np.diag(np.sqrt(s))

	# Turn Xref and Uref from list to array:
	Xref_array, Uref_array = cp.vstack(Xref), cp.vstack(Uref)
	DELTA_PARAM = stats_norm.ppf(1-DELTA)
	# DELTA_PARAM = np.sqrt(DELTA_PARAM)
	s = cp.Variable()

	if prob_type == "type-1":
		obj_func = (cp.norm(RQbig.T @ halfcovX, "fro")**2
						+ cp.quad_form(fUbar-Xref_array, Qbig)
						+ cp.norm(RRbig.T @ halfcovU,"fro")**2
						+ cp.quad_form(uffvar-Uref_array, Rbig)
			)
		# obj_func = (cp.norm(RQbig.T @ halfcovX, "fro")**2
		# 				+ cp.quad_form(fUbar-Xref_array, Qbig)
		# 				)
		# obj_func = cp.quad_form(uffvar-Uref_array, Rbig)
		obj = cp.Minimize(obj_func)
		constr = []
		for constr_item in ObsAvoidConstr:
			atilde = constr_item[0]
			btilde = constr_item[1]
			constr.append(atilde.T@fUbar - btilde  >=
							DELTA_PARAM * cp.norm(atilde.T@halfcovX, 2) )

		# constr.append(cp.sum_squares(uffvar) +
		# 			  cp.norm(Rw.T @ Kvar.T, "fro")**2 +
		# 			  cp.norm(Rsigma0.T @ Lvar.T, "fro")**2
		# 			  <= min_var_bound)
		prob = cp.Problem(obj, constr)
		prob.solve(solver=solver, verbose=False)


	elif prob_type == "type-2":
		obj_func = cp.norm(zetaK, "fro")**2
		obj_func += cp.norm(fUbar[-nx:-nx+2,:]-mud, 2)**2
		constr = []
		for constr_item in ObsAvoidConstr:
			atilde = constr_item[0]
			btilde = constr_item[1]
			constr.append(atilde.T@fUbar - btilde  >=
							DELTA_PARAM * cp.norm(atilde.T@halfcovX, 2) )

		# constr.append(cp.sum_squares(uffvar) +
		# 			  cp.norm(Rw.T @ Kvar.T, "fro")**2 +
		# 			  cp.norm(Rsigma0.T @ Lvar.T, "fro")**2
		# 			  <= min_var_bound)
		obj = cp.Minimize(obj_func)
		prob = cp.Problem(obj, constr)
		prob.solve(solver=solver, verbose=False)
	else:

		pass
	problem_status = prob.status
	# if problem_status != 'optimal':
	# 	pdb.set_trace()

	mean_list, cov_list = [], []
	input_mean_list, input_cov_list = [], []
	CovX = halfcovX@halfcovX.T
	CovU = halfcovU@halfcovU.T
	for i in range(N+1):
		mean_list.append(fUbar[i*nx:(i+1)*nx,:].value)
		cov_list.append(CovX[i*nx:(i+1)*nx,i*nx:(i+1)*nx].value)
		if i < N:
			input_mean_list.append(uffvar[i*nu:(i+1)*nu,:].value)
			input_cov_list.append(CovU[i*nu:(i+1)*nu,i*nu:(i+1)*nu].value)
	datadict = {}
	datadict["mean_list"] = mean_list
	datadict["cov_list"] = cov_list
	datadict["input_mean_list"] = input_mean_list
	datadict["input_cov_list"] = input_cov_list

	return uffvar.value, Lvar.value, Kvar.value, problem_status, datadict


def getTrackConstrIn(Xlist, N, Rin):
	"""
	Function to generate obstacle avoidance constrants for covariance steering
	in path tracking scenario. Only generates inner halfspaces.
	"""
	nx = Xlist[0].shape[0]
	constrData = []
	help_points = []

	for k in range(N+1):
		Fpos_k = np.zeros((2, (N+1)*nx))
		Fpos_k[:, k*nx:k*nx+2] = np.eye(2)

		mu_k_prev = Xlist[k][0:2]
		deltaz = mu_k_prev/np.linalg.norm(mu_k_prev)
		z = deltaz*Rin
		help_points.append(z)

		ain_k = deltaz
		bin_k = ain_k.T@z

		aintilde_k = Fpos_k.T @ ain_k
		bintilde_k = bin_k

		constrData.append((aintilde_k, bintilde_k))
		# pdb.set_trace()

	return constrData, help_points

def getTrackConstrOut(Xlist, N, Rout, N_seg=12):
	"""
	Function to generate
	"""
	nx = Xlist[0].shape[0]
	constrData = []
	# help_points = []

	seg_angle = 2*np.pi/N_seg
	half_angle = seg_angle/2.

	for k in range(N+1):
		Fpos_k = np.zeros((2, (N+1)*nx))
		Fpos_k[:, k*nx:k*nx+2] = np.eye(2)

		# Determine Angle:
		px, py = Xlist[k][0], Xlist[k][1]
		angle = np.arctan2(py, px)
		if angle < 0. : angle = 2*np.pi + angle
		# get_3_segment_indices:
		cur_seg = int(angle//seg_angle)
		next_seg = cur_seg+1
		if next_seg >= N_seg: next_seg = 0
		prev_seg = cur_seg-1
		if prev_seg <= -1: prev_seg = N_seg-1
		segment_indices = [prev_seg, cur_seg, next_seg]
		# get_halfspaces_from_segment indices
		# Multiply with Fpos_k and append to constrData
		for index in segment_indices:
			ang_hs = half_angle + index * seg_angle
			deltaz = np.array([[np.cos(ang_hs)],[np.sin(ang_hs)]])
			aout_k = -1.0*deltaz
			zoutk = deltaz*Rout
			bout_k = aout_k.T@zoutk

			atildek = Fpos_k.T@aout_k
			btildek = bout_k

			constrData.append((atildek, btildek))

	return constrData



def getTrackConstr(Xlist, N, Rin, Rout):
	"""
	Function to generate obstacle avoidance constraints for covariance steering
	in path tracking scenario.
	"""
	nx = Xlist[0].shape[0]
	constrData = []
	help_points =  []

	for k in range(N+1):
		Fpos_k = np.zeros((2, (N+1)*nx))
		Fpos_k[:, k*nx:k*nx+2] = np.eye(2)
		# help_points.append([])
		mu_k_prev = Xlist[k][0:2]
		deltaz = mu_k_prev / np.linalg.norm(mu_k_prev)
		zink = deltaz*Rin
		zoutk = deltaz*Rout
		help_points.append(zink)
		help_points.append(zoutk)

		ain_k = deltaz
		bin_k = ain_k.T@zink
		aout_k = -deltaz
		bout_k = aout_k.T@zoutk

		aintilde_k = Fpos_k.T @ ain_k
		bintilde_k = bin_k
		aouttilde_k = Fpos_k.T @ aout_k
		bouttilde_k = bout_k

		constrData.append((aintilde_k, bintilde_k))
		constrData.append((aouttilde_k, bouttilde_k))

	return constrData, help_points

def getObsConstr(Xlist, N, obstacles):
	"""
	Function to generate obstacle avoidance constraints for covariance steering
	"""
	nx = Xlist[0].shape[0]
	constrData = []
	help_points =  []

	for k in range(N+1):
		Fpos_k = np.zeros((2, (N+1)*nx))
		Fpos_k[:, k*nx:k*nx+2] = np.eye(2)
		help_points.append([])
		for obs_tuple in obstacles:
			z0, Robs = obs_tuple
			z0 = np.array([z0]).T
			mu_k_prev = Xlist[k][0:2]
			deltaz = mu_k_prev - z0
			deltaz = deltaz/np.linalg.norm(deltaz, 2) # norm_2(deltaz) = 1.
			xbark = z0 + deltaz * Robs
			help_points[-1].append(xbark)

			a_k = deltaz
			b_k = a_k.T@xbark

			atildek = Fpos_k.T @ a_k
			btildek = b_k
			constrData.append((atildek, btildek))

	return constrData, help_points


def getMatrices(Alist, Blist, dlist, Wlist, gamma=0):
	"""
	Get necessary constrant and variable matrices for the covariance steering
	computation.
	"""
	nx, nu = Alist[0].shape[1], Blist[0].shape[1]
	N = len(Alist)

	Gu = np.zeros((nx*(N+1), nu*N))
	Gw = np.zeros((nx*(N+1), nx*N))
	G0 = np.zeros((nx*(N+1), nx))

	G0[0:nx, :] = np.eye(nx)
	for i in range(1, N+1):
		G0[i*nx:(i+1)*nx, :] = _phi(Alist, i, 0)
		for j in range(i):
			Gw[i*nx:(i+1)*nx, j*nx:(j+1)*nx] = _phi(Alist, i, j+1)
			Gu[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = _phi(Alist, i, j+1) @ Blist[j]

	D = np.zeros((nx*N, 1))
	for i in range(N):
		D[i*nx:(i+1)*nx, :] = dlist[i]

	W = np.zeros((N*nx, N*nx))
	for i in range(N):
		W[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = Wlist[i]

	ufflist, Llist, Klist = [], [], []
	for i in range(N):
		ufflist.append(cp.Variable((nu,1)))

	for i in range(N):
		Llist.append([])
		Llist[-1].append(cp.Variable((nu, nx)))

	for i in range(N):
		Klist.append([])
		for j in range(N):
			if j <= i-1 and j >= (i-1)-gamma:
				Klist[-1].append(cp.Variable((nu, nx)))
			else:
				Klist[-1].append(np.zeros((nu, nx)))

	# pdb.set_trace()
	uffvar = cp.vstack(ufflist)
	Kvar = cp.bmat(Klist)
	Lvar = cp.bmat(Llist)


	return G0, Gu, Gw, W, D, uffvar, Kvar, Lvar

def _phi(Alist, k2, k1):
	"""
	Function to compute state transition matrix.
	k1: initial time
	k2: final time
	"""
	nx = Alist[0].shape[1]
	Phi = np.eye(nx)
	for k in range(k1, k2):
		Phi = Alist[k] @ Phi
	return Phi


if __name__ == "__main__":


	A = np.array([[1., 1.],[0., 1.]])

	B = np.array([[0.0], [1.0]])

	d = np.zeros((2,1))

	Sigmaw = np.array([[1.0, 0.0],[0.0, 1.0]])

	mud = np.array([[10.],[0.]])

	Sigmad = np.array([[1., 0.],[0., 1.]])

	mu0 = np.array([[0.],[0.]])

	Sigma0 = np.array([[1., 0.],[0., 1.]])

	N = 50

	Alist, Blist = [A for i in range(N)], [B for i in range(N)]
	Wlist, dlist = [Sigmaw for i in range(N)], [d for i in range(N)]

	G0, Gu, Gw, W, D, uffvar, Kvar, Lvar = getMatrices(Alist, Blist,
														dlist, Wlist)

	Xref = [np.array([[np.cos(k*0.1)],[np.sin(k*0.1)]]) for k in range(N+1)]

	uff, L, K = linCovSteer(Alist, Blist, dlist, Wlist, mu0, Sigma0, Xref=Xref)
	pdb.set_trace()

	pass
