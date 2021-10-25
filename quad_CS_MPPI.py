# CSMPPI 2d quadrotor test script

import numpy as np
from costFunctions.costfun import LinBaselineCost, LinBaselineSoftCost
from costFunctions.costfun import QuadHardCost, QuadSoftCost, QuadSoftCost2
from costFunctions.costfun import QuadObsCost, QuadPosCost

from sysDynamics.sysdyn import integratorDyn
from sysDynamics.sysdyn import rk4

from controllers.MPPI import MPPI, MPPI_thread, MPPI_pathos
from controllers.LinCovSteer import linCovSteer, getObsConstr
from controllers.LQG import LQG

from Plotting.plotdata import plot_circle
from Plotting.plotdata import plot_quad

from matplotlib import pyplot as plt

from pdb import set_trace
from tqdm import tqdm
import argparse
import os

def main():
	parser = argparse.ArgumentParser("Covariance Steering MPPI for 2d " +
									 "quadrotor obstacle avoidance ")
	parser.add_argument('-mu', help='Mu parameter for MPPI', default=1.,
						type=float)
	parser.add_argument('-nu', default=1., type=float,
						help='Nu parameter for Sampling '+
						'with higher variance default=1., pick >=1.')
	parser.add_argument('-K', help='MPPI sample size parameter', default=500,
						type=int)
	parser.add_argument('-T', help='MPPI horizon parameter', default=10,
						type=int)
	parser.add_argument('-Tsim', help='Simulation Time steps', default=200,
						type=int)
	parser.add_argument('-lambda', dest="LAMBDA", default=0.1, type=float,
						help="Cost Function Parameter lambda default=0.1")
	parser.add_argument('-dt', type=float, default=0.05,
						help="Discrete time step. Default dt=0.05")
	parser.add_argument('-Rexit', type=float, default=20.,
						help="Simulation exit limits if abs(px) or abs(py) "+
						">= Rexit, then the simulation is terminated. "+
						"Default=20.")
	parser.add_argument('-seed', type=int, default=100,
						help="Random Number Generator Seed")
	parser.add_argument('-no-noise', default=False, action="store_true",
						dest="nonoise",
						help="Flag to simulate without noise on the input")
	parser.add_argument('-add-noise', type=float, default=0.0, dest="addnoise",
						help="additional noise to the system")
	parser.add_argument('-paramfile', default='./quad_params/quad_params1.txt',
						help="parameters file directory for simulations")
	parser.add_argument('-filename', type=str, default=None,
						help="Directory to save results")
	parser.add_argument('-qmult', type=float, default=1.0,
						help="Multiplier of state cost function default")
	parser.add_argument('-des-pos',  dest='des_pos_str', type=str,
						help="string of desired position given as (px, py)"+
						" default : (5., 5.)", default="(5., 5.)")
	parser.add_argument('-obs-file', dest="obs_file", help="obstacles file. "+
						"default = './quad_params/obs1.npy'",
						default='./quad_params/obs1.npy')
	parser.add_argument('-cost', type=str, default="sep",
						choices=["sep", "hard", "soft", "soft2"],
						help="Cost Type. Default:sep, "+
						"options: sep, hard, soft, soft2")
	args = parser.parse_args()


	mu = args.mu
	NU_MPPI = args.nu
	K = args.K
	T = args.T
	iteration = args.Tsim
	dt = args.dt
	lambda_ = args.LAMBDA
	seed = args.seed
	ADD_NOISE = args.addnoise
	Q_MULT = args.qmult
	des_pos_str = args.des_pos_str
	DES_POS_LIST = des_pos_str.replace('(', '').replace(')', '').split(',')
	DES_POS = tuple([float(x) for x in DES_POS_LIST])
	Rexit = args.Rexit
	OBS_FILE = args.obs_file
	COST_TYPE = args.cost

	np.random.seed(seed)

	FILENAME = args.filename

	PARAMFILE = args.paramfile
	if os.path.exists(PARAMFILE):
		with open(PARAMFILE) as f:
			filelist = f.readlines()

		for line in filelist:
			if "Natural System Noise Parameter" in line:
				mu = float(line.split(':')[1])
			elif "Control Sampling Covariance Parameter" in line:
				NU_MPPI = float(line.split(':')[1])
			elif "Number of Samples" in line:
				K = int(line.split(':')[1])
			elif "MPC Horizon" in line:
				T = int(line.split(':')[1])
			elif "Number of Simulation Timesteps" in line:
				iteration = int(line.split(':')[1])
			elif "Discretization time-step" in line:
				dt = float(line.split(':')[1])
			elif "Control Cost Parameter" in line:
				lambda_ = float(line.split(':')[1])
			elif "Random Number Generator" in line:
				seed = int(line.split(':')[1])
			elif "Q Multiplier" in line:
				Q_MULT = float(line.split(':')[1])
			elif "Additional Noise" in line:
				ADD_NOISE = float(line.split(':')[1])
			elif "Desired Position" in line:
				des_pos_string = line.split(':')[1]
				des_list = des_pos_string.split('(')[1].replace(')','').split(',')
				DES_POS = tuple([float(x) for x in des_list])
			elif "Obstacle File" in line:
				OBS_FILE = line.split("'")[1]
			elif "Cost Type" in line:
				COST_TYPE = line.split(':')[1].replace(' ', '').replace('\n','')


	x0 = np.array([[0.0],
				   [0.0],
				   [0.0],
				   [0.0]])


	Sigma = mu * np.eye(2)
	Sigmainv = np.linalg.inv(Sigma)
	Ubar = np.ones((2,T))


	F = lambda x, u : integratorDyn(x, u)

	obs_list = np.load(OBS_FILE, allow_pickle=True)
	# print(COST_TYPE)

	if COST_TYPE == 'sep':
		C = lambda x : Q_MULT*QuadObsCost(x, dt, obstacles=obs_list)
		Phi = lambda x : Q_MULT*T*QuadPosCost(x, dt, pdes=DES_POS)
	elif COST_TYPE == 'hard':
		C = lambda x : Q_MULT*QuadHardCost(x, dt, pdes=DES_POS,
											obstacles=obs_list)
		Phi = lambda x : 0.
	elif COST_TYPE == 'soft':
		C = lambda x : Q_MULT*QuadSoftCost(x, dt, pdes=DES_POS,
											obstacles=obs_list)
		Phi = lambda x : 0.
	elif COST_TYPE == 'soft2':
		C = lambda x : Q_MULT*QuadSoftCost2(x, dt, pdes=DES_POS,
											obstacles=obs_list)
		Phi = lambda x : 0.
	else:
		print('Undefined Cost Function!!')
		exit()

	# Linear Double Integrator Dynamics and Noise Covariance:
	Ak = np.eye(4)+ dt*np.array([[0., 0., 1., 0.],
				  				 [0., 0., 0., 1.],
				  				 [0., 0., 0., 0.],
				  				 [0., 0., 0., 0.]])
	Bk = dt*np.array([[0., 0.],
				  	  [0., 0.],
				  	  [1., 0.],
				  	  [0., 1.]])
	dk = np.zeros((4,1))
	Wk = np.eye(4)*dt
	Wk[0:2,0:2] = np.zeros((2,2))
	Wk = Wk * ADD_NOISE
	nx, nu = Ak.shape[1], Bk.shape[1]

	# Covariance Steering and LQG cost function parameters
	Qk, Rk = 100*np.eye(nx), 0.001*np.eye(nu)
	Qk[2:,2:] = 0.1*np.eye(2)
	Qfinal = Qk

	Alist, Blist, dlist, Wlist, Qlist, Rlist = [], [], [], [], [], []

	T_CS = 5	# Covariance Steering Horizon
	Alist, Blist = [Ak for k in range(T_CS)], [Bk for k in range(T_CS)]
	dlist, Wlist = [dk for k in range(T_CS)], [Wk for k in range(T_CS)]
	Qlist, Rlist = [Qk for k in range(T_CS+1)], [Rk for k in range(T_CS)]

	Sigma_threshold = 1.

	Xreal = []
	Ureal = []
	Xreal.append(x0)
	xk = x0
	xk_nom = xk
	Sigmak = np.eye(4) * 0. # Initial Covariance is set to 0.
	total_cost = 0.0
	Unom, U = Ubar, Ubar
	for i in tqdm(range(iteration), disable=False):

		Xnom, Unom, Snom = MPPI_pathos(xk_nom, F, K, T, Sigma, Phi, C, lambda_,
							Unom, Nu_MPPI=NU_MPPI, dt=dt, progbar=False)


		# Since LQG function takes Xref and Uref as python lists turn Xnom, Unom
		# into python lists:
		Xnom_list, Unom_list = [xk_nom], []
		for k in range(Xnom.shape[1]):
			Xnom_list.append(Xnom[:,k:k+1])
			Unom_list.append(Unom[:,k:k+1])

		# Obstacle Avoidance Constraints are generated from nominal state
		# trajectory
		constrData, help_points = getObsConstr(Xnom_list, T_CS,
												obstacles=obs_list)

		# Feedforward and feedback control laws from linear Covariance Steering
		uff_, L_, K_, prob_status = linCovSteer(Alist, Blist, dlist, Wlist,
								   mu_0=xk_nom,
								   Sigma_0=Sigmak,
								   prob_type="type-1", solver="MOSEK",
								   Qlist=Qlist, Rlist=Rlist,
								   Xref=Xnom_list[:T_CS+1],
								   Uref=Unom_list[:T_CS],
								   ObsAvoidConstr=constrData)

		if prob_status != "optimal":
			print('Optimization Problem is infeasible')
			print(xk)
			set_trace()
			break

		ubark, Kfbk = uff_[0:nu, :], L_[0:nu, :]

		Ui = Unom[:,0:T_CS].reshape(T_CS*nu, 1)

		uk = ubark + Kfbk @ (xk - xk_nom)
		# uk = ubark
		# uk = Unom[:,0:1] + K_fb @ (xk - xk_nom)

		eps = np.random.multivariate_normal(np.zeros(2), np.eye(2), (1,)).T * mu
		wk = np.random.multivariate_normal(
								np.zeros(4), Wk, (1,)).T


		xkp1 = xk + F(xk, uk)*dt + wk
		xkp1_nom = xk_nom + F(xk_nom, ubark)*dt
		Sigmakp1 = ((Ak + Bk@Kfbk) @ Sigmak @ (Ak + Bk@Kfbk).T) + Wk

		Xreal.append(xkp1)
		Ureal.append(uk)

		xk = xkp1
		xk_nom = xkp1_nom
		Sigmak = Sigmakp1
		# print(max(np.linalg.eig(Sigmak)[0]))
		if max(np.linalg.eig(Sigmak)[0]) >= Sigma_threshold:
			xk_nom = xk
			Sigmak = np.zeros(Sigmak.shape)

		Udummy = np.zeros(U.shape)
		Udummy[:,0:-1] = U[:,1:]
		Udummy[:,-1:] = U[:,-2:-1]
		U = Udummy

		Udummy = np.zeros(Unom.shape)
		Udummy[:,0:-1] = Unom[:,1:]
		Udummy[:,-1:] = Unom[:,-2:-1]
		Unom = Udummy

		Rkp1 = np.linalg.norm(xkp1[0:2], 2)
		total_cost += (C(xk) + (lambda_/2.)*(uk.T@Sigmainv@uk))*dt
		if np.abs(xkp1[0]) >= Rexit or np.abs(xkp1[1]) >= Rexit :
			print('Major Violation of Safety, Simulation Ended prematurely')
			break


	X = np.block([Xreal])
	Xpos = X[0:2, :]
	Xvel = X[2:, :]
	# Rvst = np.sqrt(np.sum(np.square(Xpos), 0))
	Vvst = np.sqrt(np.sum(np.square(Xvel), 0))
	Vmean = np.mean(Vvst)
	U = np.block([Ureal])
	Uxvst, Uyvst = U[0:1,:].squeeze(), U[1:2, :].squeeze()

	figtraj, axtraj = plot_quad(X, obs_list, DES_POS)

	fig2, ax2 = plt.subplots()
	# ax1.plot(Rvst)
	# ax1.title.set_text('R vs t')
	ax2.plot(Vvst)
	ax2.title.set_text('V vs t')

	fig3, (ax3, ax4) = plt.subplots(2)
	ax3.plot(Uxvst)
	ax3.title.set_text('$u_{x}$ vs t')
	ax4.plot(Uyvst)
	ax4.title.set_text('$u_{y}$ vs t')

	print('Total Cost: {:.2f}'.format(float(total_cost)))
	# print('Average Speed: {:.2f}'.format(float(Vmean)))

	paramslist = []
	# paramslist.append('Smooth Cost with Wpos:{} and Wvel:{}'.format(Wpos, Wvel) if SOFTCOST else 'Sparse Cost')
	paramslist.append('Natural System Noise Parameter, mu : {}'.format(mu))
	paramslist.append('Control Sampling Covariance Parameter, nu : {}'.format(NU_MPPI))
	paramslist.append('Number of Samples, K : {}'.format(K))
	paramslist.append('MPC Horizon, T : {}'.format(T))
	paramslist.append('Number of Simulation Timesteps, iteration : {}'.format(iteration))
	paramslist.append('Discretization time-step, dt : {}'.format(dt))
	paramslist.append('Control Cost Parameter, Lambda : {}'.format(lambda_))
	paramslist.append('Random Number Generator, seed : {}'.format(seed))
	# paramslist.append('Desired Speed : {}'.format(V_DES))
	paramslist.append('Q Multiplier : {}'.format(Q_MULT))
	paramslist.append('Cost Type : {}'.format(COST_TYPE))
	paramslist.append('Additional Noise Parameter, W : {}'.format(ADD_NOISE))
	paramslist.append('Desired Position : {}'.format(DES_POS))
	paramslist.append('-------RESULTS-------')
	paramslist.append('Total Cost : {:.2f}'.format(float(total_cost)))
	paramslist.append('Average Cost : {:.2f}'.format( float(total_cost/(iteration*dt)) ) )
	paramslist.append('Average Speed : {:.2f}'.format(Vmean))

	if FILENAME is None:
		print('\n'.join(paramslist))
		plt.show()
	elif type(FILENAME) is str:
		if not os.path.exists(FILENAME):
			os.system('mkdir {}'.format(FILENAME))
		np.save(FILENAME + '/X.npy', X)
		np.save(FILENAME + '/obs_list.npy', obs_list)
		figtraj.savefig(FILENAME + '/fig_traj.pdf')
		fig2.savefig(FILENAME + '/fig_v.pdf')
		fig3.savefig(FILENAME + '/fig_u.pdf')

		with open(FILENAME+'/params.txt', 'w+') as f:
			f.write('\n'.join(paramslist)+'\n')

	else:
		pass

	pass


if __name__ == "__main__":
	main()
