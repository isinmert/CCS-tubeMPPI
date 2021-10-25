# MPPI python file
import numpy as np
from pdb import set_trace
from tqdm import tqdm

from queue import Queue
import queue
import threading

from pathos import multiprocessing as mp
from pathos.pools import ParallelPool
import pathos

from sysDynamics.sysdyn import rk4

_F = None
_C = None
_Phi = None

def rollout_trajectory(x0, T, dt,  F, Ubar, EpsSetk, C, Sigmainv, lambda_,
					   Nu_MPPIinv, Phi):
	xt = x0
	Sk = 0.
	for t in range(T):
		utm1 = Ubar[:,t:t+1]
		epstm1 = EpsSetk[:,t:t+1]
		xt = xt + F(xt, utm1+epstm1)*dt
		# xt = rk4(F, xt, utm1+epstm1, dt)

		# Sk += C(xt) + (1/2)*lambda_*(utm1.T @ np.linalg.inv(Sigma) @
		# 														(utm1+epstm1))
		# Sk += C(xt) + lambda_*(utm1.T @ np.linalg.inv(Sigma) @ epstm1)
		# Sk += C(xt) + lambda_*( (epstm1 +
		# 	(1/2)*utm1).T @ np.linalg.inv(Sigma) @ utm1)
		Sk += C(xt) + (lambda_/2)*(utm1.T@Sigmainv@utm1
							+ 2*utm1.T@Sigmainv@epstm1
							+ (1-Nu_MPPIinv)*epstm1.T@Sigmainv@epstm1
								)

	Sk += Phi(xt)
	return Sk

class rolloutWorker(threading.Thread):
	def __init__(self, task_q, result_q):
		threading.Thread.__init__(self)
		self.task_q = task_q
		self.result_q = result_q

	def run(self):
		global _F, _C, _Phi # get functions inside worker thread
		proc_name = self.name
		while True:
			try:
				item = self.task_q.get(block=True, timeout=3)
			except queue.Empty:
				print("No more jobs remaining for {}".format(proc_name))
				break
			# Ending signal to stop the thread
			if item is None:
				# print("Finished command sent to {}".format(proc_name))
				self.task_q.task_done()
				break
			else:
				x0, T, dt,  Ubar, EpsSetk, Sigmainv, lambda_, Nu_MPPIinv, k = item
				S = rollout_trajectory(x0, T, dt, _F, Ubar, EpsSetk, _C,
										Sigmainv, lambda_, Nu_MPPIinv, _Phi)

				self.result_q.put((k, S))
				self.task_q.task_done()
		return

def MPPI_pathos(x0, F, K, T, Sigma, Phi, C, lambda_, Ubar,
				Nu_MPPI=1.0, dt=0.1, progbar=False, num_workers=None,
				print_w=False):
	"""
	Nominal Controller MPPI Function:
	Inputs:
	x0 : Initial State (numpy array (nx,1))
	F : Transition Function (xk(numpy array (nx,1)), uk(numpy array (nu,1)))
			-> (xkp1)
	K : Number of Samples (integer)
	T : Number of Timesteps (integer)
	Sigma : Natural Input Uncertainty Covariance (numpy array (nx, nx))
	Phi : Final State Cost function (xk) -> R (real number)
	C : Running State Cost function (xk) -> R (real number)
	lambda_ : lambda parameter that appears in the MPPI theory (real number)
	Ubar : control input sequence (numpy array (nu, T))
	Nu_MPPI : MPPI control Sampling covariance multiplier
	dt : time discretization of the dynamics, default=0.1
	progbar : Show Progress Bar for computing MPPI. default:False
	num_workers : Number of Threads for compuatation
	Outputs:
	X : Computed State Sequence
	UbarNew : Computed New Input Sequence
	"""

	# Pick the number of cpu cores to distribute the processes
	if num_workers is None:
		num_workers = mp.cpu_count() - 1

	pool = mp.ProcessingPool(num_workers)
	# pool = pathos.pools.ParallelPool(num_workers)
	# pool = ParallelPool(num_workers)

	# Set necesarry variables and lists
	Sk_array = np.zeros(K)
	EpsSetAllk = []
	nx, nu = x0.shape[0], Sigma.shape[0]
	# Sigma_MPPI = Nu_MPPI * Sigma
	Sigma_MPPI = Nu_MPPI * np.eye(nu)
	Sigmainv = np.linalg.inv(Sigma)
	Nu_MPPIinv = 1/Nu_MPPI
	Nu_MPPIinv = 0.

	# Generate samples from input distribution
	for k in range(K):
		EpsSetk = np.random.multivariate_normal(np.zeros(nu),
													Sigma_MPPI, size=(T)).T
		if k == 0:
			EpsSetk = EpsSetk * 0.
		EpsSetAllk.append(EpsSetk)

	# Create function handle for trajectory rollout function
	rollout_handle = lambda x, eps_set : rollout_trajectory(x, T, dt,  F,
													Ubar, eps_set, C, Sigmainv,
													lambda_, Nu_MPPIinv, Phi)

	# Create x0_list for using map function and get results
	x0_list = [x0 for k in range(K)]
	Sk_list = pool.map(rollout_handle, x0_list, EpsSetAllk)

	# Compute weight updates for MPPI
	Sk_array = np.array(Sk_list)
	rho = min(Sk_array)

	wtilde_arr = np.exp((-1/lambda_)*(Sk_array-rho))
	wtilde_sum = np.sum(wtilde_arr)

	if print_w:
		print(wtilde_arr.squeeze())
		# input('press any key to continue')

	# Compute new control input sequence using weights and past control sequence
	UbarAdded = np.zeros(EpsSetk.shape)
	for k in range(K):
		UbarAdded += wtilde_arr[k]*EpsSetAllk[k]
	UbarAdded = UbarAdded/wtilde_sum
	UbarNew = Ubar + UbarAdded

	# Simulate new control input sequence
	X = np.zeros((nx, T))
	xbart = x0
	Sout = 0.
	for t in range(T):
		ubartm1 = UbarNew[:,t:t+1]
		xbart = xbart + F(xbart, ubartm1)*dt
		# xbart = rk4(F, xbart, ubartm1, dt)
		X[:,t:t+1] = xbart
		utm1 = Ubar[:,t:t+1]
		Sout += C(xbart) + (lambda_/2.)*(ubartm1.T @ Sigmainv@utm1)

	return X, UbarNew, Sout

def MPPI_thread(x0, F, K, T, Sigma, Phi, C, lambda_, Ubar,
					Nu_MPPI=1.0, dt=0.1, progbar=False, num_workers=20):
	"""
	Nominal Controller MPPI Function:
	Inputs:
	x0 : Initial State (numpy array (nx,1))
	F : Transition Function (xk(numpy array (nx,1)), uk(numpy array (nu,1)))
			-> (xkp1)
	K : Number of Samples (integer)
	T : Number of Timesteps (integer)
	Sigma : Natural Input Uncertainty Covariance (numpy array (nx, nx))
	Phi : Final State Cost function (xk) -> R (real number)
	C : Running State Cost function (xk) -> R (real number)
	lambda_ : lambda parameter that appears in the MPPI theory (real number)
	Ubar : control input sequence (numpy array (nu, T))
	Nu_MPPI : MPPI control Sampling covariance multiplier
	dt : time discretization of the dynamics, default=0.1
	progbar : Show Progress Bar for computing MPPI. default:False
	num_workers : Number of Threads for compuatation
	Outputs:
	X : Computed State Sequence
	UbarNew : Computed New Input Sequence
	"""
	global _F, _C, _Phi
	_F = F
	_C = C
	_Phi = Phi

	Sk_array = np.zeros(K)
	EpsSetAllk = []
	nx, nu = x0.shape[0], Sigma.shape[0]
	# Sigma_MPPI = Nu_MPPI * Sigma
	Sigma_MPPI = Nu_MPPI * np.eye(nu)
	Sigmainv = np.linalg.inv(Sigma)
	Nu_MPPIinv = 1/Nu_MPPI
	Nu_MPPIinv = 0.

	# Create rollout and cost Queues...
	rollout_q = Queue()
	cost_q = Queue()
	process_workers = [rolloutWorker(rollout_q, cost_q)
												for i in range(num_workers)]

	# set_trace()
	# Start worker threads
	for worker in process_workers:
		worker.start()


	# Create Jobs to be completed by workers
	for k in tqdm(range(K), disable = not progbar):
		# set_trace()
		# xt = x0 # Set initial nominal state
		EpsSetk = np.random.multivariate_normal(np.zeros(nu),
													Sigma_MPPI, size=(T)).T
		if k == 0:
			EpsSetk = EpsSetk * 0.
		EpsSetAllk.append(EpsSetk)

		rollout_q.put((x0, T, dt, Ubar, EpsSetk, Sigmainv, lambda_,
															Nu_MPPIinv, k))



	for i in range(num_workers):
		rollout_q.put(None)

	# set_trace()

	# Collect Results from Workers
	remaining_rollouts = K
	for k in tqdm(range(K), disable = not progbar):
		try:
			k_finished, cost = cost_q.get(block=True, timeout=10)
		except queue.Empty:
			break
		Sk_array[k_finished] = cost
		remaining_rollouts -= 1

	# set_trace()
	# Cleanup threading stuff
	# cost_q.close()
	rollout_q.join()
	# cost_q.join()

	rho = min(Sk_array)

	wtilde_arr = np.exp((-1/lambda_)*(Sk_array-rho))
	wtilde_sum = np.sum(wtilde_arr)

	UbarAdded = np.zeros(EpsSetk.shape)
	for k in range(K):
		UbarAdded += wtilde_arr[k]*EpsSetAllk[k]
	UbarAdded = UbarAdded/wtilde_sum
	UbarNew = Ubar + UbarAdded

	X = np.zeros((nx, T))
	xbart = x0
	Sout = 0.
	for t in range(T):
		ubartm1 = UbarNew[:,t:t+1]
		# xbart = xbart + F(xbart, ubartm1)*dt
		xbart = rk4(F, xbart, ubartm1, dt)
		X[:,t:t+1] = xbart
		utm1 = Ubar[:,t:t+1]
		Sout += C(xbart) + (lambda_/2.)*(ubartm1.T @ Sigmainv@utm1)

	return X, UbarNew, Sout


def MPPI(x0, F, K, T, Sigma, Phi, C, lambda_, Ubar,
				Nu_MPPI=1.0, dt=0.1, progbar=False):
	"""
	Nominal Controller MPPI Function:
	Inputs:
	x0 : Initial State (numpy array (nx,1))
	F : Transition Function (xk(numpy array (nx,1)), uk(numpy array (nu,1)))
			-> (xkp1)
	K : Number of Samples (integer)
	T : Number of Timesteps (integer)
	Sigma : Natural Input Uncertainty Covariance (numpy array (nx, nx))
	Phi : Final State Cost function (xk) -> R (real number)
	C : Running State Cost function (xk) -> R (real number)
	lambda_ : lambda parameter that appears in the MPPI theory (real number)
	Ubar : control input sequence (numpy array (nu, T))
	Nu_MPPI : MPPI control Sampling covariance multiplier
	dt : time discretization of the dynamics, default=0.1
	Outputs:
	X : Computed State Sequence
	UbarNew : Computed New Input Sequence
	"""
	Sk_array = np.zeros(K)
	EpsSetAllk = []
	nx, nu = x0.shape[0], Sigma.shape[0]
	# Sigma_MPPI = Nu_MPPI * Sigma
	Sigma_MPPI = Nu_MPPI * np.eye(nu)
	Sigmainv = np.linalg.inv(Sigma)
	Nu_MPPIinv = 1/Nu_MPPI

	for k in tqdm(range(K), disable = not progbar):
		# set_trace()
		xt = x0 # Set initial nominal state
		EpsSetk = np.random.multivariate_normal(np.zeros(nu),
													Sigma_MPPI, size=(T)).T
		if k == 0:
			EpsSetk = EpsSetk * 0.
		EpsSetAllk.append(EpsSetk)
		Sk = 0.
		for t in range(T):
			utm1 = Ubar[:,t:t+1]
			epstm1 = EpsSetk[:,t:t+1]
			# xt = xt + F(xt, utm1+epstm1)*dt
			xt = rk4(F, xt, utm1+epstm1, dt)

			# Sk += C(xt) + (1/2)*lambda_*(utm1.T @ np.linalg.inv(Sigma)
			# 												@ (utm1+epstm1))
			# Sk += C(xt) + lambda_*(utm1.T @ np.linalg.inv(Sigma) @ epstm1)
			# Sk += C(xt) + lambda_*( (epstm1 + (1/2)*utm1).T @
			# 		np.linalg.inv(Sigma) @ utm1)
			Sk += C(xt) + (lambda_/2)*(utm1.T@Sigmainv@utm1
								+ 2*utm1.T@Sigmainv@epstm1
								+ (1-Nu_MPPIinv)*epstm1.T@Sigmainv@epstm1
								)

		Sk += Phi(xt)
		Sk_array[k] = Sk

	rho = min(Sk_array)

	wtilde_arr = np.exp((-1/lambda_)*(Sk_array-rho))
	wtilde_sum = np.sum(wtilde_arr)

	UbarAdded = np.zeros(EpsSetk.shape)
	for k in range(K):
		UbarAdded += wtilde_arr[k]*EpsSetAllk[k]
	UbarAdded = UbarAdded/wtilde_sum
	UbarNew = Ubar + UbarAdded

	X = np.zeros((nx, T))
	xbart = x0
	Sout = 0.
	for t in range(T):
		ubartm1 = UbarNew[:,t:t+1]
		# xbart = xbart + F(xbart, ubartm1)*dt
		xbart = rk4(F, xbart, ubartm1, dt)
		X[:,t:t+1] = xbart
		Sout += C(xbart) + (lambda_/2.)*(ubartm1.T @ Sigmainv@utm1)

	return X, UbarNew, Sout
	# return X, UbarNew


if __name__ == "__main__":
	pass
