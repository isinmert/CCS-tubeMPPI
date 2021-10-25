import numpy as np
import pdb

from matplotlib import pyplot as plt

def LQG(Alist, Blist, dlist, Wlist, Qlist, Rlist, 
		Qfinal=None, Xref=None, Uref=None):
	"""
	Linear Quadratic Gaussian Regulator for given parameters:
	Inputs:
		Alist (list) of (nx, nx) nd.arrays which represent Ak for every k.
		Blist (list) of (nx, nu) nd.arrays which represent Bk for every k.
		dlist (list) of (nx, 1)  nd.arrays which represent dk for every k.
		Wlist (list) of (nx, nx) nd.arrays which represent Wk for every k.
		Qlist (list) of (nx, nx) nd.arrays which represent Qk for every k.
		Rlist (list) of (nu, nu) nd.arrays which represent Rk for every k.
		optionals:
		Qfinal (nd.array) with (nx, nx) which represent the Qfinal.

	Outputs:
	"""
	resdict = {}
	ufflist, Kfblist = [], []
	nx, nu, N, = Alist[0].shape[1], Blist[0].shape[1], len(Alist)
	
	if Qfinal is None:
		Qfinal = 0. * np.eye(nx)
	
	if Xref is None:
		Xref = []
		for i in range(N+1):
			Xref.append(np.zeros((nx,1)))

	if Uref is None:
		Uref = []
		for i in range(N):
			Uref.append(np.zeros((nu, 1)))

	pass

	clist, qlist, Plist = [], [], []
	dhatlist = []
	for k in range(N):
		Ak, Bk = Alist[k], Blist[k]
		dk = dlist[k]
		xbark, xbarkp1 = Xref[k], Xref[k+1]
		ubark = Uref[k]
		# pdb.set_trace()
		dhatlist.append(dk + Ak@xbark + Bk@ubark - xbarkp1)
	pass

	clist.insert(0, 0.)
	qlist.insert(0, np.zeros((nx, 1)))
	Plist.insert(0, Qfinal)
	
	for k in range(N-1,-1,-1):
		# pdb.set_trace()
		Pkp1, qkp1, ckp1 = Plist[0], qlist[0], clist[0]

		ubark = Uref[k]
		Rk, Qk = Rlist[k], Qlist[k]
		Wk = Wlist[k]
		Ak = Alist[k]
		Bk = Blist[k]
		dhatk = dhatlist[k]

		Rtildeinv = np.linalg.inv(Bk.T@Pkp1@Bk + Rk)

		Kk = -Rtildeinv @ (Bk.T@Pkp1@Ak)
		uffk = -Rtildeinv @ (Bk.T@(Pkp1@dhatk + qkp1/2))

		Aclk = Ak + Bk@Kk
		Pk = Qk + Kk.T@Rk@Kk + Aclk.T@Pkp1@Aclk
		qk = 2*(Kk.T@Rk@uffk + Aclk.T@Pkp1@(Bk@uffk + dhatk) ) + Aclk@qkp1
		ck = (np.trace(Wk@Pkp1) + ckp1 + uffk.T@Rk@uffk 
			+ (Bk@uffk + dhatk).T@Pkp1@(Bk@uffk + dhatk) 
				+ qkp1.T@(Bk@uffk + dhatk) )

		Plist.insert(0, Pk)
		qlist.insert(0, qk)
		clist.insert(0, ck)

		Kfblist.insert(0, Kk)
		ufflist.insert(0, uffk+ubark)
		

	pass

	resdict["Plist"] = Plist
	resdict["qlist"] = qlist
	resdict["clist"] = clist

	return ufflist, Kfblist, resdict

if __name__ == "__main__":

	dt = 0.1

	Ac, Bc = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.], [1.]])
	nx, nu = Ac.shape[1], Bc.shape[1] 
	Ak = np.eye(nx) + Ac*dt
	Bk = Bc*dt
	Wk = np.eye(nx)*dt*0.001
	dk = 0.*np.array([[1.],[1.]])*dt

	Rk, Qk = 0.001*np.eye(nu), 1.*np.eye(nx)
	Qk[1,1] = 0.
	Qfinal = np.eye(nx)

	N = 100
	Alist = []
	Blist = []
	dlist = []
	Wlist = []
	Qlist = []
	Rlist = []
	xref = []
	omega = .1
	for k in range(N):
		Alist.append(Ak)
		Blist.append(Bk)
		dlist.append(dk)
		Wlist.append(Wk)
		Qlist.append(Qk)
		Rlist.append(Rk)
		# xref.append(np.array([[np.cos(k*omega)],[np.sin(k*omega)]]))
		if k <= N/2:
			xref.append( np.array([[3.],[0.]]) )
		else:
			xref.append( np.array([[-3.],[0.]]) )

	k += 1
	# xref.append(np.array([[np.cos(k*omega)],[np.sin(k*omega)]]))
	xref.append(np.array([[-3.],[0.]]))

	uff, K, res = LQG(Alist, Blist, dlist, Wlist, Qlist, Rlist, Qfinal=Qfinal,
						Xref=xref)

	x0 = np.array([[1.],[1.]])

	Xreal = []
	Xreal.append(x0)
	xk = x0
	for k in range(N):

		wk = np.random.multivariate_normal(np.zeros(2), Wk).reshape(2,1)
		uk = uff[k] + K[k]@(xk - xref[k])

		xkp1 = Ak@xk + Bk@uk + dk + wk
		Xreal.append(xkp1)
		xk = xkp1


	Xreal = np.array(Xreal).squeeze().T
	Xref = np.array(xref).squeeze().T
	plt.figure(1)
	plt.plot(Xref[0,:])
	plt.plot(Xreal[0,:])

	plt.figure(2)
	plt.plot(Xref[1,:])
	plt.plot(Xreal[1,:])

	plt.show()
	# pdb.set_trace()




