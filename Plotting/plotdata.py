# Plot Circle

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pdb

plt.rcParams["figure.autolayout"] = True

def plot_circle(X, R=2., h=0.125, help_points=[], X2=[]):

	Rin = R - h
	Rout = R + h
	thetadummy = np.linspace(0, 2*np.pi, 100)
	Rindata = Rin*np.array([np.cos(thetadummy), np.sin(thetadummy)])
	Routdata = Rout*np.array([np.cos(thetadummy), np.sin(thetadummy)])

	fig, ax = plt.subplots()
	# import pdb; pdb.set_trace()
	ax.plot(Rindata[0], Rindata[1], linewidth=2, color='k')
	ax.plot(Routdata[0], Routdata[1], linewidth=2, color='k')

	px, py = X[0,:], X[1,:]
	ax.plot(px, py, linewidth=1, color='r', marker=' ')

	if help_points:
		# h1 = []
		# for ii in range(len(help_points)):
		# 	for jj in range(len(help_points[ii])):
		# 		h1.append(help_points[ii][jj])
		help_arr = np.array(help_points).squeeze().T
		x_coord = help_arr[0,:]
		y_coord = help_arr[1,:]
		ax.plot(x_coord, y_coord,
				'bo', markersize=5.)
	if len(X2)>0:
		px2, py2 = X2[0,:], X2[1,:]
		ax.plot(px2, py2, linewidth=2, color='g', linestyle='--', marker='*')


	return fig, ax

def plot_quad(X, obs_list, pdes, fit_traj=False):
	thetadummy = np.linspace(0, 2*np.pi, 100)

	fig, ax = plt.subplots()

	# Create the list of obstacle patches:
	patch_obs_list = []
	# import pdb; pdb.set_trace()
	for obs_tuple in obs_list:
		obsX, obsY, obsR = obs_tuple[0][0], obs_tuple[0][1], obs_tuple[1]
		ax.add_artist(
			mpatches.Circle((obsX, obsY), obsR, fill=True, alpha=0.8, color='k')
			)
		# patch_obs_list.append(
		# 	mpatches.Circle((obsX, obsY), obsR, fill=True, alpha=0.5, color='k')
		# 		)
	# collection_obs = PatchCollection(patch_obs_list)

	px, py = X[0,:], X[1,:]
	ax.plot(px, py, linewidth=2, color='r')
	# ax.plot(px, py, linewidth=2, color='r', marker='+')
	ax.plot(pdes[0], pdes[1], 'g+', markersize=4)

	xlim_right = 10.5
	xlim_left = -0.5
	ylim_up = 10.5
	ylim_down = -0.5

	if fit_traj:
		maxpx, minpx, maxpy, minpy = max(px), min(px), max(py), min(py)

		xlim_right = max(xlim_right, maxpx)
		xlim_left = min(xlim_left, minpx)
		ylim_up = max(ylim_up, maxpy)
		ylim_down = min(ylim_down, minpy)

	ax.set_xlim([xlim_left, xlim_right])
	ax.set_ylim([ylim_down, ylim_up])

	return fig, ax
if __name__ == "__main__":
	pass
