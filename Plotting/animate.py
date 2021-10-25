import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib as mpl

import matplotlib.patches as mpatches
# from matplotlib.collections import PatchCollection

import pdb
import argparse

def animate_linear(X, dt, animation_speed):

	fig, ax = plt.subplots()
	line, = ax.plot([], [], lw=2)
	current_pos, = ax.plot([], [])
	text_speed = ax.text(1.6, 2.5, '')
	text_time = ax.text(-2.5, 2.5, '')
	text_avgspeed = ax.text(-1.0, 2.5, '')

	ax.set_xlim([-3, 3])
	ax.set_ylim([-3, 3])
	line_inner, = ax.plot([], [], lw=3, ls='-', c='k')
	line_outer, = ax.plot([], [], lw=3, ls='-', c='k')

	dummy = np.linspace(0, 2*np.pi, 500)
	R, h = 2.0, 0.125
	Rinner, Router = R-h, R+h
	RinnerCoor = Rinner*np.array([np.cos(dummy), np.sin(dummy)])
	RouterCoor = Router * np.array([np.cos(dummy), np.sin(dummy)])

	line_inner.set_data(RinnerCoor)
	line_outer.set_data(RouterCoor)

	def init():
		line.set_data([], [])
		text_speed.set_text('')
		current_pos.set_data([], [])
		return line, text_speed, text_time, text_avgspeed, current_pos

	xdata, ydata = [], []

	def animate(i):
		# t is a parameter
		t = dt*i

		# x, y values to be plotted
		x = X[0,i]
		y = X[1,i]
		vx, vy = X[2,i], X[3,i]
		speed = np.sqrt(vx**2 + vy**2)
		Speed_Data = np.array([X[2,0:i+1], X[3,0:i+1]])
		# Speed_Data = Speed_Data**2
		avg_speed = np.mean(np.sqrt(np.sum(Speed_Data**2, 0))).squeeze()
		text_speed.set_text('|V(t)|={:.2f} m/s'.format(speed))
		text_time.set_text('t={:.1f} s'.format(t))
		text_avgspeed.set_text('|V(avg)|={:.2f} m/s'.format(avg_speed))

		# appending new points to x, y axes points list
		xdata.append(x)
		ydata.append(y)
		line.set_data(xdata, ydata)
		line.set_color('r')
		line.set_linewidth(1.5)

		current_pos.set_data(x, y)
		current_pos.set_marker('o')
		current_pos.set_markersize(6)
		current_pos.set_markerfacecolor('b')
		current_pos.set_markeredgecolor('b')


		return line, text_speed, text_time, text_avgspeed, current_pos

	anim = FuncAnimation(fig, animate, init_func=init,
						 frames=X.shape[1],
						 interval=1000*(dt/animation_speed), repeat=False)

	return anim

def animate_quad(X, pdes, obs_list, dt, animation_speed):
	fig, ax = plt.subplots()
	line, = ax.plot([], [], lw=2)
	current_pos, = ax.plot([], [])
	text_speed = ax.text(1.6, 2.5, '')
	text_time = ax.text(-2.5, 2.5, '')
	text_avgspeed = ax.text(-1.0, 2.5, '')

	ax.set_xlim([-0.5, 13.5])
	ax.set_ylim([-0.5, 13.5])

	for obs_tuple in obs_list:
		obsX, obsY, obsR = obs_tuple[0][0], obs_tuple[0][1], obs_tuple[1]
		ax.add_artist(
			mpatches.Circle((obsX, obsY), obsR, fill=True, alpha=0.8, color='k')
			)

	ax.plot(pdes[0], pdes[1], 'g+', markersize=4)

	def init():
		line.set_data([], [])
		text_speed.set_text('')
		current_pos.set_data([], [])
		return line, text_speed, text_time, text_avgspeed, current_pos

	xdata, ydata = [], []

	def animate(i):
		# t is a parameter
		t = dt*i

		# x, y values to be plotted
		x = X[0,i]
		y = X[1,i]
		vx, vy = X[2,i], X[3,i]
		speed = np.sqrt(vx**2 + vy**2)
		Speed_Data = np.array([X[2,0:i+1], X[3,0:i+1]])
		# Speed_Data = Speed_Data**2
		avg_speed = np.mean(np.sqrt(np.sum(Speed_Data**2, 0))).squeeze()
		text_speed.set_text('|V(t)|={:.2f} m/s'.format(speed))
		text_time.set_text('t={:.1f} s'.format(t))
		text_avgspeed.set_text('|V(avg)|={:.2f} m/s'.format(avg_speed))

		# appending new points to x, y axes points list
		xdata.append(x)
		ydata.append(y)
		line.set_data(xdata, ydata)
		line.set_color('r')
		line.set_linewidth(1.5)

		current_pos.set_data(x, y)
		current_pos.set_marker('o')
		current_pos.set_markersize(6)
		current_pos.set_markerfacecolor('b')
		current_pos.set_markeredgecolor('b')


		return line, text_speed, text_time, text_avgspeed, current_pos

	anim = FuncAnimation(fig, animate, init_func=init,
						 frames=X.shape[1],
						 interval=1000*(dt/animation_speed), repeat=False)

	return anim


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-filename', type=str, help="The path of the Data")
	parser.add_argument('-animation-speed', type=float, default=1.,
						help='Speed of the Animation', dest="speedanimation")
	parser.add_argument('-savevid', default=False, action="store_true",
						help="save animation as mp4 file")
	parser.add_argument('-quad', default=False, action="store_true",
						help="Use it to simulate obstacle avoidance quadrotor")

	args = parser.parse_args()

	FILENAME = args.filename
	QUAD = args.quad
	FILENAME_DATA = FILENAME + '/X.npy'
	FILENAME_PARAMS = FILENAME + '/params.txt'
	if QUAD:
		FILENAME_OBS = FILENAME + '/obs_list.npy'
		obs_list = np.load(FILENAME_OBS, allow_pickle=True)
		with open(FILENAME_PARAMS, 'r') as f:
			paramslist = f.readlines()
		for line in paramslist:
			if 'Desired Position' in line:
				des_pos_string = line.split(':')[1]
				des_list = des_pos_string.split('(')[1].replace(')','').split(',')
				DES_POS = tuple([float(x) for x in des_list])

	ANIMATION_SPEED = args.speedanimation
	SAVE_VIDEO = args.savevid


	with open(FILENAME_PARAMS) as f:
		paramslist = f.readlines()
		for line in paramslist:
			if 'dt' in line:
				dt = float(line.split(':')[1])


	X = np.load(FILENAME_DATA)

	if QUAD:
		anim = animate_quad(X, DES_POS, obs_list, dt,
							animation_speed=ANIMATION_SPEED)
	else:
		anim = animate_linear(X, dt, animation_speed=ANIMATION_SPEED)

	# anim = animate_linear(X, dt, animation_speed=ANIMATION_SPEED)

	if SAVE_VIDEO:
		mpl.rcParams['animation.ffmpeg_path'] = r'/Users/isinbalci/Downloads/ffmpeg'
		FPS = int(1/dt)
		mywriter = animation.FFMpegWriter(FPS)
		anim.save(FILENAME+'/animation.mp4', writer=mywriter)
		# anim.save(FILENAME+'/animation.gif', writer='pillow', fps=FPS)
	else:
		plt.show()
	# pdb.set_trace()
