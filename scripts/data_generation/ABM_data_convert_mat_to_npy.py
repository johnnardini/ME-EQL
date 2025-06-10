import numpy as np
from scipy.io import loadmat
import glob, pdb

for file in glob.glob("../../data/ABM/logistic_ABM_sim*.mat"):
	mat = loadmat(file)

	t = mat['t'].T

	u_ABM = mat['ABM_sim']
	u_ABM_t = mat['ABM_sim_t']

	print(t.shape)

	data = {}
	data['inputs'] = t
	data['outputs'] = u_ABM
	data['derivative_names'] = ['u','u_t']
	data['m'] = mat['m']
	data['rp'] = mat['rp']
	data['rd'] = mat['rd']
	data['rm'] = mat['rm']
	data['F'] = mat['F']

	data['variables'] = [t,u_ABM,u_ABM_t]
	data['variable_names'] = ['x_1','u','u_x1']

	print(file[:-4]+".npy")
	np.save(file[:-4]+".npy",data)
