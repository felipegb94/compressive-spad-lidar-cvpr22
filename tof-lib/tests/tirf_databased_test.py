'''
	Test functionality of toflib.tirf.py
	Example run command:
		run tests/tirf_databased_test.py -data_dirpath ./sample_data -scene_id vgroove
'''
#### Standard Library Imports
import argparse
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt

#### Local imports
from toflib import input_args_utils
from toflib import tirf, tof_utils
from research_utils.signalproc_ops import get_random_expgaussian_pulse_params  


def generate_transient_data_tirf(scene_id, data_dirpath='./sample_data'):
	assert(os.path.exists(data_dirpath)), "Input dirpath does not exist"
	transient_img_fpath = os.path.join(data_dirpath, 'scene_irf_data/{}_lres_transient_img.npz'.format(scene_id))
	print(" * transient_img_fpath: {}".format(transient_img_fpath))
	transient_tirf = tirf.TransientImgTIRF(transient_img_fpath)
	return transient_tirf

def generate_depth_data_tirf(scene_id, n_tbins, data_dirpath='./sample_data', delta_depth=1.):
	assert(os.path.exists(data_dirpath)), "Input dirpath does not exist"
	depth_img_fpath = os.path.join(data_dirpath, 'depth_data/{}.npy'.format(scene_id))
	print("    * depth_img_fpath: {}".format(depth_img_fpath))
	depth_tirf = tirf.DepthImgTIRF(depth_img_fpath, n_tbins=n_tbins, delta_depth=delta_depth)
	return depth_tirf

def generate_data_tirfs(scene_id, data_dirpath='./sample_data'):
	transient_tirf = generate_transient_data_tirf(scene_id, data_dirpath=data_dirpath)
	(n_rows, n_cols, n_tbins) = transient_tirf.tirf.shape
	rep_freq = 15e6
	rep_tau = 1. / rep_freq
	max_depth = tof_utils.time2depth(rep_tau)
	delta_depth = max_depth / n_tbins
	print("Rep Tau = {:.2f}ns, Max Depth = {}m, delta_depth = {},".format(rep_tau*1e9, max_depth, delta_depth))
	depth_tirf = generate_depth_data_tirf(scene_id, n_tbins, data_dirpath='./sample_data', delta_depth=delta_depth)
	return (transient_tirf, depth_tirf)


if __name__=='__main__':
	print("---- Running data-based tirf test ----")
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for tirf_test.')
	parser = input_args_utils.add_data_tirf_args(parser)
	args = parser.parse_args()
	#### Test data-based tirf
	# get input args
	scene_id = args.scene_id
	data_dirpath = args.data_dirpath
	# Create objects
	(transient_tirf, depth_tirf) = generate_data_tirfs(scene_id, data_dirpath)
	# Plot
	plt.clf()
	(n_rows, n_cols, n_tbins) = transient_tirf.tirf.shape
	(plot_row, plot_col) = (np.random.randint(low=0, high=n_rows),np.random.randint(low=0, high=n_cols))
	plt.plot(transient_tirf.t_domain, transient_tirf.tirf[plot_row,plot_col,:], linewidth=2)
	plt.plot(depth_tirf.t_domain, depth_tirf.tirf[plot_row,plot_col,:], linewidth=2)
	plt.title("Row: {}, Col: {}".format(plot_row, plot_col))
