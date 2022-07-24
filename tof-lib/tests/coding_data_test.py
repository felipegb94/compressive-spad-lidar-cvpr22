'''
	Test functionality of toflib.coding.py:DataCoding Class
	Sample Run command: 
		run tests/coding_data_test.py -c_data_fpath sample_data/sample_corrfs/keatof_2MHz_ham-k3_min.npy -scene_id vgroove
'''
#### Standard Library Imports
import argparse
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils.timer import Timer
from research_utils import plot_utils
from toflib import coding
from toflib import tof_utils, input_args_utils
from tirf_databased_test import generate_depth_data_tirf

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for scene_tirf_test.')
	parser = input_args_utils.add_data_coding_args(parser)
	args = parser.parse_args()
	# Parse input args
	c_data_fpath = args.c_data_fpath
	data_dirpath = args.data_dirpath
	scene_id = args.scene_id
	rep_freq = args.rep_freq
	
	# Load coding data
	if(not ('.npy' in c_data_fpath)): c_data_fpath+='.npy'	
	c_data = np.load(c_data_fpath)
	if(c_data.shape[-1] > c_data.shape[-2]): c_data = np.swapaxes(c_data, -1, -2)
	(n_tbins, n_codes) = c_data.shape
	# Initialize coding
	c = coding.DataCoding(C=c_data[..., 0::2])
	trunc_fourier_c = coding.TruncatedFourierCoding(n_tbins, n_freqs=5)
	print("Reconstruction Algos in DataCoding: {}".format(c.rec_algos_avail))
	print("Reconstruction Algos in FourierCoding: {}".format(trunc_fourier_c.rec_algos_avail))

	# Get delta depth to construct scene
	rep_tau = 1. / rep_freq
	max_depth = tof_utils.time2depth(rep_tau)
	delta_depth = max_depth / n_tbins
	# Initialize some scene
	depth_tirf = generate_depth_data_tirf(scene_id, n_tbins=n_tbins, data_dirpath=data_dirpath, delta_depth=delta_depth)
	gt_depths = depth_tirf.depth_img*1000
	# Encode tirf with c
	with Timer("Encode"):
		c_vals = c.encode(depth_tirf.tirf)

	with Timer("Trunc Fourier Encode"):
		trunc_fourier_c_vals = trunc_fourier_c.encode(depth_tirf.tirf)

	with Timer("c zncc_reconstruction"):
		zncc_depths = c.max_peak_decoding(c_vals, rec_algo_id='zncc')*delta_depth*1000

	with Timer("trunc_fourier_c ifft_reconstruction"):
		trunc_fourier_c.lres_mode = False
		if(trunc_fourier_c.lres_mode): true_delta_depth = delta_depth*trunc_fourier_c.lres_factor 
		else: true_delta_depth = delta_depth
		trunc_ift_depths = trunc_fourier_c.max_peak_decoding(trunc_fourier_c_vals, rec_algo_id='ifft')*true_delta_depth*1000

	with Timer("coarse-fine zncc_reconstruction"):
		cf_zncc_depths = c.dualres_zncc_depth_decoding(c_vals)*delta_depth*1000

	plt.clf()
	plt.subplot(3,2,1)
	plt.imshow(cf_zncc_depths, vmin=500, vmax=850)
	plt.title('Coarse-Fine Depths')
	plt.subplot(3,2,2)
	plt.imshow(zncc_depths, vmin=500, vmax=850)
	plt.title('ZNCC Depths')
	plt.subplot(3,2,3)
	curr_img = plt.imshow(np.abs(gt_depths-cf_zncc_depths), vmin=0, vmax=40)
	plot_utils.set_cbar(curr_img, cbar_orientation='horizontal')
	plt.title('Coarse-Fine Errors (mm)')
	plt.subplot(3,2,4)
	curr_img = plt.imshow(np.abs(gt_depths-zncc_depths), vmin=0, vmax=40)
	plot_utils.set_cbar(curr_img, cbar_orientation='horizontal')
	plt.title('ZNCC Errors (mm)')
	plt.subplot(3,2,5)
	plt.plot(c.C, linewidth=2)
	plt.title('Codes')
	plt.subplot(3,2,6)
	curr_img = plt.imshow(np.abs(gt_depths-trunc_ift_depths), vmin=0, vmax=40)
	plot_utils.set_cbar(curr_img, cbar_orientation='horizontal')
	plt.title('IFFT Errors (mm)')
