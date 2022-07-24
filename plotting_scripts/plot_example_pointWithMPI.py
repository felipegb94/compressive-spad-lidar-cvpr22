'''
	plots isometric contours of a given coding approach with respect to the ideal setting where you have the full histogram available
'''
## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.plot_utils import *
from research_utils import io_ops, improc_ops
from toflib import tof_utils
from eval_coding_utils import compose_coding_params_str
from datasets import FlashLidarSceneData
from simulate_flash_lidar_scene import get_scene_fname

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	colors = get_color_cycle()
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams/point_with_MPI')

	## Params
	scene_id='kitchen-2'
	view_id=0
	(n_rows,n_cols,n_tbins) = (240, 320, 2000)
	tbin_size = 50 # 50 picosec
	tau = tbin_size*n_tbins
	time_domain = np.arange(0,n_tbins)*tbin_size
	f0 = 1e-6 / (tau*1e-12) # Funamental Freq in MHz
	freq_domain = np.arange(0, (n_tbins//2) + 1)*f0
	directonly=False
	scene_fname = get_scene_fname(scene_id=scene_id, n_rows=n_rows, n_cols=n_cols, n_tbins=n_tbins, directonly=directonly, view_id=view_id)

	# Load dirpaths and correct them depending on the ntbins input
	transient_images_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['transient_images_dirpath'])
	rgb_images_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['rgb_images_dirpath'])
	depth_images_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['depth_images_dirpath'])
	# check that ntbins match
	dirpath_ntbins = int(transient_images_dirpath.split('_')[-1].split('-')[-1])
	assert(dirpath_ntbins == n_tbins), 'make sure ntbins of images used matches input ntbins'
	## Set rep frequency depending on the domain of the simulated transient
	max_transient_path_length = 20 #
	(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils.calc_tof_domain_params(n_tbins, max_path_length=max_transient_path_length)
	## Load flash lidar scene data
	fl_dataset = FlashLidarSceneData(transient_images_dirpath, rgb_images_dirpath, depth_images_dirpath)
	# Get sample from dataset
	(data_sample, _) = fl_dataset.get_sample(scene_fname, simulate=False)
	transient_img_sim = data_sample[0].transpose((-2,-1,-3)) # simulated transient img
	transient_img = data_sample[1].transpose((-2,-1,-3)) # clean transieng img
	ambient_img = data_sample[2]
	rgb_img = data_sample[3].transpose((-2,-1,-3))
	rgb_img = improc_ops.gamma_tonemap(rgb_img, gamma=1/4)	
	depth_img = data_sample[4]
	(min_depth_val, max_depth_val) = get_good_min_max_range(depth_img[depth_img < max_depth])
	(min_depth_error_val, max_depth_error_val ) = (0, 110)
	(min_depth_val, max_depth_val) = (min_depth_val*1000, max_depth_val*1000)


	## Choose point to plot
	if(scene_id == 'kitchen-2'):
		(plot_row,plot_col) = (117,180)
		end_direct_bin = 325
		start_plot_bin = 275
		end_plot_bin = 600
	else:
		(plot_row,plot_col) = (n_rows//2,n_cols//2)
	
	## Split into direct and indirect component
	direct = np.array(transient_img[plot_row,plot_col,:])
	direct[end_direct_bin:] = 0
	indirect =  transient_img[plot_row,plot_col,:] - direct

	f_direct = np.fft.rfft(direct, axis=-1)
	f_indirect = np.fft.rfft(indirect, axis=-1)

	plt.clf()
	img=plt.imshow(rgb_img)
	remove_ticks()
	save_currfig(dirpath=out_dirpath, filename='rgb_{}'.format(scene_fname), file_ext='svg')

	plt.clf()
	plt.plot(time_domain, direct, linewidth=3, label="Direct Reflection")
	plt.xlim([time_domain[start_plot_bin], time_domain[end_plot_bin]])
	set_ticks(fontsize=14)
	update_fig_size(height=3, width=5)
	remove_yticks()
	plt.grid(axis='y', which='major',linestyle='--', linewidth=1)
	set_xy_box()
	plt.legend(fontsize=14)
	# plt.xticks(time_ticks, time_ticks_str)
	save_currfig(dirpath=out_dirpath, filename='time-irf-directonly_{}_{}-{}'.format(scene_fname, plot_row, plot_col))

	plt.plot(time_domain, indirect, linewidth=3, label="Indirect Reflections")
	set_xy_box()
	plt.legend(fontsize=14)
	save_currfig(dirpath=out_dirpath, filename='time-irf_{}_{}-{}'.format(scene_fname, plot_row, plot_col))

	plt.clf()
	plt.plot(freq_domain, np.abs(f_direct), '*',linewidth=2)
	plt.plot(freq_domain, np.abs(f_indirect), '*',linewidth=2)
	# plt.xlim([0, end_freq_factor*n_tbins])
	set_ticks(fontsize=14)
	update_fig_size(height=3, width=5)
	set_xy_box()
	plt.grid(axis='y', which='major',linestyle='--', linewidth=1)
	remove_yticks()
	# plt.xticks(freq_ticks, freq_ticks_str)
	save_currfig(dirpath=out_dirpath, filename='freq-irf_{}_{}-{}'.format(scene_fname, plot_row, plot_col))
