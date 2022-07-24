'''
	plots the depths and depth error images for the lidar sim results
	Make sure you have run `eval_coding_flash_lidar_scene_batch.sh` with the correct parameters before running this script
	When running this script you need to set the following parameters to match what you have simulated with the eval_coding_flash_lidar_scene script:
	- sbr: set to sbr that has been simulated
	- n_photons: set to n_photons htat have been simulated
	- scene_id: set to 'kitchen-2' or 'bathroom-cycles-2'
	- n_codes_all: set to the n_codes (i.e., K) that have been simulated

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
from research_utils import plot_utils, io_ops, improc_ops, np_utils
from toflib import tof_utils
from eval_coding_utils import compose_coding_params_str
from datasets import FlashLidarSceneData
from simulate_flash_lidar_scene import get_scene_fname


def get_rec_algo_id(coding_id):
	rec_algo_id = 'ncc'
	if('Gated' == coding_id): rec_algo_id =  'linear'
	elif('GatedWide' == coding_id): rec_algo_id =  'linear'
	elif('Timestamp' == coding_id): rec_algo_id =  'matchfilt'
	elif('Identity' == coding_id): rec_algo_id =  'matchfilt'
	return rec_algo_id

def plot_and_save_depths(decoded_depths, abs_depth_errors, fname_base, out_dirpath, min_depth_val, max_depth_val, min_error_val, max_error_val):
	depthmap = decoded_depths*1000
	depth_errs = abs_depth_errors
	plt.clf()
	img=plt.imshow(depthmap, vmin=min_depth_val, vmax=max_depth_val)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath=out_dirpath, filename='Depthmap_'+fname_base, file_ext='svg')
	plot_utils.set_cbar(img, cbar_orientation='horizontal', fontsize=24)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='Depthmap_'+fname_base+'_withcbar', file_ext='svg')
	plt.clf()
	img=plt.imshow(depth_errs, vmin=min_error_val, vmax=max_error_val)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath=out_dirpath, filename='AbsErrors_'+fname_base, file_ext='svg')
	plot_utils.set_cbar(img, cbar_orientation='horizontal', fontsize=24)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='AbsErrors_'+fname_base+'_withcbar', file_ext='svg')

def plot_and_save_images(results, fname_base, out_dirpath, min_depth_val, max_depth_val, min_error_val, max_error_val):
	depthmap = results['decoded_depths'].squeeze()
	depth_errs = results['abs_depth_errors'].squeeze()
	plot_and_save_depths(depthmap, depth_errs, fname_base, out_dirpath, min_depth_val, max_depth_val, min_error_val, max_error_val)

def print_errors_metrics(results, fname_base):
	error_metrics = results['error_metrics'].item()
	print(fname_base)
	np_utils.print_error_metrics(error_metrics, prefix='    ')


if __name__=='__main__':

	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"
	## Set Simulation Params
	## Main paper parameters:
	#	* kitchen-2: (1000,0.25)
	#	* bathroom-cycles-2: (1000,0.5)
	sbr = 1.0
	n_photons = 2000
	sim_params_str = 'np-{:.2f}_sbr-{:.2f}'.format(n_photons, sbr)
	## Set Scene Params
	# scene_id='bathroom-cycles-2'
	scene_id='kitchen-2'
	view_id=0
	(n_rows,n_cols,n_tbins) = (240, 320, 2000)
	directonly=False
	scene_fname = get_scene_fname(scene_id=scene_id, n_rows=n_rows, n_cols=n_cols, n_tbins=n_tbins, directonly=directonly, view_id=view_id)
	## Get the dirpath where the results are stored
	results_data_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_data'], 'eval_coding_flash_lidar/'+scene_fname+'/'+sim_params_str)
	out_dirpath = os.path.join(out_base_dirpath, 'eval_coding_flash_lidar/'+scene_fname+'/'+sim_params_str)

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
	depth_img = data_sample[4]
	(min_depth_val, max_depth_val) = plot_utils.get_good_min_max_range(depth_img[depth_img < max_depth])
	(min_depth_error_val, max_depth_error_val ) = (0, 110)
	(min_depth_val, max_depth_val) = (min_depth_val*1000, max_depth_val*1000)

	## Set coding schemes we want to plot
	account_irf = True
	n_codes_all = [20, 40]
	pw_factor_shared = 1.0
	coding_ids = ['PSeriesFourier','TruncatedFourier','PSeriesGray','Gated','GatedWide']
	coding_ids = ['PSeriesFourier','TruncatedFourier','PSeriesGray', 'Gated']
	coding_ids = ['PSeriesGray', 'Random', 'HighFreqFourier']
	coding_ids = ['PSeriesGray', 'TruncatedFourier']

	## Plot RGB 
	plt.clf()
	img=plt.imshow(improc_ops.gamma_tonemap(rgb_img, gamma=1/4))
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath=out_dirpath, filename='rgb', file_ext='svg')

	## Plot ground truth depths
	plot_and_save_depths(depth_img, np.zeros_like(depth_img), 'gt-depths', out_dirpath, min_depth_val, max_depth_val, 0, 1)

	## Load Identity and plot those images
	frh_fname = compose_coding_params_str('Identity', n_tbins, get_rec_algo_id('Identity'), pw_factor_shared, account_irf=account_irf)
	frh_results = np.load(os.path.join(results_data_dirpath, frh_fname+'.npz'), allow_pickle=True)
	plot_and_save_images(frh_results, frh_fname, out_dirpath, min_depth_val, max_depth_val, min_depth_error_val, max_depth_error_val)
	print_errors_metrics(frh_results, frh_fname)

	## Plot photon count image
	simulated_hist_img = frh_results['c_vals'].squeeze()
	photon_count_img = simulated_hist_img.sum(axis=-1)
	peak_signal = simulated_hist_img.max(axis=-1)
	plt.clf()
	img=plt.imshow(photon_count_img)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath=out_dirpath, filename='photon-counts', file_ext='svg')
	plot_utils.set_cbar(img, cbar_orientation='horizontal', fontsize=24)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='photon-counts_withcbar', file_ext='svg')
	
	plt.clf()
	img=plt.imshow(peak_signal)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath=out_dirpath, filename='peak_signal', file_ext='svg')
	plot_utils.set_cbar(img, cbar_orientation='horizontal', fontsize=24)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='peak_signal_withcbar', file_ext='svg')

	idx = [0]
	idx = np.arange(0, len(coding_ids))

	idx2 = [0]
	idx2 = np.arange(0, len(n_codes_all))

	for j in idx2:
		n_codes = n_codes_all[j]
		for i in idx:
			if(coding_ids[i]=='GatedWide'):
				fname_base = compose_coding_params_str('Gated', n_codes, get_rec_algo_id(coding_ids[i]), n_tbins/n_codes, account_irf=account_irf)
			else:
				fname_base = compose_coding_params_str(coding_ids[i], n_codes, get_rec_algo_id(coding_ids[i]), pw_factor_shared, account_irf=account_irf)

			results = np.load(os.path.join(results_data_dirpath, fname_base+'.npz'), allow_pickle=True)

			plot_and_save_images(results, fname_base, out_dirpath, min_depth_val, max_depth_val, min_depth_error_val, max_depth_error_val)
			print_errors_metrics(results, fname_base)
