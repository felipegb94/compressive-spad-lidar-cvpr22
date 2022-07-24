'''
Description: 
	This script simulates a flash lidar scene using the ground truth rendered transient images and rgb images

	Output data and images are saved in the folder specified by joining the paths of 
	io_dirpaths['data_base_dirpath'] and io_dirpaths['results_dirpath'] and io_dirpaths['results_data']  

Example Run Commands:

	python eval_coding_flash_lidar_scene.py -scene_id kitchen-2 -sbr 0.5 -nphotons 1000 -n_rows 240 -n_cols 320 -n_tbins 2000 -coding Gray -n_bits 10 --account_irf --save_data_results

	run eval_coding_flash_lidar_scene.py -scene_id kitchen-2 -sbr 0.5 -nphotons 1000 -n_rows 240 -n_cols 320 -n_tbins 2000 -coding PSeriesFourier -n_freqs 8
	run eval_coding_flash_lidar_scene.py -scene_id kitchen-2 -sbr 0.5 -nphotons 1000 -n_rows 240 -n_cols 320 -n_tbins 2000 -coding TruncatedFourier -n_freqs 8

	run eval_coding_flash_lidar_scene.py -scene_id bathroom-cycles-2 -sbr 1 -nphotons 100 -n_tbins 200
	run eval_coding_flash_lidar_scene.py -scene_id bathroom-cycles-2 -sbr 1 -nphotons 100 -n_tbins 2000
'''
#### Standard Library Imports
import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from utils.input_args_parser import add_flash_lidar_scene_args
from toflib.input_args_utils import add_eval_coding_args
from toflib import tof_utils, tirf, tirf_scene, coding, coding_utils
from research_utils import plot_utils, np_utils, io_ops, improc_ops 
from research_utils.timer import Timer 
from datasets import FlashLidarSceneData
from simulate_flash_lidar_scene import get_scene_fname
import eval_coding_utils

if __name__=='__main__':
	# Set random seed to reproduce simulation results
	np.random.seed(0)
	
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for flash lidar simulation.')
	add_flash_lidar_scene_args(parser)
	add_eval_coding_args(parser)
	parser.add_argument('--save_results', default=False, action='store_true', help='Save result images.')
	parser.add_argument('--save_data_results', default=False, action='store_true', help='Save results data.')
	args = parser.parse_args()
	# Parse input args
	n_tbins = args.n_tbins
	max_path_length = args.max_transient_path_len
	(n_rows, n_cols, n_tbins) = (args.n_rows, args.n_cols, args.n_tbins)
	(scene_id, view_id) = (args.scene_id, args.view_id)
	colors = plot_utils.get_color_cycle()
	## Get scene that will be processed
	scene_fname = get_scene_fname(scene_id=scene_id, n_rows=n_rows, n_cols=n_cols, n_tbins=n_tbins, directonly=args.directonly, view_id=view_id)

	## Get dirpaths for input and output data
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_dirpath'], 'eval_coding_flash_lidar')
	out_data_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_data'], 'eval_coding_flash_lidar/'+scene_fname)
	os.makedirs(out_base_dirpath, exist_ok=True)
	os.makedirs(out_data_base_dirpath, exist_ok=True)
	# Load dirpaths and correct them depending on the ntbins input
	transient_images_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['transient_images_dirpath'])
	rgb_images_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['rgb_images_dirpath'])
	depth_images_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['depth_images_dirpath'])
	# check that ntbins match
	dirpath_ntbins = int(transient_images_dirpath.split('_')[-1].split('-')[-1])
	assert(dirpath_ntbins == n_tbins), 'make sure ntbins of images used matches input ntbins'

	## Set rep frequency depending on the domain of the simulated transient
	(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils.calc_tof_domain_params(n_tbins, max_path_length=max_path_length)

	## Get coding ids and reconstruction algos and verify their lengths
	coding_ids = args.coding
	rec_algos_ids = args.rec
	pw_factors = np_utils.to_nparray(args.pw_factors)
	n_coding_schemes = len(coding_ids)
	(coding_scheme_ids, rec_algos_ids, pw_factors) = eval_coding_utils.generate_coding_scheme_ids(coding_ids, rec_algos_ids, pw_factors)

	## Set signal and sbr levels at which the MAE will be calculated at
	(signal_levels, sbr_levels, nphotons_levels) = eval_coding_utils.parse_signalandsbr_params(args)
	(X_sbr_levels, Y_nphotons_levels) = np.meshgrid(sbr_levels, nphotons_levels)
	n_nphotons_lvls = len(nphotons_levels)
	n_sbr_lvls = len(sbr_levels)

	## Create GT gaussian pulses for each coding. Different coding may use different pulse widths
	pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors*tbin_res, mu=0, t_domain=t_domain)

	## initialize coding strategies
	coding_list = coding_utils.init_coding_list(coding_ids, n_tbins, args, pulses_list=pulses_list)

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

	## Create objects for scene
	# For each pulse type we need to create a new scene_obj
	scene_obj_list = []
	for i in range(n_coding_schemes):
		transient_obj = tirf.TemporalIRF(pulses_list[i].apply(transient_img), t_domain=t_domain)
		scene_obj = tirf_scene.ToFScene(transient_obj, rgb_img)
	
	for i in range(n_coding_schemes):
		coding_id = coding_ids[i]
		pw_factor = pw_factors[i]
		coding_obj = coding_list[i]
		rec_algo = rec_algos_ids[i]
		coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo=rec_algo, pw_factor=pw_factor, account_irf=args.account_irf)
		for j in range(n_nphotons_lvls):
			for k in range(n_sbr_lvls):
				## Simulate a dtof image
				curr_mean_sbr = X_sbr_levels[j, k]
				curr_mean_nphotons = Y_nphotons_levels[j, k]
				transient_img_sim = scene_obj.dtof_sim(mean_nphotons=curr_mean_nphotons, mean_sbr=curr_mean_sbr)
				## Encode
				c_vals = coding_obj.encode(transient_img_sim)
				# Estimate depths
				decoded_depths = eval_coding_utils.decode_peak(coding_obj, c_vals, coding_id, rec_algo, pw_factor)*tbin_depth_res
				## Calc error metrics
				abs_depth_errors = np.abs(decoded_depths.squeeze() - depth_img)*1000
				error_metrics = np_utils.calc_error_metrics(abs_depth_errors, delta_eps = tbin_depth_res*1000)
				np_utils.print_error_metrics(error_metrics)
				## Plot depths and depth errors
				plt.clf()
				plt.subplot(2,2,1)
				img=plt.imshow(improc_ops.gamma_tonemap(rgb_img, gamma=1/4))
				plot_utils.remove_ticks()
				plt.title("RGB Image")
				plt.subplot(2,2,2)
				img=plt.imshow(depth_img*1000, vmin=min_depth_val, vmax=max_depth_val)
				plot_utils.remove_ticks()
				plot_utils.set_cbar(img)
				plt.title("Depth Image")
				plt.subplot(2,2,3)
				img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
				plot_utils.remove_ticks()
				plot_utils.set_cbar(img)
				plt.title("Absolute depth errors")
				plt.subplot(2,2,4)
				img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
				plot_utils.remove_ticks()
				plot_utils.set_cbar(img)
				plt.title("Decoded Depths")

				if(args.save_data_results):
					out_data_dirpath = os.path.join(out_data_base_dirpath, 'np-{:.2f}_sbr-{:.2f}'.format(curr_mean_nphotons, curr_mean_sbr))
					os.makedirs(out_data_dirpath, exist_ok=True)
					out_fname_base = coding_params_str
					np.savez(os.path.join(out_data_dirpath, out_fname_base+'.npz')
								, decoded_depths=decoded_depths
								, abs_depth_errors=abs_depth_errors
								, error_metrics=error_metrics
								, c_vals=c_vals
								, rep_freq=rep_freq
								, rep_tau=rep_tau
								, tbin_res=tbin_res
								, t_domain=t_domain
								, max_depth=max_depth
								, tbin_depth_res=tbin_depth_res
								, Cmat=coding_obj.C
								, Cmat_decoding=coding_obj.decoding_C
					)
					
				if(args.save_results):
					sim_params_str = '{}_np-{:.2f}_sbr-{:.2f}'.format(scene_id, curr_mean_nphotons, curr_mean_sbr)
					out_dirpath = os.path.join(out_base_dirpath, sim_params_str)
					coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo, pw_factor)

					plt.figure()
					plot_utils.update_fig_size(height=5, width=6)
					img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
					plot_utils.remove_ticks()
					plot_utils.set_cbar(img, cbar_orientation='horizontal')
					plot_utils.save_currfig(dirpath=out_dirpath, filename=coding_params_str+'_depths', file_ext='svg')
					
					plt.figure()
					plot_utils.update_fig_size(height=5, width=6)
					img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
					plot_utils.remove_ticks()
					plot_utils.set_cbar(img, cbar_orientation='horizontal')
					plot_utils.save_currfig(dirpath=out_dirpath, filename=coding_params_str+'_deptherrs', file_ext='svg')

	if(args.save_results):
		plt.figure()
		plot_utils.update_fig_size(height=5, width=6)
		img=plt.imshow(improc_ops.gamma_tonemap(rgb_img, gamma=1/4))
		plot_utils.remove_ticks()
		plot_utils.save_currfig(dirpath=out_base_dirpath, filename=scene_id+'_rgb', file_ext='svg')

		plt.figure()
		plot_utils.update_fig_size(height=5, width=6)
		img=plt.imshow(depth_img*1000, vmin=min_depth_val, vmax=max_depth_val)
		plot_utils.remove_ticks()
		# plot_utils.set_cbar(img, cbar_orientation='horizontal')
		plot_utils.save_currfig(dirpath=out_base_dirpath, filename=scene_id+'_depths', file_ext='svg')

