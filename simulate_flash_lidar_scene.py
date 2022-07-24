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
from datasets import FlashLidarSceneData
from toflib.input_args_utils import add_tofsim_args
from toflib import tof_utils, tirf, tirf_scene
from research_utils import plot_utils, np_utils, io_ops, improc_ops


def get_scene_fname(scene_id='cbox', n_rows=120, n_cols=160, n_tbins=2000, directonly=False, view_id=0):
	scene_fname = '{}_nr-{}_nc-{}_nt-{}_samples-2048'.format(scene_id, n_rows, n_cols, n_tbins)
	if(directonly): scene_fname += '_directonly'
	return scene_fname + '_view-{}'.format(view_id)

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for flash lidar simulation.')
	add_flash_lidar_scene_args(parser)
	add_tofsim_args(parser)
	args = parser.parse_args()
	# Parse input args
	n_tbins = args.n_tbins
	mean_signal_photons = args.n_photons
	mean_sbr = args.sbr
	max_path_length = args.max_transient_path_len
	(n_rows, n_cols, n_tbins) = (args.n_rows, args.n_cols, args.n_tbins)
	(scene_id, view_id) = (args.scene_id, args.view_id)
	colors = plot_utils.get_color_cycle()

	# Get dirpaths for data
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	transient_images_dirpath = io_dirpaths['transient_images_dirpath']
	rgb_images_dirpath = io_dirpaths['rgb_images_dirpath']
	depth_images_dirpath = io_dirpaths['depth_images_dirpath']    

	## Set rep frequency depending on the domain of the simulated transient
	(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils.calc_tof_domain_params(n_tbins, max_path_length=max_path_length)

	## Load flash lidar scene data
	scene_fname = get_scene_fname(scene_id=scene_id, n_rows=n_rows, n_cols=n_cols, n_tbins=n_tbins, directonly=args.directonly, view_id=view_id)
	fl_dataset = FlashLidarSceneData(transient_images_dirpath, rgb_images_dirpath, depth_images_dirpath)

	# Get sample from dataset
	(data_sample, _) = fl_dataset.get_sample(scene_fname, mean_signal=mean_signal_photons, mean_sbr=mean_sbr)
	transient_img_sim = data_sample[0].transpose((-2,-1,-3)) # simulated transient img
	transient_img = data_sample[1].transpose((-2,-1,-3)) # clean transieng img
	ambient_img = data_sample[2]
	rgb_img = data_sample[3].transpose((-2,-1,-3))
	depth_img = data_sample[4] 

	## Estimate depths
	(min_depth_val, max_depth_val) = plot_utils.get_good_min_max_range(depth_img[depth_img < max_depth])
	(min_tbin, max_tbin) = plot_utils.get_good_min_max_range(transient_img.argmax(axis=-1).squeeze())
	plt.clf()
	plt.subplot(3,2,1)
	plt.imshow(improc_ops.gamma_tonemap(rgb_img, gamma=1/4))
	plot_utils.remove_ticks()
	plt.title("RGB Image")
	plt.subplot(3,2,2)
	plt.imshow(depth_img, vmin=min_depth_val, vmax=max_depth_val)
	plot_utils.remove_ticks()
	plt.title("Depth Image")
	plt.subplot(3,2,3)
	plt.imshow(transient_img.argmax(axis=-1).squeeze(), vmin=min_tbin, vmax=max_tbin)
	plot_utils.remove_ticks()
	plt.title("Argmax of clean dToF image")
	plt.subplot(3,2,4)
	plt.imshow(transient_img_sim.argmax(axis=-1).squeeze(), vmin=min_tbin, vmax=max_tbin)
	plot_utils.remove_ticks()
	plt.title("Argmax of noisy dToF image")
	plt.subplot(3,2,5)
	plt.imshow(transient_img_sim.sum(axis=-1).squeeze())
	plot_utils.remove_ticks()
	plt.title("Sum of Transient")
	plt.subplot(3,2,6)
	plt.imshow(transient_img_sim.max(axis=-1).squeeze())
	plot_utils.remove_ticks()
	plt.title("Max Peak Height")
