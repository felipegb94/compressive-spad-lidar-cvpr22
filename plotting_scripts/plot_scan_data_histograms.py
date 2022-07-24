'''
	plots isometric contours of a given coding approach with respect to the ideal setting where you have the full histogram available
'''
## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils import plot_utils, io_ops
from scan_data_scripts.scan_data_utils import get_hist_img_fname, get_nt, get_unimodal_nt

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	scan_params = io_ops.load_json('./scan_data_scripts/scan_params.json')
	base_dirpath = scan_params["scan_data_base_dirpath"]
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = io_dirpaths['paper_results_dirpath']
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"

	## Load processed scene:
	# scene_id = '20190209_deer_high_mu/free'
	scene_id = '20190207_face_scanning_low_mu/free'
	# scene_id = '20190207_face_scanning_low_mu/ground_truth'
		
	## Set parameters
	is_unimodal = True
	hist_tbin_factor = 1
	scene_id = '20190207_face_scanning_low_mu/free'
	# scene_id = '20190207_face_scanning_low_mu/ground_truth'
	assert(scene_id in scan_params['scene_ids']), "{} not in scene_ids".format(scene_id)
	## Dirpaths
	hist_dirpath = os.path.join(base_dirpath, scene_id, 'intermediate_results')
	
	## Get scan params
	nr = scan_params['scene_params'][scene_id]['n_rows_fullres']
	nc = scan_params['scene_params'][scene_id]['n_cols_fullres']
	min_tbin_size = scan_params['min_tbin_size'] # Bin size in ps
	hist_tbin_size = min_tbin_size*hist_tbin_factor # increase size of time bin to make histogramming faster
	hist_img_tau = scan_params['hist_preprocessing_params']['hist_end_time'] - scan_params['hist_preprocessing_params']['hist_start_time']
	nt = get_nt(hist_img_tau, hist_tbin_size)
	unimodal_nt = get_unimodal_nt(nt, scan_params['irf_params']['pulse_len'], hist_tbin_size)
	unimodal_hist_img_tau = unimodal_nt*hist_tbin_size
	if(is_unimodal):
		nt = unimodal_nt
		hist_img_tau = unimodal_hist_img_tau
	
	## Load Pre-processed Histogram Image
	hist_img_fname = get_hist_img_fname(nr, nc, hist_tbin_size, hist_img_tau, is_unimodal=is_unimodal)
	hist_img = np.load(os.path.join(hist_dirpath, hist_img_fname))
	d_hist_img = gaussian_filter(hist_img, sigma=1.25, mode='wrap', truncate=3)
	t_domain = np.arange(0,nt)*hist_tbin_size


	## Select which pixels to show
	if((scene_id=='20190207_face_scanning_low_mu/ground_truth') and ('r-204-c-116' in hist_img_fname)):
		rows = [109, 60, 17, 17  , 150  ]
		cols = [50,  97, 72, 109 , 109  ]
	elif((scene_id=='20190207_face_scanning_low_mu/free') and ('r-204-c-116' in hist_img_fname)):
		# rows = [109, 60, 17, 17, 150]
		# cols = [50,  97, 72, 109, 109]
		rows = [109, 150, 60]
		cols = [50,  109, 97]
		start_time = 1900 
	else:
		rows = [109, 17]
		cols = [50, 72]


	## Plot and save histograms
	out_dirpath = os.path.join(out_base_dirpath, 'scan_data_results', scene_id, hist_img_fname.split('.npy')[0])
	out_fname = 'ExampleUncompressedHistograms'
	os.makedirs(out_dirpath, exist_ok=True)

	plt.clf()
	for i in range(len(rows)):
		(r,c) = (rows[i], cols[i])
		plt.plot(t_domain, hist_img[r,c,:], linewidth=3, alpha=0.8**(i), label='R: {}, C: {} | Photon Cts: {}'.format(r,c, int(hist_img[r,c,:].sum())))

	plt.xlim([0, t_domain[-1]])
	# plt.grid()
	plot_utils.set_ticks(fontsize=18)
	plot_utils.update_fig_size(height=3, width=10)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	plt.legend(fontsize=18)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'-WithLegend', file_ext='svg')

	## Plot denoised hist image
	plt.clf()
	for i in range(len(rows)):
		(r,c) = (rows[i], cols[i])
		plt.plot(t_domain, d_hist_img[r,c,:], linewidth=3, alpha=0.8**(i), label='R: {}, C: {} | Photon Cts: {}'.format(r,c, int(hist_img[r,c,:].sum())))

	plt.xlim([0, t_domain[-1]])
	# plt.grid()
	plot_utils.set_ticks(fontsize=18)
	plot_utils.update_fig_size(height=3, width=10)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'_denoised', file_ext='svg')

	plt.legend(fontsize=18)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'-WithLegend_denoised', file_ext='svg')

	################ ZOOOMED Plots
	plt.clf()
	for i in range(len(rows)):
		(r,c) = (rows[i], cols[i])
		plt.plot(t_domain, hist_img[r,c,:], linewidth=3, alpha=0.8**(i), label='R: {}, C: {} | Photon Cts: {}'.format(r,c, int(hist_img[r,c,:].sum())))

	plt.xlim([start_time, t_domain[-1]])
	# plt.grid()
	plot_utils.set_ticks(fontsize=18)
	plot_utils.update_fig_size(height=3, width=10)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'_zoomed', file_ext='svg')

	plt.legend(fontsize=18)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'-WithLegend_zoomed', file_ext='svg')

	## Plot denoised hist image
	plt.clf()
	for i in range(len(rows)):
		(r,c) = (rows[i], cols[i])
		plt.plot(t_domain, d_hist_img[r,c,:], linewidth=3, alpha=0.8**(i), label='R: {}, C: {} | Photon Cts: {}'.format(r,c, int(hist_img[r,c,:].sum())))

	plt.xlim([start_time, t_domain[-1]])
	# plt.grid()
	plot_utils.set_ticks(fontsize=18)
	plot_utils.update_fig_size(height=3, width=10)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'_denoised_zoomed', file_ext='svg')

	plt.legend(fontsize=18)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname+'-WithLegend_denoised_zoomed', file_ext='svg')






