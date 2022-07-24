'''
Description:
	This script will generate the estimated depths for the real-world experimental data results.
	This script:

	1. Loads the pre-processed experimental histogram data, which should have been downloaded and stored under `cvpr22_data/scan_data`
	2. Creates different coding scheme objects with which the data is going to be processed with
	3. Computes depths, point clouds, and depth errors for all coding schemes
		3.1 Depth errors are computed with respect to the method that uses the full-resolution histogram and uses a matched filter to compute depths.
	4. Saves the results depths, xyz, errors, etc in data files under `results_data/scan_data_results`

Parameters:

	If you want to change the scene edit the `scene_name` variable in this file. 

	If you want to change the number of coding functions used to generate the results change the `n_codes` variable. Note that some implemenations only work if `N` is divisible by `n_codes` (i.e., K).

To run : 
	`python scan_data_scripts/process_hist_img.py`

	Make sure to run from top-level folder

'''

#### Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')
sys.path.append('.')
sys.path.append('..')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from scipy.ndimage import gaussian_filter, median_filter
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from scan_data_scripts.scan_data_utils import *
from research_utils.timer import Timer
from research_utils.plot_utils import *
from toflib.coding_utils import create_basic_coding_obj
from toflib.coding import IdentityCoding, GrayCoding, TruncatedFourierCoding, PSeriesFourierCoding, PSeriesGrayCoding
from research_utils.io_ops import load_json
from research_utils import np_utils, improc_ops
from toflib import tof_utils

depth_offset = 0.0

def printDivisors(n) :
	i = 1
	while i <= n:
	    if (n % i==0):
	        print(i)
	    i = i + 1

def depths2xyz(depths, fov_major_axis=40, mask=None):
	(n_rows, n_cols) = depths.shape
	(fov_horiz, fov_vert) = improc_ops.calc_fov(n_rows, n_cols, fov_major_axis)
	(phi_img, theta_img) = improc_ops.calc_spherical_coords(fov_horiz, fov_vert, n_rows, n_cols, is_deg=True)
	depths+=depth_offset
	(x,y,z) = improc_ops.spherical2xyz(depths, phi_img, theta_img)
	zmap = np.array(z)
	if(not (mask is None)):
		(x,y,z) = (x[mask], y[mask], z[mask])
		zmap[np.logical_not(mask)] = np.nan
	xyz = np.concatenate((x.flatten()[...,np.newaxis], y.flatten()[...,np.newaxis], z.flatten()[...,np.newaxis]), axis=-1)	
	return (xyz, zmap)

def process_compressive_hist(hist_img, coding_id, n_codes, h_irf, account_irf, rec_algo_id='ncc', hist_tbin_size=8):
	'''
		hist_tbin_size in picosecs
	'''
	coding_obj = create_basic_coding_obj(coding_id, hist_img.shape[-1], n_codes, h_irf, account_irf)
	c_vals = coding_obj.encode(hist_img)
	lookup = coding_obj.reconstruction(c_vals, rec_algo_id=rec_algo_id).squeeze()
	decoded_tof = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo_id).squeeze()*hist_tbin_size
	return (decoded_tof, lookup, coding_obj)

def compose_output_fname(coding_id, n_codes, rec_algo, account_irf=True):
	out_fname = '{}_ncodes-{}_rec-{}'.format(coding_id, n_codes, rec_algo)
	if(account_irf):
		return out_fname + '-irf'
	else:
		return out_fname
	
def save_results(out_fpath, decoded_tof, decoded_xyz, decoded_zmap, tof_errs, masked_error_metrics, depth_errs, masked_depth_error_metrics, gt_tof, gt_xyz, medfilt_decoded_tof, medfilt_decoded_xyz):
	np.savez(out_fpath 
				, decoded_xyz=decoded_xyz
				, decoded_tof=decoded_tof
				, decoded_depths=tof_utils.time2depth(decoded_tof*1e-12)
				, decoded_zmap=decoded_zmap
				, tof_errs=tof_errs
				, depth_errs=depth_errs
				, masked_error_metrics=masked_error_metrics
				, masked_depth_error_metrics=masked_depth_error_metrics
				, gt_tof=gt_tof
				, gt_xyz=gt_xyz
				, medfilt_decoded_tof=medfilt_decoded_tof
				, medfilt_decoded_depths=tof_utils.time2depth(medfilt_decoded_tof*1e-12)
				, medfilt_decoded_xyz=medfilt_decoded_xyz
				)

def process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz):
	'''
	Outputs:
		* decoded_tof == Decoded per-pixel time-of-flight
		* decoded_xyz == Decoded per-pixel x,y,z coords using the fov_major_axis parameter
		* decoded_zmap == zmap extracted from xyz
		* tof_errs == tof errors computed wrt ground truth (in units of time)
		* error_metrics == tof errors metrics
		* depth_errs == depth errors computed wrt ground truth (in units of time)
		* depth_error_metrics == depth errors metrics
		NOTE: xyz and zmap will have the depth_offset vairable applied to them so they will not exactly match depthmaps obtained from decoded_tof.
	'''
	(decoded_tof, lookup, c_obj) = process_compressive_hist(hist_img, coding_id, n_codes, irf, account_irf, rec_algo_id=rec_algo_id, hist_tbin_size=hist_tbin_size)
	medfilt_decoded_tof = median_filter(decoded_tof, size=(3,3))
	(decoded_xyz, decoded_zmap) = depths2xyz(tof_utils.time2depth(decoded_tof*1e-12), fov_major_axis=scan_data_params['fov_major_axis'], mask=validsignal_mask)
	(medfilt_decoded_xyz, medfilt_decoded_zmap) = depths2xyz(tof_utils.time2depth(medfilt_decoded_tof*1e-12), fov_major_axis=scan_data_params['fov_major_axis'], mask=validsignal_mask)
	tof_errs = np.abs(gt_tof - decoded_tof)
	depth_errs=tof_utils.time2depth(tof_errs*1e-12)*1000
	error_metrics = np_utils.calc_error_metrics(tof_errs[validsignal_mask], delta_eps = hist_tbin_size)
	depth_error_metrics = np_utils.calc_error_metrics(depth_errs[validsignal_mask], delta_eps = 1000*tof_utils.time2depth(hist_tbin_size*1e-12))
	tof_errs[nosignal_mask] = np.nan
	depth_errs[nosignal_mask] = np.nan
	print("{}-{} Coding:".format(coding_id,rec_algo_id))
	np_utils.print_error_metrics(depth_error_metrics)
	out_fname = compose_output_fname(coding_id, n_codes, rec_algo=rec_algo_id, account_irf=account_irf)
	save_results(os.path.join(results_data_dirpath, out_fname+'.npz'), decoded_tof, decoded_xyz, decoded_zmap, tof_errs, error_metrics, depth_errs, depth_error_metrics, gt_tof, gt_xyz, medfilt_decoded_tof, medfilt_decoded_xyz)
	return (decoded_tof, lookup, c_obj, decoded_xyz, decoded_zmap, tof_errs, error_metrics, depth_errs, depth_error_metrics)


if __name__=='__main__':
	
	## Load parameters shared by all
	scan_data_params = load_json('scan_data_scripts/scan_params.json')
	io_dirpaths = load_json('io_dirpaths.json')
	results_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_dirpath'])
	results_data_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_data'], 'scan_data_results')
	results_dirpath = os.path.join(results_dirpath, 'real_data_results/hist_imgs')
	base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths["scan_data_base_dirpath"])

	## Load processed scene:
	# scene_id = '20190209_deer_high_mu/free'
	scene_id = '20190207_face_scanning_low_mu/free'
	# scene_id = '20190207_face_scanning_low_mu/ground_truth'
	assert(scene_id in scan_data_params['scene_ids']), "{} not in scene_ids".format(scene_id)
	dirpath = os.path.join(base_dirpath, scene_id)
	hist_dirpath = os.path.join(dirpath, 'intermediate_results')

	## Histogram image params
	is_unimodal = True # Leave as True
	use_scene_specific_irf = True # Leave as True
	downsamp_factor = 1 # Spatial downsample factor
	hist_tbin_factor = 1.0 # increase tbin size to make histogramming faster
	n_rows_fullres = scan_data_params['scene_params'][scene_id]['n_rows_fullres']
	n_cols_fullres = scan_data_params['scene_params'][scene_id]['n_cols_fullres']
	(nr, nc) = (n_rows_fullres // downsamp_factor, n_cols_fullres // downsamp_factor) # dims for face_scanning scene  
	min_tbin_size = scan_data_params['min_tbin_size'] # Bin size in ps
	hist_tbin_size = min_tbin_size*hist_tbin_factor # increase size of time bin to make histogramming faster
	hist_img_tau = scan_data_params['hist_preprocessing_params']['hist_end_time'] - scan_data_params['hist_preprocessing_params']['hist_start_time']
	nt = get_nt(hist_img_tau, hist_tbin_size)
	unimodal_nt = get_unimodal_nt(nt, scan_data_params['irf_params']['pulse_len'], hist_tbin_size)
	unimodal_hist_img_tau = unimodal_nt*hist_tbin_size
	if(is_unimodal):
		nt = unimodal_nt
		hist_img_tau = unimodal_hist_img_tau

	## Load histogram image
	hist_img_fname = get_hist_img_fname(nr, nc, hist_tbin_size, hist_img_tau, is_unimodal=is_unimodal)
	hist_img_fpath = os.path.join(hist_dirpath, hist_img_fname)
	hist_img = np.load(hist_img_fpath)

	## set the dirpath where we will save the results data
	results_data_dirpath = os.path.join(results_data_base_dirpath, scene_id, hist_img_fname.split('.npy')[0])
	os.makedirs(results_data_dirpath, exist_ok=True)

	global_shift = 0
	hist_img = np.roll(hist_img, global_shift, axis=-1)

	denoised_hist_img = gaussian_filter(hist_img, sigma=0.75, mode='wrap', truncate=1)
	(tbins, tbin_edges) = get_hist_bins(hist_img_tau, hist_tbin_size)

	## Load IRF
	irf_tres = scan_data_params['min_tbin_size'] # in picosecs

	if(use_scene_specific_irf):
		irf = get_scene_irf(scene_id, nt, tlen=hist_img_tau, is_unimodal=is_unimodal)
	else:
		irf = get_irf(nt, tlen=hist_img_tau, is_unimodal=is_unimodal)

	## Load nosignal mask
	nosignal_mask_fname = get_nosignal_mask_fname(n_rows_fullres, n_cols_fullres)
	nosignal_mask_fpath = os.path.join(hist_dirpath, nosignal_mask_fname)
	if(os.path.exists(nosignal_mask_fpath)):
		nosignal_mask = plt.imread(nosignal_mask_fpath)
		if(nosignal_mask.ndim==3): nosignal_mask = nosignal_mask[..., 0]
		nosignal_mask = np.round(transform.resize(nosignal_mask,output_shape=(nr,nc))).astype(int).astype(bool)
	else:
		print("No signal mask for {} does not exist...".format(scene_id))
		nosignal_mask = np.zeros((nr,nc)).astype(bool)
	validsignal_mask = np.logical_not(nosignal_mask)

	## Coding object for full histogram
	c_obj = IdentityCoding(hist_img.shape[-1], h_irf=irf, account_irf=True)
	# Get ground truth depths using a denoised histogram image
	gt_tof = c_obj.max_peak_decoding(denoised_hist_img, rec_algo_id='matchfilt').squeeze()*hist_tbin_size
	gt_depths = tof_utils.time2depth(gt_tof*1e-12)
	(gt_xyz, gt_zmap) = depths2xyz(tof_utils.time2depth(gt_tof*1e-12), fov_major_axis=scan_data_params['fov_major_axis'], mask=validsignal_mask)
	# Process FRG
	(coding_id, rec_algo_id) = ('Identity', 'matchfilt')
	(decoded_tof, lookup, c_obj, decoded_xyz, decoded_zmap, tof_errs, error_metrics, depth_tof_errs, depth_error_metrics) =  process_csph_full(hist_img, coding_id, nt, True, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)


	## estimated signal to background ratio
	nphotons = hist_img.sum(axis=-1)
	bkg_per_bin = np.median(hist_img, axis=-1) 
	signal = np.sum(hist_img - bkg_per_bin[...,np.newaxis], axis=-1)
	signal[signal < 0] = 0
	bkg = bkg_per_bin*nt
	sbr = signal / (bkg + 1e-3)
	np.savez(os.path.join(results_data_dirpath,'hist_stats.npz')
				, nphotons=nphotons
				, bkg_per_bin=bkg_per_bin
				, signal=signal
				, bkg=bkg
				, sbr=sbr
				)

	## Set CSPH params
	account_irf = True
	# Codes for N=1144: 8, 22, 26, 44, 52, 88, 104
	# Codes for N=832: 8, 16, 32, 52, 64, 104
	n_codes = 32
	out_fname_base = 'ncodes-{}_accountirf-{}'.format(n_codes, account_irf)

	(coding_id, rec_algo_id) = ('Gated', 'linear')
	(gated_decoded_tof, gated_lookup, gated_c_obj, gated_decoded_xyz, gated_decoded_zmap, gated_tof_errs, gated_error_metrics, gated_depth_errs, gated_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)

	(coding_id, rec_algo_id) = ('Gated', 'zncc')
	(gatedzncc_decoded_tof, gatedzncc_lookup, gatedzncc_c_obj, gatedzncc_decoded_xyz, gatedzncc_decoded_zmap, gatedzncc_tof_errs, gatedzncc_error_metrics, gatedzncc_depth_errs, gatedzncc_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)

	(coding_id, rec_algo_id) = ('PSeriesGray', 'ncc')
	(psergray_decoded_tof, psergray_lookup, psergray_c_obj, psergray_decoded_xyz, psergray_decoded_zmap, psergray_tof_errs, psergray_error_metrics, psergray_depth_errs, psergray_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)

	(coding_id, rec_algo_id) = ('PSeriesFourier', 'ncc')
	(pserfourier_decoded_tof, pserfourier_lookup, pserfourier_c_obj, pserfourier_decoded_xyz, pserfourier_decoded_zmap, pserfourier_tof_errs, pserfourier_error_metrics, pserfourier_depth_errs, pserfourier_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)

	(coding_id, rec_algo_id) = ('TruncatedFourier', 'ncc')
	(fourier_decoded_tof, fourier_lookup, fourier_c_obj, fourier_decoded_xyz, fourier_decoded_zmap, fourier_tof_errs, fourier_error_metrics, fourier_depth_errs, fourier_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)

	######## The following are follow-up implementations for Gray-based Fourier. These are not used in the paper, but they perform the same as PSeriesFourier
	# (coding_id, rec_algo_id) = ('PSeriesGrayBasedFourier', 'ncc')
	# (psergrayfourier_decoded_tof, psergrayfourier_lookup, psergrayfourier_c_obj, psergrayfourier_decoded_xyz, psergrayfourier_decoded_zmap, psergrayfourier_tof_errs, psergrayfourier_error_metrics, psergrayfourier_depth_errs, psergrayfourier_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)
	# (coding_id, rec_algo_id) = ('HybridGrayBasedFourier', 'ncc')
	# (hybridgrayfourier_decoded_tof, hybridgrayfourier_lookup, hybridgrayfourier_c_obj, hybridgrayfourier_decoded_xyz, hybridgrayfourier_decoded_zmap, hybridgrayfourier_tof_errs, hybridgrayfourier_error_metrics, hybridgrayfourier_depth_errs, hybridgrayfourier_depth_error_metrics) =  process_csph_full(hist_img, coding_id, n_codes, account_irf, rec_algo_id, hist_tbin_size, scan_data_params, validsignal_mask, gt_tof, gt_xyz)

	## Visualize depths
	(min_d, max_d) = get_depth_lims(scene_id)
	(min_err, max_err) = (0, 150)

	fg_px = (104,57)
	# fg_px2 = (79,109)
	# fg_px3 = (85,57)
	bkg_px = (24,80)

	plt.clf()
	plt.subplot(4,3,1)
	plt.imshow(decoded_tof, vmin=min_d, vmax=max_d, label="Full-Hist (N={}) \n Depths ".format(hist_img.shape[-1])); 
	plt.colorbar()
	plt.title("Full-Hist (N={}) \n Depths ".format(hist_img.shape[-1]))
	remove_ticks()
	plt.subplot(4,3,2)
	plt.imshow(tof_errs, vmin=min_err, vmax=max_err, label="Full-Hist (N={}) \n Errors".format(hist_img.shape[-1])); 
	plt.colorbar()
	plt.title("Full-Hist (N={}) \n Errors".format(hist_img.shape[-1]))
	remove_ticks()
	plt.subplot(4,3,3)
	plt.plot(hist_img[fg_px[0],fg_px[1],:] / hist_img[fg_px[0],fg_px[1],:].max(), linewidth=2, label='FG Hist ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(hist_img[bkg_px[0],bkg_px[1],:] / hist_img[bkg_px[0],bkg_px[1],:].max(), linewidth=2, label='BKG Hist ({},{})'.format(bkg_px[0], bkg_px[1]))
	# plt.plot(hist_img[fg_px2[0],fg_px2[1],:] / hist_img[fg_px2[0],fg_px2[1],:].max(), linewidth=2, label='Ear Hist ({},{})'.format(fg_px2[0], fg_px2[1]))
	# plt.plot(hist_img[fg_px3[0],fg_px3[1],:] / hist_img[fg_px3[0],fg_px3[1],:].max(), linewidth=2, label='Ear Hist ({},{})'.format(fg_px3[0], fg_px3[1]))
	plt.legend()
	plt.title("Foreground & Background \n Hist and Lookup Tables")
	plt.subplot(4,3,4)
	plt.imshow(psergray_decoded_tof, vmin=min_d, vmax=max_d, label="pserGray (K={}) \n Depths".format(psergray_c_obj.n_codes)); 
	plt.colorbar()
	plt.title("pserGray (K={}) \n Depths".format(psergray_c_obj.n_codes))
	remove_ticks()
	plt.subplot(4,3,5)
	plt.imshow(psergray_tof_errs, vmin=min_err, vmax=max_err, label="pserGray (K={}) \n Errors".format(psergray_c_obj.n_codes)); 
	plt.colorbar()
	plt.title("pserGray (K={}) \n Errors".format(psergray_c_obj.n_codes))
	remove_ticks()
	plt.subplot(4,3,6)
	plt.plot(hist_img[fg_px[0],fg_px[1],:] / hist_img[fg_px[0],fg_px[1],:].max(), linewidth=2, label='FG Hist ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(hist_img[bkg_px[0],bkg_px[1],:] / hist_img[bkg_px[0],bkg_px[1],:].max(), linewidth=2, label='BKG Hist ({},{})'.format(bkg_px[0], bkg_px[1]))
	plt.plot(psergray_lookup[fg_px[0],fg_px[1],:], linewidth=2, label='FG pserGray Lookup ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(psergray_lookup[bkg_px[0],bkg_px[1],:], linewidth=2, label='BKG Hist ({},{})'.format(bkg_px[0], bkg_px[1]))
	plt.subplot(4,3,7)
	plt.imshow(fourier_decoded_tof, vmin=min_d, vmax=max_d, label="Fourier (K={}) \n Depths".format(fourier_c_obj.n_codes)); 
	plt.colorbar()
	plt.title("Fourier (K={}) \n Depths".format(fourier_c_obj.n_codes))
	remove_ticks()
	plt.subplot(4,3,8)
	plt.imshow(fourier_tof_errs, vmin=min_err, vmax=max_err, label="Fourier (K={}) \n Errors".format(fourier_c_obj.n_codes)); 
	plt.colorbar()
	plt.title("Fourier (K={}) \n Errors".format(fourier_c_obj.n_codes))
	remove_ticks()
	plt.subplot(4,3,9)
	plt.plot(hist_img[fg_px[0],fg_px[1],:] / hist_img[fg_px[0],fg_px[1],:].max(), linewidth=2, label='FG Hist ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(hist_img[bkg_px[0],bkg_px[1],:] / hist_img[bkg_px[0],bkg_px[1],:].max(), linewidth=2, label='BKG Hist ({},{})'.format(bkg_px[0], bkg_px[1]))
	plt.plot(fourier_lookup[fg_px[0],fg_px[1],:], linewidth=2, label='FG Fourier Lookup ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(fourier_lookup[bkg_px[0],bkg_px[1],:], linewidth=2, label='BKG Fourier Hist ({},{})'.format(bkg_px[0], bkg_px[1]))

	plt.subplot(4,3,10)
	plt.imshow(pserfourier_decoded_tof, vmin=min_d, vmax=max_d, label="pserfourier (K={}) \n Depths".format(pserfourier_c_obj.n_codes)); 
	plt.colorbar()
	plt.title("pserfourier (K={}) \n Depths".format(pserfourier_c_obj.n_codes))
	remove_ticks()
	plt.subplot(4,3,11)
	plt.imshow(pserfourier_tof_errs, vmin=min_err, vmax=max_err, label="pserfourier (K={}) \n Errors".format(pserfourier_c_obj.n_codes)); 
	plt.colorbar()
	plt.title("pserfourier (K={}) \n Errors".format(pserfourier_c_obj.n_codes))
	remove_ticks()
	plt.subplot(4,3,12)
	plt.plot(hist_img[fg_px[0],fg_px[1],:] / hist_img[fg_px[0],fg_px[1],:].max(), linewidth=2, label='FG Hist ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(hist_img[bkg_px[0],bkg_px[1],:] / hist_img[bkg_px[0],bkg_px[1],:].max(), linewidth=2, label='BKG Hist ({},{})'.format(bkg_px[0], bkg_px[1]))
	plt.plot(pserfourier_lookup[fg_px[0],fg_px[1],:], linewidth=2, label='FG pserfourier Lookup ({},{})'.format(fg_px[0], fg_px[1]))
	plt.plot(pserfourier_lookup[bkg_px[0],bkg_px[1],:], linewidth=2, label='BKG pserfourier Hist ({},{})'.format(bkg_px[0], bkg_px[1]))

	out_fname = '{}_tres-{}ps_tlen-{}ps_globalshift-{}'.format(scene_id.replace('/','--'), int(irf_tres), int(hist_img_tau), global_shift)
	plt.pause(0.1)
	save_currfig_png(os.path.join(results_dirpath, 'globalshift'), out_fname)


	plt.show()