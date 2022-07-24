#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt

#### Local imports

irf_dirpath = './scan_data_scripts/system_irf'

def verify_hist_tau(hist_img_tau, hist_tbin_size):
	if((hist_img_tau % hist_tbin_size) != 0):
		print("Invalid hist tau. Try adding {} to end time".format(hist_img_tau % hist_tbin_size))
	assert((hist_img_tau % hist_tbin_size) == 0), "hist tau needs to be a multiple of the bin size"

def time2bin(t, bin_size):
	'''
		time2bin index. Bin at index 0 will have times from 0-1 (not including 1)
	'''
	return int(np.floor(t / bin_size))

def bin2time(bin, bin_size):
	'''
		bin2time. given bin index return the mid point in time of that bin
	'''
	return bin*bin_size + (0.5*bin_size)

def get_unimodal_nt(nt, pulse_len, tres):
	return nt - time2bin(pulse_len, tres)

def get_nt(hist_img_tau, hist_tbin_size):
	verify_hist_tau(hist_img_tau, hist_tbin_size)
	return time2bin(hist_img_tau, hist_tbin_size)

def get_hist_bins(max_tbin, tbin_size):
	verify_hist_tau(max_tbin, tbin_size)
	bin_edges = np.arange(0, max_tbin + tbin_size, tbin_size)
	bins = bin_edges[0:-1] + 0.5*tbin_size
	return (bins, bin_edges)

def timestamps2histogram(tstamps_vec, max_tbin, min_tbin_size, hist_tbin_factor=1):
	''' Build histogram from timestamps loaded by the above functions
	Outputs:
		* tstamps_vec: unitless timestamps
		* max_tbin: maximum tbin value
		* min_tbin_size: time resolution. tstamps_vec*min_tbin_size == tstamps in time units
		* hist_tbin_factor: If we want to make histogram smaller. If set to 2 the histogram will be 2x smaller, 3 --> 3x smaller, etc.
	'''
	hist_tbin_size = min_tbin_size*hist_tbin_factor # increase size of time bin to make histogramming faster
	tstamps_vec = min_tbin_size*tstamps_vec # time counter to timestamps
	(bins, bin_edges) = get_hist_bins(max_tbin, hist_tbin_size)
	# Use Numpy histogram function (much faster)
	counts, _ = np.histogram(tstamps_vec, bins=bin_edges)
	# counts, _, _ = plt.hist(tstamps_vec,bins=bins)
	# plt.clf()
	return (counts, bin_edges, bins)

def vector2img(v, nr, nc):
	'''
		Transform vectorized pixels to img. This function is specifically tailored to the way that scan data was acquired
	'''
	assert((nr*nc) == v.shape[0]), "first dim length should equal num pixels"
	assert((v.ndim == 1) or (v.ndim == 2)), "should be a vector or an array of vectors"
	if(v.ndim == 1):
		img = v.reshape((nr, nc))
	else:
		img = v.reshape((nr, nc, v.shape[-1]))
	return np.flipud(np.swapaxes(img, 0, 1))

def get_hist_img_fname(nr, nc, tres, tlen, is_unimodal=False):
	if(is_unimodal):
		return 'unimodal-hist-img_r-{}-c-{}_tres-{}ps_tlen-{}ps.npy'.format(nr, nc, int(tres), int(tlen))
	else:
		return 'hist-img_r-{}-c-{}_tres-{}ps_tlen-{}ps.npy'.format(nr, nc, int(tres), int(tlen))

def get_irf_fname(tres, tlen, is_unimodal=False): 
	if(is_unimodal):
		return 'unimodal-irf_tres-{}ps_tlen-{}ps.npy'.format(int(tres), int(tlen))
	else:
		return 'irf_tres-{}ps_tlen-{}ps.npy'.format(int(tres), int(tlen))

def get_nosignal_mask_fname(nr, nc): 
	return 'nosignal-mask_r-{}-c-{}.png'.format(nr, nc)

def fit_irf(irf):
	## Fit a cubic spline function to be able to generate any 
	from scipy.interpolate import interp1d
	nt = irf.size
	# Extend x and y and interpolate
	ext_x_fullres = np.arange(-nt, 2*nt) * (1. / nt)
	ext_irf = np.concatenate((irf, irf, irf), axis=-1)
	f = interp1d(ext_x_fullres, ext_irf, axis=-1, kind='cubic')
	return f

def get_irf(n, tlen, tres=8, is_unimodal=False):
	'''
		Load IRF data stored for a particular histogram length (tlen)
		Fit a curve to the data, and then re-sample it at the desired resolution (n)
		PARAMETERS:
			* n = desired resolution of irf
			* tlen = length of irf in picoseconds
			* tres = time resolution of irf data
		NOTE: The IRF data is usually saved at the lowest tres possible (8ps)
	'''
	irf_data_fname = get_irf_fname(tres, tlen, is_unimodal)
	irf_data_fpath = os.path.join(irf_dirpath, irf_data_fname)
	assert(os.path.exists(irf_data_fpath)), "irf does not exist. make sure to run preprocess_irf.py for this hist len first"
	irf_data = np.load(irf_data_fpath)
	irf_f = fit_irf(irf_data)
	x_fullres = np.arange(0, n)*(1./n)
	irf = irf_f(x_fullres)
	irf[irf < 1e-8] = 0
	return irf

def get_scene_irf(scene_id, n, tlen, tres=8, is_unimodal=False):
	'''
		Load IRF data stored for a particular histogram length (tlen)
		Fit a curve to the data, and then re-sample it at the desired resolution (n)
		PARAMETERS:
			* n = desired resolution of irf
			* tlen = length of irf in picoseconds
			* tres = time resolution of irf data
		NOTE: The IRF data is usually saved at the lowest tres possible (8ps)
	'''
	irf_data_fname = get_irf_fname(tres, tlen, is_unimodal)
	irf_data_fpath = os.path.join(os.path.join(irf_dirpath, scene_id), irf_data_fname)
	assert(os.path.exists(irf_data_fpath)), "irf does not exist. make sure to run preprocess_irf.py for this hist len first"
	irf_data = np.load(irf_data_fpath)
	irf_f = fit_irf(irf_data)
	x_fullres = np.arange(0, n)*(1./n)
	irf = irf_f(x_fullres)
	irf[irf < 1e-8] = 0
	return irf


def get_depth_lims(scene_id):
	# if(scene_id == '20190207_face_scanning_low_mu/ground_truth'): (min_d, max_d) = (2300, 3600) 
	# elif(scene_id == '20190207_face_scanning_low_mu/free'): (min_d, max_d) = (3200, 4500) 
	# else: (min_d, max_d) = (1000, 4250)
	if(scene_id == '20190207_face_scanning_low_mu/ground_truth'): (min_d, max_d) = (1100, 2400) 
	elif(scene_id == '20190207_face_scanning_low_mu/free'): (min_d, max_d) = (2000, 3300) 
	else: (min_d, max_d) = (300, 2600)
	return (min_d, max_d)

def calc_n_empty_laser_cycles(sync_vec):
	max_laser_cycles = sync_vec.max()
	u, c = np.unique(sync_vec, return_counts=True)
	n_nonempty_laser_cycles = u.size
	assert(max_laser_cycles >= n_nonempty_laser_cycles), "something is wrong with sync_vec"
	return max_laser_cycles - n_nonempty_laser_cycles
