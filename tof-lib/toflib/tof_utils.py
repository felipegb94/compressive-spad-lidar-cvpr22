## Standard Library Imports

## Library Imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace
## Local Imports
from research_utils.shared_constants import *
from research_utils.timer import Timer
from research_utils import np_utils

def linearize_phase(phase):
	# If phase  < 0 then we need to add 2pi.
	corrected_phase = phase + (TWOPI*(phase < 0))
	return corrected_phase
	
def phase2depth(phase, repetition_tau):
	return time2depth(phase2time(phase, repetition_tau))

def phase2time(phase, repetition_tau):
	'''
		Assume phase is computed with np.atan2
	'''
	# If phase  < 0 then we need to add 2pi.
	corrected_phase = linearize_phase(phase)
	return (corrected_phase*repetition_tau / TWOPI )

def time2depth(time):
	return (SPEED_OF_LIGHT * time) / 2.

def freq2depth(freq):
	return (SPEED_OF_LIGHT * (1./freq)) / 2.

def depth2time(depth):
	return (2*depth /  SPEED_OF_LIGHT)

def phasor2time(phasor, repetition_tau):
	phase = np.angle(phasor)
	return phase2time(phase, repetition_tau)

def norm_t(C, axis=-1):
	'''
		Divide by standard deviation across given axis
	'''
	return C / (np.linalg.norm(C, ord=2, axis=axis, keepdims=True) + EPSILON)

def zero_norm_t(C, axis=-1):
	'''
		Apply zero norm transform to give axis
		This performs exactly the same as the old zero_norm_t_old, but in the old version denominator is scale by a factor of (1/sqrt(K)) which is part of the standard deviation formula
	'''
	return norm_t(C - C.mean(axis=axis, keepdims=True), axis=axis)

# def zero_norm_t_old(C, axis=-1):
# 	'''
# 		Apply zero norm transform to give axis
# 	'''
# 	return (C - C.mean(axis=axis, keepdims=True)) / (C.std(axis=axis, keepdims=True) + EPSILON)

def set_sbr(v, sbr=None, axis=-1, inplace=False):
	'''
		If inplace is False, return a copy of
	'''
	if(not (sbr is None)):
		# Make sbr an np array if needed and add a dimension to match v
		sbr_arr = np_utils.to_nparray(sbr).squeeze()
		assert((sbr_arr.ndim == 0) or ((sbr_arr.ndim+1) == v.ndim)), "incorrect input sbr dims"
		if(sbr_arr.ndim > 0): sbr_arr = np.expand_dims(sbr_arr, axis=axis)
		assert(np.all(sbr > 0)), "sbr needs to be > 0"
		n_photons = v.sum(axis=axis, keepdims=True)
		n_ambient_photons = n_photons / sbr_arr
		ambient = n_ambient_photons / v.shape[-1] 
		if(inplace):
			# do not create copy of vector
			v += ambient
			return v 
		else: 
			return v + ambient 
	return v

def set_signal_n_photons(v, n_photons=None, sbr=None, axis=-1, inplace=False):
	'''
		If inplace is False, return a copy of v scaled and vertically shifted according to n_photons and sbr
		If inplace is True, return v scaled and vertically shifted according to n_photons and sbr
	'''
	# Scale the signal according to n_photons if provided, otherwise use the signal in v
	if(not inplace): v_out = np.array(v)
	else: v_out = v
	if(not (n_photons is None)):
		n_photons_arr = np_utils.to_nparray(n_photons).squeeze()
		assert((n_photons_arr.ndim == 0) or ((n_photons_arr.ndim+1) == v.ndim)), "incorrect input n_photons dims"
		if(n_photons_arr.ndim > 0): n_photons_arr = np.expand_dims(n_photons_arr, axis=axis)
		assert(np.all(n_photons > 0)), "n_photons need to be > 0"
		# Set area under the curve to be n_photons_arr
		v_out *= (n_photons_arr / v.sum(axis=axis, keepdims=True))
	# Add constant offset = n_photons / sbr
	v_out = set_sbr(v_out, sbr=sbr, axis=axis, inplace=True)
	return v_out

def set_n_photons(v, n_photons=None, sbr=None, axis=-1): 
	return set_flux_n_photons(v, n_photons=n_photons, sbr=sbr, axis=axis) 

def set_flux_n_photons(v, n_photons=None, sbr=None, axis=-1):
	'''
		Returns a copy of v scaled and vertically shifted according to n_photons and sbr
	'''
	# Scale the signal according to n_photons if provided, otherwise use the signal in v
	v_out = np.array(v)
	# Add constant offset = n_photons / sbr
	v_out = set_sbr(v_out, sbr=sbr, axis=axis, inplace=True)
	# set area under the curve equal to n_photons 
	v_out = set_signal_n_photons(v_out, n_photons=n_photons, sbr=None, axis=axis, inplace=True)
	return v_out

def add_poisson_noise(transient, n_mc_samples=1):
	new_size = (n_mc_samples,) + transient.shape
	return np.random.poisson(lam=transient, size=new_size).astype(transient.dtype)

def simulate_n_photons(transient, n_photons=1, n_mc_samples=1):
	# Vectorize transient such that it is only a 2D matrix where the last dimension is the time dim
	(transient_vec, transient_shape) = np_utils.vectorize_tensor(transient, axis=-1)
	n_elems = transient_vec.shape[0]
	tmp_size = (n_mc_samples,) + transient_vec.shape
	final_size = (n_mc_samples,) + transient.shape
	simulated_transient = np.zeros(tmp_size).astype(transient.dtype)
	# For each different transient draw n_photon samples suing the curr_transient as the probability distribution
	rng = np.random.default_rng()
	for i in range(n_elems):
		curr_transient = transient_vec[i,:]
		# Need to case as float64 to avoid multinomial value error
		simulated_transient[:, i, :] = rng.multinomial(n=n_photons, pvals=curr_transient.astype(np.float64)/curr_transient.astype(np.float64).sum(), size=n_mc_samples)
	# reshape to place the n_mc_samples dimension at the end
	return simulated_transient.reshape(final_size).squeeze()

def depthmap2tirf(depthmap, n_tbins, delta_depth):
	'''
		Take each pixel of a depth image and turn it into a 1D delta temporal impulse response. The delta will be located according to the depth value.  
	'''
	# Transform depths to non-zero indeces
	nonzero_indeces = np.round(depthmap / delta_depth).astype(np.int) 
	tirf = np.zeros(depthmap.shape + (n_tbins,))
	for i in range(tirf.shape[0]):
		for j in range(tirf.shape[1]):
			tirf[i,j,nonzero_indeces[i,j]] = 1.
	return tirf

def get_time_domain(repetition_tau, n_tbins):
	'''
		repetition_tau in seconds
		n_tbins number of time bins. 
	'''
	tbin_res = repetition_tau / n_tbins
	# The time domain refers to the exact time where we are sampling
	time_domain = np.arange(0, n_tbins)*tbin_res
	# The time bin bounds correspond to the edges of each histogram bin.
	tbin_bounds = (np.arange(0, n_tbins + 1) * tbin_res) - 0.5*tbin_res
	return (time_domain, tbin_res, tbin_bounds)

def calc_tof_domain_params(n_tbins, rep_tau=None, max_path_length=None, max_depth=None):
	'''
		Set discrete time domain parameters.
		- If rep_tau is given use that, otherwise, if max_depth is given use that, and finally
		- rep_tau is expected to be in secs, and rep_freq will be in hz
	'''
	if(not (rep_tau is None)): max_depth = time2depth(rep_tau)
	elif(not (max_depth is None)): rep_tau = depth2time(max_depth)		
	elif(not (max_path_length is None)):
		max_depth = 0.5*max_path_length
		rep_tau = depth2time(max_depth)		
	else: rep_tau = 1
	(t_domain, tbin_res, tbin_bounds) = get_time_domain(rep_tau, n_tbins) 
	## Set rep frequency depending on the domain of the simulated transient
	rep_freq = 1 / rep_tau # In Hz
	tbin_depth_res = time2depth(tbin_res)
	return (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res)

def hist2timestamps(hist_tensor, max_n_timestamps=None):
	'''
		Input:
			* hist_tensor: Tensor whose last dimension is the histogram dimension. Example a tensor with dimsn n_rows x n_cols x n_hist_bins
			* max_n_timestamps: Max number of timestamps that we will accept. If None, then this is derived from the hist with the most timestamps
		Output
			* timestamps_tensor: tensor whose first K-1 dimensions are equal to the hist_tensor. The last dimension depends on max_n_timestamps
	'''
	(hist_tensor, hist_shape) = np_utils.vectorize_tensor(hist_tensor)
	hist_tensor = hist_tensor.astype(int)
	n_hists = hist_tensor.shape[0]
	n_bins = hist_tensor.shape[-1]
	n_timestamps_per_hist = hist_tensor.sum(axis=-1)
	if(max_n_timestamps is None): max_n_timestamps = np.max(n_timestamps_per_hist)
	timestamp_tensor = -1*np.ones((n_hists, max_n_timestamps)).astype(np.int)
	n_timestamp_per_elem = np.zeros((n_hists,)).astype(np.int)
	for i in range(n_hists):
		curr_hist = hist_tensor[i]
		tmp_timestamp_arr = -1*np.ones((n_timestamps_per_hist[i],))
		curr_idx = 0
		for j in range(n_bins):
			curr_bin_n = curr_hist[j]
			if(curr_bin_n > 0):
				tmp_timestamp_arr[curr_idx:curr_idx+curr_bin_n] = j
				curr_idx = curr_idx+curr_bin_n
		# If number of timestamps is larger than max_n_timestamps, randomly sample max_n
		if(n_timestamps_per_hist[i] >= max_n_timestamps):
			timestamp_tensor[i,:] = np.random.choice(tmp_timestamp_arr, size=(max_n_timestamps,), replace=False)
			n_timestamp_per_elem[i] = max_n_timestamps
		else:
			timestamp_tensor[i,0:n_timestamps_per_hist[i]] = tmp_timestamp_arr
			n_timestamp_per_elem[i] = n_timestamps_per_hist[i]
	return timestamp_tensor.reshape(hist_shape[0:-1] + (max_n_timestamps,)),  n_timestamp_per_elem.reshape(hist_shape[0:-1])

def timestamps2hist(timestamp_tensor, n_timestamps_per_elem, n_bins):
	(timestamp_tensor, tensor_shape) = np_utils.vectorize_tensor(timestamp_tensor)
	n_elems = timestamp_tensor.shape[0]
	n_timestamps_per_elem = n_timestamps_per_elem.reshape((n_elems,))
	hist_tensor = np.zeros((n_elems, n_bins)).astype(timestamp_tensor.dtype)
	for i in range(timestamp_tensor.shape[0]):
		for j in range(n_timestamps_per_elem[i]):
			hist_tensor[i, timestamp_tensor[i, j]] += 1
	return hist_tensor.reshape(tensor_shape[0:-1] + (n_bins,))

