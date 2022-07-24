'''
	Functions in this file use the same name as tof_utils.py. So naming conflicts may arise if you import all functions from both
	files.
'''
## Standard Library Imports

## Library Imports
import numpy as np
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from toflib import tof_utils
from research_utils.shared_constants import *

def linearize_phase(phase): 
	return tof_utils.linearize_phase(phase)
	
def phase2depth(phase, repetition_tau): 
	return tof_utils.phase2depth(phase, repetition_tau)

def phase2time(phase, repetition_tau):
	'''
		Assume phase is computed with np.atan2
	'''
	return tof_utils.phase2time(phase, repetition_tau)

def time2depth(time):
	return tof_utils.time2depth(time)

def depth2time(depth):
	return tof_utils.depth2time(depth)

def phasor2time(phasor, repetition_tau):
	phase = torch.angle(phasor)
	return phase2time(phase, repetition_tau)

def norm_t(C, axis=-1):
	'''
		Divide by standard deviation across given axis
	'''
	return C / (torch.norm(C, p=2, dim=axis, keepdim=True) + EPSILON)

def zero_norm_t(C, axis=-1):
	'''
		Apply zero norm transform to give axis
		This performs exactly the same as the old zero_norm_t_old, but in the old version denominator is scale by a factor of (1/sqrt(K)) which is part of the standard deviation formula
	'''
	return norm_t(C - C.mean(dim=axis, keepdim=True), axis=axis)

# def set_sbr(v, sbr=None, axis=-1):
# 	if(not (sbr is None)):
# 		# Make sbr an np array if needed and add a dimension to match v
# 		sbr_arr = np_utils.to_nparray(sbr).squeeze()
# 		assert((sbr_arr.ndim == 0) or ((sbr_arr.ndim+1) == v.ndim)), "incorrect input sbr dims"
# 		if(sbr_arr.ndim > 0): sbr_arr = np.expand_dims(sbr_arr, axis=axis)
# 		assert(np.all(sbr > 0)), "sbr needs to be > 0"
# 		n_photons = v.sum(axis=axis, keepdims=True)
# 		n_ambient_photons = n_photons / sbr_arr
# 		ambient = n_ambient_photons / v.shape[-1] 
# 		return v + ambient
# 	return v

# def set_signal_n_photons(v, n_photons=None, sbr=None, axis=-1):
# 	# Scale the signal according to n_photons if provided, otherwise use the signal in v
# 	if(not (n_photons is None)):
# 		assert(n_photons > 0), "n_photons need to be > 0"
# 		# if(n_photons < 8):
# 		#     print("WARNING: For very low photon counts you might see larger mean depth errors at farther depths. This is because at low photon counts you will start seeing cases where 0 photons are recorded. In these instances most depth estimation algorithms predict a depth = 0, which is favorable for shorter distrances.")
# 		v = v * (n_photons / v.sum(axis=axis, keepdims=True))
# 	v = set_sbr(v, sbr=sbr, axis=axis)
# 	return v

# def add_poisson_noise(transient, n_mc_samples=1):
# 	new_size = (n_mc_samples,) + transient.shape
# 	return np.random.poisson(lam=transient, size=new_size).astype(transient.dtype)

# def simulate_n_photons(transient, n_photons=1, n_mc_samples=1):
# 	# Vectorize transient such that it is only a 2D matrix where the last dimension is the time dim
# 	(transient_vec, transient_shape) = np_utils.vectorize_tensor(transient, axis=-1)
# 	n_elems = transient_vec.shape[0]
# 	tmp_size = (n_mc_samples,) + transient_vec.shape
# 	final_size = (n_mc_samples,) + transient.shape
# 	simulated_transient = np.zeros(tmp_size).astype(transient.dtype)
# 	# For each different transient draw n_photon samples suing the curr_transient as the probability distribution
# 	rng = np.random.default_rng()
# 	for i in range(n_elems):
# 		curr_transient = transient_vec[i,:]
# 		simulated_transient[:, i, :] = rng.multinomial(n=n_photons, pvals=curr_transient/curr_transient.sum(), size=n_mc_samples)
# 	# reshape to place the n_mc_samples dimension at the end
# 	return simulated_transient.reshape(final_size).squeeze()

# def depthmap2tirf(depthmap, n_tbins, delta_depth):
# 	'''
# 		Take each pixel of a depth image and turn it into a 1D delta temporal impulse response. The delta will be located according to the depth value.  
# 	'''
# 	# Transform depths to non-zero indeces
# 	nonzero_indeces = np.round(depthmap / delta_depth).astype(np.int) 
# 	tirf = np.zeros(depthmap.shape + (n_tbins,))
# 	for i in range(tirf.shape[0]):
# 		for j in range(tirf.shape[1]):
# 			tirf[i,j,nonzero_indeces[i,j]] = 1.
# 	return tirf

# def get_time_domain(repetition_tau, n_tbins):
# 	'''
# 		repetition_tau in seconds
# 		n_tbins number of time bins. 
# 	'''
# 	tbin_res = repetition_tau / n_tbins
# 	# The time domain refers to the exact time where we are sampling
# 	time_domain = np.arange(0, n_tbins)*tbin_res
# 	# The time bin bounds correspond to the edges of each histogram bin.
# 	tbin_bounds = (np.arange(0, n_tbins + 1) * tbin_res) - 0.5*tbin_res
# 	return (time_domain, tbin_res, tbin_bounds)

# def calc_tof_domain_params(n_tbins, rep_tau=None, max_path_length=None, max_depth=None):
# 	'''
# 		Set discrete time domain parameters.
# 		- If rep_tau is given use that, otherwise, if max_depth is given use that, and finally
# 		- rep_tau is expected to be in secs, and rep_freq will be in hz
# 	'''
# 	if(not (rep_tau is None)): max_depth = time2depth(rep_tau)
# 	elif(not (max_depth is None)): rep_tau = depth2time(max_depth)		
# 	elif(not (max_path_length is None)):
# 		max_depth = 0.5*max_path_length
# 		rep_tau = depth2time(max_depth)		
# 	else: rep_tau = 1
# 	(t_domain, tbin_res, tbin_bounds) = get_time_domain(rep_tau, n_tbins) 
# 	## Set rep frequency depending on the domain of the simulated transient
# 	rep_freq = 1 / rep_tau # In Hz
# 	tbin_depth_res = time2depth(tbin_res)
# 	return (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res)

