'''
	Useful function when operating on numpy arrays
	This module should not depend on anything else other than numpy.
'''
#### Standard Library Imports

#### Library imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from .shared_constants import *


def vectorize_tensor(tensor, axis=-1):
	'''
		Take an N-Dim Tensor and make it a 2D matrix. Leave the first or last dimension untouched, and basically squeeze the 1st-N-1
		dimensions.
		This is useful when applying operations on only the first or last dimension of a tensor. Makes it easier to input to different
		number of pytorch functions.
	'''
	assert((axis==0) or (axis==-1)), 'Error: Input axis needs to be the first or last axis of tensor'
	tensor_shape = tensor.shape
	n_untouched_dim = tensor.shape[axis]
	n_elems = int(round(tensor.size / n_untouched_dim))
	if(axis == -1):
		return (tensor.reshape((n_elems, n_untouched_dim)), tensor_shape)
	else:
		return (tensor.reshape((n_untouched_dim, n_elems)), tensor_shape)

def unvectorize_tensor(tensor, tensor_shape):
	'''
		Undo vectorize_tensor operation
	'''
	return tensor.reshape(tensor_shape)

def to_nparray( a ):
	'''
		cast to np array. If a is scalar, make it a 1D 1 element vector
	'''
	# Don't to anything if a is already an numpy array
	if(isinstance(a, np.ndarray)): return a
	# Turn it into a numpy array
	a_arr = np.array(a)
	# if it was a scalar add dimension
	if(a_arr.ndim == 0): return a_arr[np.newaxis]
	# otherwise simply return the new nparray
	return a_arr

def extend_tensor_circularly(tensor, axis=-1):
	'''
		Take a tensor of any dimension and create a new tensor that is 3x longer along the speified axis
		We take concatenate 3 copies of the tensor along the specified axis 
	'''
	return np.concatenate((tensor, tensor, tensor), axis=axis)

def get_extended_domain(domain, axis=-1):
	'''
		Take a domain defined between [min_val, max_val] with n elements. Extend it along both directions.
		So if we have the domain = [0, 1, 2, 3]. Then we output: [-4,-3,-2,-1,  0,1,2,3,  4,5,6,7] 
	'''
	n = domain.shape[axis]
	min_val = domain.min(axis=axis)
	assert(min_val >= 0), "get_extended_domain currentl only works for non-negative domains"
	max_val = domain.max(axis=axis)
	delta = domain[1] - domain[0]
	domain_left = domain-(max_val + delta)
	domain_right = domain+(max_val + delta)
	return np.concatenate((domain_left, domain, domain_right), axis=axis)

def calc_mean_percentile_errors(errors, percentiles=[0.5, 0.75, 0.95, 0.99]):
	'''
		Sort the errors from lowest to hightest.
		Given a list of percentiles calculate the mean of the sorted errors within each percentile.
		For instance, if percentiles=[0.5,0.75,1.0], then
		we calculate the mean of the lowest 50% errors, then the mean of the errors in the 50-75% percentile, 
		and finally the errors in the 75-100% percentile.
	'''
	errors_shape = errors.shape
	errors = errors.flatten()
	n_elems = errors.size
	# Sort errors
	sorted_errors = np.sort(errors)
	# Verify the input percentiles and find the indeces where we split the errors
	percentiles = to_nparray(percentiles)
	assert(not (np.any(percentiles > 1) or np.any(percentiles < 0))), "Percentiles need to be between 0 and 1"
	percentile_indeces = np.round(n_elems*percentiles).astype(np.int)
	# Calculate mean for each percentile
	percentile_mean_errors = np.zeros_like(percentiles)
	percentile_mask = np.zeros_like(errors)-1.
	for i in range(percentiles.size):
		start_idx = 0
		if(i > 0): start_idx = percentile_indeces[i-1] 
		end_idx = percentile_indeces[i]
		percentile_mean_errors[i] = np.mean(sorted_errors[start_idx:end_idx])
		# Find which pixels were used to calculate this percentile mae
		low_error_threshold = sorted_errors[start_idx]
		high_error_threshold = sorted_errors[end_idx-1]
		percentile_mask[np.logical_and(errors >= low_error_threshold, errors <= high_error_threshold)] = i
	errors = errors.reshape(errors_shape)
	percentile_mask = percentile_mask.reshape(errors_shape)
	return (percentile_mean_errors, percentile_mask)

def calc_eps_tolerance_error(errors, eps = 0.):
	assert(eps >= 0.), "eps should be non-negative"
	n_eps_tol_errors = np.sum(errors <= (eps + EPSILON)).astype(errors.dtype)
	return n_eps_tol_errors / errors.size

def calc_error_metrics(errors, percentiles=[0.5, 0.75, 0.95, 0.99], eps_list=[1.], delta_eps = 1.):
	'''
		delta_eps is the delta for the X_epsilon errors
	'''
	metrics = {}
	metrics['mae'] = np.mean(errors)
	metrics['rmse'] = np.sqrt(np.mean(np.square(errors)))
	metrics['medae'] = np.median(errors)
	(percentile_mean_errors, percentile_mask) = calc_mean_percentile_errors(errors, percentiles=percentiles)
	metrics['percentile_mae'] = percentile_mean_errors
	metrics['percentiles'] = percentiles
	assert(delta_eps > 0.), "delta_eps should be nonnegative"
	scaled_errors = errors / delta_eps
	metrics['0_tol_errs'] = calc_eps_tolerance_error(scaled_errors, eps = 0.)
	for i in range(len(eps_list)):
		metrics['{}_tol_errs'.format(int(eps_list[i]))] = calc_eps_tolerance_error(scaled_errors, eps = eps_list[i])
	return metrics

def print_error_metrics(metrics, prefix=''):
	print("{} mae = {:.2f}".format(prefix, metrics['mae']))
	# print("{} rmse = {:.2f}".format(prefix, metrics['rmse']))
	print("{} medae = {:.2f}".format(prefix, metrics['medae']))
	np.set_printoptions(suppress=True)
	print("{} percentile_mae = {}".format(prefix, metrics['percentile_mae'].round(decimals=2)))
	np.set_printoptions(suppress=False)
	print("{} 1_tol_errs = {:.2f}".format(prefix, metrics['1_tol_errs']))
	# print("{} 0_tol_errs = {}".format(prefix, metrics['0_tol_errs']))

def domain2index(val, max_domain_val, n, is_circular=True):
	'''
		Assumes domain is between 0 and max_domain_val
	'''
	delta = max_domain_val / n
	indeces = np.array(np.round(val / delta)).astype(np.int32)
	if(is_circular): indeces[indeces == n] = 0 # Wrap around the indeces that were closer to the top boundary
	else: indeces[indeces == n] = n-1 # do not wrap around if domain is not circular
	return indeces

def are_orthogonal(v, u):
	'''
		Check if v is orthogonal to u
	'''
	assert(v.ndim == 1), "v should be a vector"
	assert(u.ndim == 1), "u should be a vector"
	assert(u.shape == v.shape), "u and v should match dims"
	return np.abs(np.dot(v, u) / v.size) <= EPSILON

def is_mutually_orthogonal(X):
	'''
		Check if all cols are mutually orthogonal
	'''
	assert(X.ndim == 2), "X should be a matrix"
	(n_rows, n_cols) = X.shape
	for i in range(n_cols):
		v_i = X[:, i]
		for j in range(n_cols):
			v_j = X[:, j]
			# If i equals j skip,  If vectors are not orthogonal return false
			if((i != j) and (not are_orthogonal(v_i, v_j))):
				return False
	return True

def circular_signal_fit(signal):
	'''
		Fit a periodic signal whose domain is assumed to be between 0-1
	'''
	## Fit a cubic spline function to be able to generate any 
	from scipy.interpolate import interp1d
	nt = signal.size
	# Extend x and y and interpolate
	ext_x_fullres = np.arange(-nt, 2*nt) * (1. / nt)
	ext_signal = np.concatenate((signal, signal, signal), axis=-1)
	f = interp1d(ext_x_fullres, ext_signal, axis=-1, kind='cubic')
	return f