#### Standard Library Imports
import os
import json

#### Library imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


def load_json( json_filepath ):
	assert( os.path.exists( json_filepath )), "{} does not exist".format( json_filepath )
	with open( json_filepath, "r" ) as json_file: 
		return json.load( json_file )

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