'''
	Base class for temporal coding schemes
'''
## Standard Library Imports
from abc import ABC, abstractmethod
import math 
import os
import warnings

## Library Imports
import numpy as np
import scipy
from scipy import signal, interpolate
from scipy.special import softmax
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from toflib.tof_utils import norm_t, zero_norm_t, linearize_phase, hist2timestamps, timestamps2hist
from toflib import tirf
from research_utils.np_utils import to_nparray
from research_utils.shared_constants import *
from research_utils import signalproc_ops, np_utils, py_utils


class Coding(ABC):
	'''
		Abstract class for linear coding
	'''
	C = None
	h_irf = None
	def __init__(self, h_irf=None, account_irf=False):
		# Set the coding matrix C if it has not been set yet
		if(self.C is None): self.set_coding_mat()
		# 
		(self.n_maxres, self.n_codes) = (self.C.shape[-2], self.C.shape[-1])
		# Set the impulse response function (used for accounting for system band-limit and match filter reconstruction)
		self.update_irf(h_irf)
		# the account_irf flag controls if we want to account IRF when estimating shifts. 
		# This means that the C matrices used during decoding may be different from the encoding one
		self.account_irf = account_irf
		# Update all the parameters derived from C
		self.update_C_derived_params()
		# Begin with lres mode as false
		self.lres_mode = False
		# Get all functions for reconstruction and decoding available
		self.rec_algos_avail = py_utils.get_obj_functions(self, filter_str='reconstruction')
		self.compatible_dualres_rec_algos = ['zncc']
		# Set if we want to account for IRF when decoding

	@abstractmethod
	def set_coding_mat(self):
		'''
		This method initializes the coding matrix, self.C
		'''
		pass

	def update_irf(self, h_irf=None):
		# If nothing is given set to gaussian
		if(h_irf is None): 
			print("hirf is NONE")
			self.h_irf = tirf.GaussianTIRF(self.n_maxres, mu=0, sigma=1).tirf.squeeze()
		else: self.h_irf = h_irf.squeeze()
		assert(self.h_irf.ndim == 1), "irf should be 1 dim vector"
		assert(self.h_irf.shape[-1] == self.n_maxres), "irf size should match n_maxres"
		assert(np.all(self.h_irf >= 0.)), "irf should be non-negative"
		# normalize
		self.h_irf = self.h_irf / self.h_irf.sum() 

	def update_C(self, C=None):
		if(not (C is None)): self.C = C
		# update any C derived params
		self.update_C_derived_params()
	
	def update_C_derived_params(self):
		# Store how many codes there are
		(self.n_maxres, self.n_codes) = (self.C.shape[-2], self.C.shape[-1])
		assert(self.n_codes <= self.n_maxres), "n_codes ({}) should not be larger than n_maxres ({})".format(self.n_codes, self.n_maxres)
		if(self.account_irf):
			# self.decoding_C = signalproc_ops.circular_conv(self.C, self.h_irf[:, np.newaxis], axis=0)
			# self.decoding_C = signalproc_ops.circular_corr(self.C, self.h_irf[:, np.newaxis], axis=0)
			self.decoding_C = signalproc_ops.circular_corr(self.h_irf[:, np.newaxis], self.C, axis=0)
		else:
			self.decoding_C = self.C
		# Pre-initialize some useful variables
		self.zero_norm_C = zero_norm_t(self.decoding_C)
		self.norm_C = norm_t(self.decoding_C)
		# Create lres coding mats
		self.lres_factor = 10
		self.lres_n = int(np.floor(self.n_maxres / self.lres_factor))
		self.lres_C = signal.resample(x=self.decoding_C, num=self.lres_n, axis=-2) 
		self.lres_zero_norm_C = zero_norm_t(self.lres_C)
		self.lres_norm_C = norm_t(self.lres_C)
		# Set domains
		self.domain = np.arange(0, self.n_maxres)*(TWOPI / self.n_maxres)
		self.lres_domain = np.arange(0, self.lres_n)*(TWOPI / self.lres_n)

	def get_n_maxres(self):
		if(self.lres_mode): return self.lres_n
		else: return self.n_maxres

	def get_domain(self):
		if(self.lres_mode): return self.lres_domain
		else: return self.domain

	def get_input_C(self, input_C=None):
		if(input_C is None):
			if(self.lres_mode): input_C = self.lres_C
			else: input_C = self.C
		self.verify_input_c_vec(input_C) # Last dim should be the codes
		return input_C

	def get_input_zn_C(self, zn_input_C=None):
		if(zn_input_C is None):
			if(self.lres_mode): zn_input_C = self.lres_zero_norm_C
			else: zn_input_C = self.zero_norm_C
		self.verify_input_c_vec(zn_input_C) # Last dim should be the codes
		return zn_input_C
		
	def get_input_norm_C(self, norm_input_C=None):
		if(norm_input_C is None):
			if(self.lres_mode): norm_input_C = self.lres_norm_C
			else: norm_input_C = self.norm_C
		self.verify_input_c_vec(norm_input_C) # Last dim should be the codes
		return norm_input_C

	def encode(self, transient_img):
		'''
		Encode the transient image using the n_codes inside the self.C matrix
		'''
		assert(transient_img.shape[-1] == self.n_maxres), "Input c_vec does not have the correct dimensions"
		return np.matmul(transient_img[..., np.newaxis, :], self.C).squeeze(-2)

	def verify_input_c_vec(self, c_vec):
		assert(c_vec.shape[-1] == self.n_codes), "Input c_vec does not have the correct dimensions"

	def zncc_reconstruction(self, c_vec, input_C=None, c_vec_is_zero_norm=False):
		'''
		ZNCC Reconstruction: Works for any arbitrary set of codes
		'''
		self.verify_input_c_vec(c_vec)
		# Make c_vec zero norm if needed
		if(not c_vec_is_zero_norm): zero_norm_c_vec = zero_norm_t(c_vec, axis=-1)
		else: zero_norm_c_vec = c_vec
		# If no input_C is provided use one of the existing ones
		input_C = self.get_input_zn_C(input_C)
		# Perform zncc
		return np.matmul(input_C, zero_norm_c_vec[..., np.newaxis]).squeeze(-1)

	def ncc_reconstruction(self, c_vec, input_C=None, c_vec_is_norm=False):
		'''
		NCC Reconstruction: Works for any arbitrary set of zero-mean codes
		'''
		self.verify_input_c_vec(c_vec)
		# Make c_vec zero norm if needed
		if(not c_vec_is_norm): norm_c_vec = norm_t(c_vec, axis=-1)
		else: norm_c_vec = c_vec
		# If no input_C is provided use one of the existing ones
		input_C = self.get_input_norm_C(input_C)
		# Perform zncc
		return np.matmul(input_C, norm_c_vec[..., np.newaxis]).squeeze(-1)
	
	def basis_reconstruction(self, c_vec, input_C=None):
		'''
		Basis reconstruction: If the codes in C are orthogonal to each other this reconstruction seems to work fine. 
		'''
		self.verify_input_c_vec(c_vec)
		input_C = self.get_input_C(input_C=input_C)
		return np.matmul(input_C, c_vec[..., np.newaxis]).squeeze(-1)

	def get_rec_algo_func(self, rec_algo_id):
		# Check if rec algorithm exists
		rec_algo_func_name = '{}_reconstruction'.format(rec_algo_id)
		rec_algo_function = getattr(self, rec_algo_func_name, None)
		assert(rec_algo_function is not None), "Reconstruction algorithm {} is NOT available. Please choose from the following algos: {}".format(rec_algo_func_name, self.rec_algos_avail)
		# # Apply rec algo
		# print("Running reconstruction algorithm {}".format(rec_algo_func_name))
		return rec_algo_function
	
	def reconstruction(self, c_vec, rec_algo_id='zncc', **kwargs):
		rec_algo_function = self.get_rec_algo_func(rec_algo_id)
		lookup = rec_algo_function(c_vec, **kwargs)
		return lookup

	def max_peak_decoding(self, c_vec, rec_algo_id='zncc', **kwargs):
		'''
			Perform max peak decoding using the specified reconstruction algorithm
			kwargs (key-work arguments) will depend on the chosen reconstruction algorithm 
		'''
		lookup = self.reconstruction(c_vec, rec_algo_id, **kwargs)
		return np.argmax(lookup, axis=-1)

	def softmax_peak_decoding(self, c_vec, rec_algo_id='zncc', beta=100, **kwargs):
		'''
			Perform max peak decoding using the specified reconstruction algorithm
			kwargs (key-work arguments) will depend on the chosen reconstruction algorithm 
		'''
		lookup = self.reconstruction(c_vec, rec_algo_id, **kwargs)
		domain = np.arange(0, lookup.shape[-1]).astype(lookup.dtype)
		return np.matmul(softmax(beta*lookup, axis=-1), domain[:, np.newaxis]).squeeze(-1)

	def maxgauss_peak_decoding(self, c_vec, gauss_sigma, rec_algo_id='zncc', **kwargs):
		lookup = self.reconstruction(c_vec, rec_algo_id, **kwargs)
		return signalproc_ops.max_gaussian_center_of_mass_mle(lookup, sigma_tbins = gauss_sigma)

	def zncc_depth_decoding(self, c_vec, input_C=None, c_vec_is_zero_norm=False):
		return self.max_peak_decoding(c_vec, rec_algo_id='zncc', input_C=input_C, c_vec_is_zero_norm=c_vec_is_zero_norm)

	def dualres_zncc_depth_decoding(self, c_vec):
		'''
			Dualres Coarse-Fine ZNCC Depth Decoding. Can provide large speedups over zncc_depth_decoding. 
			In the minimal number of tests I have done I have observed 3-10x speedups
			Still trying to figure out how Fourier Coding can make use of this
		'''
		# print("WARNING: Dualres Coarse-Fine Depth Decoding has not been extensively tested. It seems to work fine, but sometimes it does get the wrong depth in a few pixels (very few though).")
		print("Dualres Coarse-Fine Depth Decoding")
		assert((self.n_maxres % self.lres_factor) == 0), "Coarse-Fine Depth Decoding has only been tested with self.lres_factor ({}) multiple of self.n_maxres ({})".format(self.lres_factor, self.n_maxres)
		self.verify_input_c_vec(c_vec)
		zero_norm_c_vec = zero_norm_t(c_vec, axis=-1)
		# Get Coarse Depth Map
		self.lres_mode = True
		lres_decoded_idx = self.zncc_depth_decoding(zero_norm_c_vec, c_vec_is_zero_norm=True)
		start_idx = (lres_decoded_idx-1)*self.lres_factor
		start_idx[start_idx<=0] = 0
		end_idx = start_idx + 2*self.lres_factor
		end_idx[end_idx >= self.n_maxres] = self.n_maxres 
		# For each pixel we extract the portion of the lookup table where we think the depth is
		zoomed_zero_norm_C = np.zeros(start_idx.shape + (2*self.lres_factor, self.zero_norm_C.shape[-1])).astype(self.zero_norm_C.dtype)
		for i in range(start_idx.shape[0]):
			for j in range(start_idx.shape[1]):
				n_elems = end_idx[i,j] - start_idx[i,j]
				if(n_elems != self.lres_factor*2): print(n_elems)
				zoomed_zero_norm_C[i, j, 0:n_elems] = self.zero_norm_C[...,start_idx[i,j]:end_idx[i,j],:]
		# Get Fine Depth Map
		hres_decoded_depth_idx = self.zncc_depth_decoding(zero_norm_c_vec, input_C=zoomed_zero_norm_C, c_vec_is_zero_norm=True )
		self.lres_mode = False
		return start_idx + hres_decoded_depth_idx

	def get_pretty_C(self, col2row_ratio=1.35):
		if((self.n_maxres // 2) < self.n_codes): col2row_ratio=1
		n_row_per_code = int(np.floor(self.n_maxres / self.n_codes) / col2row_ratio)
		n_rows = n_row_per_code*self.n_codes
		n_cols = self.n_maxres
		pretty_C = np.zeros((n_rows, n_cols))
		for i in range(self.n_codes):
			start_row = i*n_row_per_code
			end_row = start_row + n_row_per_code
			pretty_C[start_row:end_row, :] = self.C[:, i] 
		return pretty_C

	def get_pretty_decoding_C(self, col2row_ratio=1.35):
		if((self.n_maxres // 2) < self.n_codes): col2row_ratio=1
		n_row_per_code = int(np.floor(self.n_maxres / self.n_codes) / col2row_ratio)
		n_rows = n_row_per_code*self.n_codes
		n_cols = self.n_maxres
		pretty_C = np.zeros((n_rows, n_cols))
		for i in range(self.n_codes):
			start_row = i*n_row_per_code
			end_row = start_row + n_row_per_code
			pretty_C[start_row:end_row, :] = self.decoding_C[:, i] 
		return pretty_C

	def get_scheme_id(self):
		coding_id = self.__class__.__name__
		return '{}_ncodes-{}'.format(coding_id,self.C.shape[-1])


class DataCoding(Coding):
	'''
		Class for coding class based on input data
	'''
	def __init__(self, C, n_maxres=None, **kwargs):
		# interpolate or extrapolate C if needed (assume we have oversampled C already)
		if(n_maxres is None): n_maxres = C.shape[0]
		resampled_C = signal.resample(C, n_maxres, axis=0)
		# Set the coding matrix
		self.set_coding_mat(resampled_C)
		super().__init__(**kwargs)

	def set_coding_mat(self, C):
		self.C = C
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)


class GatedCoding(Coding):
	'''
		Gated coding class. Coding is applied like a gated camera or a coarse histogram in SPADs
		In the extreme case that we have a gate for every time bin then the C matrix is an (n_maxres x n_maxres) identity matrix
	'''
	def __init__(self, n_maxres, n_gates=None, **kwargs):
		if(n_gates is None): n_gates = n_maxres
		assert((n_maxres % n_gates) == 0), "Right now GatedCoding required n_maxres to be divisible by n_gates"
		assert((n_maxres >= n_gates)), "n_gates should always be smaller than n_maxres"
		self.n_gates = n_gates
		self.set_coding_mat(n_maxres, n_gates)
		super().__init__(**kwargs)

	def set_coding_mat(self, n_maxres, n_gates):
		self.gate_len = int(n_maxres / n_gates)
		self.C = np.zeros((n_maxres, n_gates))
		for i in range(n_gates):
			start_tbin = i*self.gate_len
			end_tbin = start_tbin + self.gate_len
			self.C[start_tbin:end_tbin, i] = 1.
	
	def encode(self, transient_img):
		'''
		Encode the transient image using the n_codes inside the self.C matrix
		For GatedCoding with many n_gates, encoding through matmul is quite slow, so we assign it differently
		'''
		assert(transient_img.shape[-1] == self.n_maxres), "Input c_vec does not have the correct dimensions"
		c_vals = np.array(transient_img[..., 0::self.gate_len])
		for i in range(self.gate_len-1):
			start_idx = i+1
			c_vals += transient_img[..., start_idx::self.gate_len]
		return c_vals

	def linear_reconstruction(self, c_vals):
		if(self.n_gates == self.n_maxres): return c_vals
		if(self.account_irf):
			print("Warning: Linear Reconstruction in Gated does not account for IRF, so unless the IRF spreads across time bins, this will produce quantized depths")
		x_fullres = np.arange(0, self.n_maxres)
		# Create a circular x axis by concatenating the first element to the end and the last element to the begining
		circular_x_lres = np.arange((0.5*self.gate_len)-0.5-self.gate_len, self.n_maxres + self.gate_len, self.gate_len)
		circular_c_vals = np.concatenate((c_vals[..., -1][...,np.newaxis], c_vals, c_vals[..., 0][...,np.newaxis]), axis=-1)
		f = interpolate.interp1d(circular_x_lres, circular_c_vals, axis=-1, kind='linear')
		return f(x_fullres)
	
	def matchfilt_reconstruction(self, c_vals):
		template = self.h_irf
		self.verify_input_c_vec(c_vals)
		zn_template = zero_norm_t(template, axis=-1)
		zn_c_vals = zero_norm_t(c_vals, axis=-1)
		shifts = signalproc_ops.circular_matched_filter(zn_c_vals, zn_template)
		# vectorize tensors
		(c_vals, c_vals_shape) = np_utils.vectorize_tensor(c_vals, axis=-1)
		shifts = shifts.reshape((c_vals.shape[0],))
		h_rec = np.zeros(c_vals.shape, dtype=template.dtype)
		for i in range(shifts.size): h_rec[i,:] = np.roll(template, shift=shifts[i], axis=-1)
		c_vals = c_vals.reshape(c_vals_shape)
		return h_rec.reshape(c_vals_shape)

class IdentityCoding(GatedCoding):
	'''
		Identity coding class. GatedCoding in the extreme case where n_maxres == n_gates
	'''
	def __init__(self, n_maxres, **kwargs):
		super().__init__(n_maxres=n_maxres, **kwargs)
class TimestampCoding(IdentityCoding):
	'''
		Timestamp coding class. 
		Same as IdentityCoding but when simulating we only record up to n_timestamps.
	'''
	def __init__(self, n_maxres, n_timestamps, **kwargs):
		super().__init__(n_maxres=n_maxres, **kwargs)
		self.n_timestamps=n_timestamps

	def encode(self, transient_img):
		'''
			Take full histogram, get all timestamps, randomly select n_timestamps, construct new timestamp limited histogram.
		'''
		# get timestamps from histogram
		(timestamps, n_timestamps_per_elem) = hist2timestamps(transient_img, max_n_timestamps=self.n_timestamps)
		# return histogram with limited number of timestamps
		return timestamps2hist(timestamps, n_timestamps_per_elem, self.n_maxres)

class GrayCoding(Coding):
	'''
		Gray coding class. 
	'''
	def __init__(self, n_maxres, n_bits, **kwargs):
		self.max_n_bits = int(np.floor(np.log2(n_maxres)))
		self.n_maxres = n_maxres
		self.n_bits = np.min((n_bits, self.max_n_bits)) 
		if(n_bits > self.max_n_bits):
			print("not using n_bits={}, using n_bits={} which is the max_n_bits for {} bins".format(n_bits, self.max_n_bits, n_maxres))
		self.set_coding_mat(self.n_maxres, self.n_bits)
		super().__init__(**kwargs)

	def np_gray_code(self, n_bits):
		return signalproc_ops.generate_gray_code(n_bits)

	def set_coding_mat(self, n_maxres, n_bits):
		self.C = np.zeros((n_maxres, n_bits))
		self.gray_codes = self.np_gray_code(n_bits)
		self.min_gray_code_length = self.gray_codes.shape[0]
		# assert(n_maxres >= self.min_gray_code_length), "n_maxres is not large enough to encode the gray code"
		if((n_maxres % self.min_gray_code_length) != 0):
			print("WARNING: Gray codes where the n_maxres is not a multiple of the gray code length, may have some small ambiguous regions")
		self.x_fullres = np.arange(0, n_maxres) * (1. / n_maxres)
		self.x_lowres = np.arange(0, self.min_gray_code_length) * (1. / self.min_gray_code_length)
		ext_x_lowres = np.arange(-1, self.min_gray_code_length+1) * (1. / self.min_gray_code_length)
		ext_gray_codes = np.concatenate((self.gray_codes[-1,:][np.newaxis, :], self.gray_codes, self.gray_codes[0,:][np.newaxis,:]), axis=0)
		f = interpolate.interp1d(ext_x_lowres, ext_gray_codes, axis=0, kind='linear')
		self.C = f(self.x_fullres)
		self.C = (self.C*2)-1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

class DecomposedGrayCoding(Coding):
	'''
		Decomposed Gray coding class. 
	'''
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.max_n_bits = int(np.floor(np.log2(n_maxres)))
		self.n_maxres = n_maxres
		self.n_bits = np.min((n_codes, self.max_n_bits))
		self.n_codes = n_codes
		if(self.n_bits > self.max_n_bits):
			print("not using n_bits={}, using n_bits={} which is the max_n_bits for {} bins".format(self.n_bits, self.max_n_bits, n_maxres))
		self.set_coding_mat(self.n_maxres, self.n_bits, self.n_codes)
		super().__init__(**kwargs)

	def np_gray_code(self, n_bits):
		return signalproc_ops.generate_gray_code(n_bits)

	def decompose_gray_code(self, gray_code, curr_bit):
		n_codes_in_gray_code = 2**curr_bit
		decomposed_code = np.zeros((gray_code.shape[0], n_codes_in_gray_code))
		decomposed_code_len = int(np.floor(gray_code.shape[0] / n_codes_in_gray_code)) 
		for i in range(n_codes_in_gray_code):
			start_idx = i*decomposed_code_len
			end_idx = start_idx+decomposed_code_len
			decomposed_code[start_idx:end_idx,i] = gray_code[start_idx:end_idx]
		return decomposed_code
	
	def binary_codes(self, n_maxres, level):
		n_codes = 2**(level-1)
		bin_codes = np.zeros((n_maxres, n_codes))
		code_len = int(np.round(n_maxres / n_codes))
		half_code_len = code_len // 2
		for i in range(n_codes):
			start_idx = i*code_len
			end_idx1 = start_idx + half_code_len
			end_idx2 = np.min([start_idx + code_len, n_maxres])
			bin_codes[start_idx:end_idx1, i] = 1
			bin_codes[end_idx1:end_idx2, i] = -1
		return bin_codes

	def set_coding_mat(self, n_maxres, n_bits, n_codes):
		self.C = np.zeros((n_maxres, n_codes))
		self.gray_codes = self.np_gray_code(n_bits)
		self.min_gray_code_length = self.gray_codes.shape[0]
		# assert(n_maxres >= self.min_gray_code_length), "n_maxres is not large enough to encode the gray code"
		if((n_maxres % self.min_gray_code_length) != 0):
			print("WARNING: Gray codes where the n_maxres is not a multiple of the gray code length, may have some small ambiguous regions")
		self.x_fullres = np.arange(0, n_maxres) * (1. / n_maxres)
		self.x_lowres = np.arange(0, self.min_gray_code_length) * (1. / self.min_gray_code_length)
		ext_x_lowres = np.arange(-1, self.min_gray_code_length+1) * (1. / self.min_gray_code_length)
		ext_gray_codes = np.concatenate((self.gray_codes[-1,:][np.newaxis, :], self.gray_codes, self.gray_codes[0,:][np.newaxis,:]), axis=0)
		f = interpolate.interp1d(ext_x_lowres, ext_gray_codes, axis=0, kind='linear')
		## Set first n_bits codes to traditional gray codes
		self.C[:, 0:self.n_bits] = f(self.x_fullres)
		self.C = (self.C*2)-1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)
		## Haar codes
		self.n_decomposed_codes = self.n_codes - self.n_bits
		if(self.n_decomposed_codes >= 1):
			curr_level = 2
			curr_start_code = self.n_bits
			n_remaining_codes = self.n_decomposed_codes
			while(n_remaining_codes > 0):
				curr_bin_codes = self.binary_codes(self.n_maxres, curr_level)
				if(curr_bin_codes.shape[-1] > n_remaining_codes):
					curr_end_code = curr_start_code+n_remaining_codes
					self.C[:, curr_start_code:curr_end_code] = curr_bin_codes[:,0:n_remaining_codes]
				else:
					curr_end_code = curr_start_code+curr_bin_codes.shape[-1]
					self.C[:, curr_start_code:curr_end_code] = curr_bin_codes
				n_remaining_codes -= curr_bin_codes.shape[-1]
				curr_start_code = curr_end_code
				curr_level += 1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)
		# ## Decompose gray codes
		# self.n_decomposed_codes = self.n_codes - self.n_bits
		# if(self.n_decomposed_codes >= 1):
		# 	curr_bit=3
		# 	curr_start_code=self.n_bits
		# 	n_remaining_codes = self.n_decomposed_codes 
		# 	while(n_remaining_codes > 0):
		# 		curr_gray_code = self.C[:, curr_bit-1]
		# 		decomposed_gray_codes = self.decompose_gray_code(curr_gray_code, curr_bit-2)
		# 		if(decomposed_gray_codes.shape[-1] > n_remaining_codes):
		# 			curr_end_code = curr_start_code+n_remaining_codes
		# 			self.C[:, curr_start_code:curr_end_code] = decomposed_gray_codes[:,0:n_remaining_codes]
		# 		else:
		# 			curr_end_code = curr_start_code+decomposed_gray_codes.shape[-1]
		# 			self.C[:, curr_start_code:curr_end_code] = decomposed_gray_codes
		# 		n_remaining_codes -= decomposed_gray_codes.shape[-1]
		# 		curr_start_code = curr_end_code
		# 		curr_bit += 1
		# self.C = self.C - self.C.mean(axis=-2, keepdims=True)

class WalshHadamardCoding(Coding):
	'''
		WalshHadamard coding class. 
		This is the one used in the paper. The binayr version has many ambiguities at low K. 
		We avoid ambiguities here by interpolating 
	'''
	def __init__(self, n_maxres, n_codes, **kwargs):
		# check n_
		assert(np.ceil(np.log2(n_codes)) == np.floor(np.log2(n_codes))), "n_codes needs to be power of 2"
		# self.n_hadamard_codes = 2**np.ceil(np.log2(n_codes+1))
		self.n_codes = n_codes
		self.n_maxres = n_maxres
		self.set_coding_mat(self.n_maxres, self.n_codes)
		super().__init__(**kwargs)

	def wh_mat(self, n_codes):
		assert(n_codes >= 1), "invalid n_codes"
		from scipy.linalg import hadamard
		return hadamard(n_codes)

	def set_coding_mat(self, n_maxres, n_codes):
		self.C = np.zeros((n_maxres, n_codes))
		self.hadamard_mat = self.wh_mat(n_codes)
		self.min_hadamard_code_length = self.hadamard_mat.shape[0]
		if((n_maxres % self.min_hadamard_code_length) != 0):
			print("WARNING: Hadamard codes where the n_maxres is not a multiple of the hadamard code length, may have some small ambiguous regions")
		self.x_fullres = np.arange(0, n_maxres) * (1. / n_maxres)
		self.x_lowres = np.arange(0, self.min_hadamard_code_length) * (1. / self.min_hadamard_code_length)
		ext_x_lowres = np.arange(-1, self.min_hadamard_code_length+1) * (1. / self.min_hadamard_code_length)
		ext_hadamard_mat = np.concatenate((self.hadamard_mat[-1,:][np.newaxis, :], self.hadamard_mat, self.hadamard_mat[0,:][np.newaxis,:]), axis=0)
		f = interpolate.interp1d(ext_x_lowres, ext_hadamard_mat, axis=0, kind='linear')
		self.C = f(self.x_fullres)
		self.C = (self.C*2)-1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

class WalshHadamardBinaryCoding(Coding):
	'''
		WalshHadamardBinary coding class. 
	'''
	def __init__(self, n_maxres, n_codes, **kwargs):
		# check n_
		self.n_codes = n_codes
		self.n_maxres = n_maxres
		self.n_bits = int(np.ceil(np.log2(self.n_maxres)))
		self.set_coding_mat(self.n_maxres, self.n_codes)
		super().__init__(**kwargs)

	def sequency_wh_mat(self, n_bits):
		assert(n_bits >= 1), "invalid n_codes"
		from scipy.linalg import hadamard
		hmat = hadamard(2**n_bits)
		gc = signalproc_ops.generate_gray_code(n_bits)
		two_pwrs = 2**np.arange(0, n_bits)
		gc_order_indeces = gc.dot(two_pwrs).astype(int)
		return hmat[gc_order_indeces, :]

	def set_coding_mat(self, n_maxres, n_codes):
		hres_hadamard_mat = self.sequency_wh_mat(self.n_bits)
		n_hres = hres_hadamard_mat.shape[-1]
		x_lres = np.arange(0, n_maxres) * (1. / n_maxres)
		ext_x_hres = np.arange(-1, n_hres+1) * (1. / n_hres)
		ext_hadamard_mat = np.concatenate((hres_hadamard_mat[-1,:][np.newaxis, :], hres_hadamard_mat, hres_hadamard_mat[0,:][np.newaxis,:]), axis=0)
		f = interpolate.interp1d(ext_x_hres, ext_hadamard_mat, axis=0, kind='linear')
		self.hadamard_mat = f(x_lres) 
		self.C = self.hadamard_mat[:, 1:n_codes+1]
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

class ReflectedGrayCoding(GrayCoding):
	def __init__(self, n_maxres, n_bits, **kwargs):
		super().__init__( n_maxres, n_bits, **kwargs )
	
	def set_coding_mat(self, n_maxres, n_bits):
		super().set_coding_mat(n_maxres, n_bits)
		reflected_C = -1*self.C
		self.C = np.concatenate((self.C, reflected_C), axis=-1)

class ComplementaryGrayCoding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.set_coding_mat(n_maxres, n_codes)
		super().__init__( **kwargs )
	
	def set_coding_mat(self, n_maxres, n_codes):
		self.gray_coding_obj = GrayCoding(n_maxres, n_codes)
		self.gray_C = np.array(self.gray_coding_obj.C)
		n_gray_codes = self.gray_C.shape[-1] 
		## If we have codes left add them
		if(n_codes > n_gray_codes):
			## Get orthogonal binary codes for all codes
			self.orth_gray_C = np.zeros_like(self.gray_C)
			for i in range(n_gray_codes):
				self.orth_gray_C[:,i] = signalproc_ops.get_orthogonal_binary_code(self.gray_C[:,i])
			## Drop the first two because the already had orthogonal codes
			self.orth_gray_C = self.orth_gray_C[:,2:]
			## Generate C using both matrices
			n_remaining_codes = n_codes - n_gray_codes
			if(n_remaining_codes <= self.orth_gray_C.shape[-1]):
				self.C = np.concatenate((self.gray_C, self.orth_gray_C[:,0:n_remaining_codes]), axis=-1)
			else:
				print("Max number of compl. gray codes is = {}. Truncating to this number".format((2*n_gray_codes) - 2))
				self.C = np.concatenate((self.gray_C, self.orth_gray_C), axis=-1)
		else:
			self.C = self.gray_C
		return self.C

class PSeriesGrayCoding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.set_coding_mat(n_maxres, n_codes)
		super().__init__( **kwargs )
	
	def set_coding_mat(self, n_maxres, n_codes):
		self.gray_coding_obj = ComplementaryGrayCoding(n_maxres, n_codes)
		self.gray_C = np.array(self.gray_coding_obj.C)
		n_gray_codes = self.gray_C.shape[-1] 
		## If we have codes left start adding binary codes
		if(n_codes > n_gray_codes):
			all_freqs = np.arange(1, n_maxres // 2)
			dominant_freqs = signalproc_ops.get_dominant_freqs(self.gray_C, axis=0)
			n_remaining_codes = n_codes - n_gray_codes
			self.binary_C = np.zeros((n_maxres, n_remaining_codes))
			i = 0
			t = np.arange(0, n_maxres) / n_maxres
			while(n_remaining_codes > 0):
				curr_freq = all_freqs[i]
				# skip if we have already used it
				if(curr_freq in dominant_freqs): 
					i+=1
					continue
				# add binary code
				binary_code = scipy.signal.square(2*np.pi*curr_freq*t)
				self.binary_C[:,n_remaining_codes-1] = binary_code
				n_remaining_codes-=1
				# add the orthogonal binary code if we can
				if(n_remaining_codes == 0): break
				orth_binary_code = signalproc_ops.get_orthogonal_binary_code(binary_code)
				self.binary_C[:,n_remaining_codes-1] = orth_binary_code
				n_remaining_codes-=1
				i+=1
			self.binary_C = np.fliplr(self.binary_C)
			self.C = np.concatenate((self.gray_C, self.binary_C), axis=-1)
		else:
			self.C = self.gray_C
		return self.C

class GrayTruncatedFourierCoding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		# Create Gray coding obj
		self.gray_coding_obj = GrayCoding(n_maxres, n_codes)
		self.n_gray_codes = self.gray_coding_obj.n_codes
		# Create Fourier Coding obj with remaining codes we want to use
		self.n_fourier_codes = n_codes - self.n_gray_codes 
		self.n_freqs = int(np.ceil(float(self.n_fourier_codes)/2.))
		if(self.n_freqs >= 1): 
			self.truncfourier_coding_obj = TruncatedFourierCoding(n_maxres, n_freqs=self.n_freqs, include_zeroth_harmonic=False)
		else: self.truncfourier_coding_obj = None
		# Set coding mat by concat the matrices
		self.set_coding_mat()
		super().__init__(**kwargs)
	
	def set_coding_mat(self):
		gray_C = self.gray_coding_obj.C
		if(self.n_fourier_codes >= 1):
			truncfourier_C = self.truncfourier_coding_obj.C[:, 0:self.n_fourier_codes]
			self.C = np.concatenate((gray_C, truncfourier_C), axis=-1)
		else:
			self.C = gray_C

class GrayTruncatedFourierV2Coding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		# Start with the first 16 freqs
		self.n_fourier_codes = np.min([16, n_codes]) 
		self.n_freqs = int(np.ceil(float(self.n_fourier_codes)/2.))
		self.truncfourier_coding_obj = TruncatedFourierCoding(n_maxres, n_freqs=self.n_freqs, include_zeroth_harmonic=False)
		# if there is codes left create gray coding
		self.n_codes_avail = n_codes - self.n_fourier_codes
		self.n_gray_codes = np.min([self.n_codes_avail, 8])
		if(self.n_gray_codes >= 1 ):
			self.gray_coding_obj = GrayCoding(n_maxres, self.n_gray_codes)
		else: self.gray_coding_obj = None
		# assign rest of freqs randomly
		self.n_codes_avail = n_codes - self.n_fourier_codes - self.n_gray_codes
		self.n_rand_freqs = int(np.ceil(float(self.n_codes_avail)/2.))
		if(self.n_rand_freqs >= 1 ):
			self.randfourier_coding_obj = RandomFourierCoding(n_maxres, n_rand_freqs=self.n_rand_freqs, include_first_harmonic=False)
		else: self.randfourier_coding_obj = None
		# Set coding mat by concat the matrices
		self.set_coding_mat()
		super().__init__(**kwargs)
	
	def set_coding_mat(self):
		truncfourier_C = self.truncfourier_coding_obj.C
		if((self.n_gray_codes >= 1) and (self.n_rand_freqs >= 1)):
			gray_C = self.gray_coding_obj.C
			randfourier_C = self.randfourier_coding_obj.C
			self.C = np.concatenate((truncfourier_C, gray_C, randfourier_C), axis=-1)
		elif(self.n_gray_codes >= 1):
			gray_C = self.gray_coding_obj.C
			self.C = np.concatenate((truncfourier_C, gray_C), axis=-1)
		else:
			self.C = truncfourier_C

class GrayEquispaced3FourierCoding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		# Create Gray coding obj
		self.gray_coding_obj = GrayCoding(n_maxres, n_codes)
		self.n_gray_codes = self.gray_coding_obj.n_codes
		# Create Fourier Coding obj with remaining codes we want to use
		self.n_fourier_codes = n_codes - self.n_gray_codes 
		self.n_freqs = int(np.ceil(float(self.n_fourier_codes)/2.))
		self.freq_spacing = 3
		self.freqs = np.arange(1, self.n_freqs+1)*self.freq_spacing
		if(self.n_freqs >= 1): 
			assert(np.max(self.freqs) <= (n_maxres//2)), 'Exceeded max possible freq'
			self.fourier_coding_obj = FourierCoding(n_maxres, freq_idx=self.freqs)
		else: self.fourier_coding_obj = None
		# Set coding mat by concat the matrices
		self.set_coding_mat()
		super().__init__(**kwargs)
	
	def set_coding_mat(self):
		gray_C = self.gray_coding_obj.C
		if(self.n_fourier_codes >= 1):
			fourier_C = self.fourier_coding_obj.C[:, 0:self.n_fourier_codes]
			self.C = np.concatenate((gray_C, fourier_C), axis=-1)
		else:
			self.C = gray_C

class HaarCoding(Coding):
	'''
		Haar coding class. 
	'''
	def __init__(self, n_maxres, n_lvls, **kwargs):
		self.n_lvls = n_lvls
		self.set_coding_mat(n_maxres, n_lvls)
		super().__init__(**kwargs)

	def set_coding_mat(self, n_maxres, n_lvls):
		self.C = signalproc_ops.haar_matrix(n_maxres, n_lvls)

class RandomCoding(Coding):
	'''
		Random coding class. 
	'''
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.set_coding_mat(n_maxres, n_codes)
		super().__init__(**kwargs)
	def set_coding_mat(self, n_maxres, n_codes):
		self.C = np.random.rand(n_maxres,n_codes)
		self.C = (signalproc_ops.standardize_signal(self.C, axis=0)*2) - 1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

class HamiltonianCoding(Coding):
	'''
		Hamiltonian coding class. 
	'''
	def __init__(self, n_maxres, k, **kwargs):
		self.k = k
		self.set_coding_mat(n_maxres, k)
		super().__init__(**kwargs)
	def set_coding_mat(self, n_maxres, k):
		if(k==3): (modfs, demodfs) = GetHamK3(n_maxres)
		elif(k==4): (modfs, demodfs) = GetHamK4(n_maxres)
		elif(k==5): (modfs, demodfs) = GetHamK5(n_maxres)
		else: assert(False), "Not implemented Hamiltonian for K>5"
		self.C = signalproc_ops.circular_corr(modfs, demodfs, axis=0)
		self.C = (signalproc_ops.standardize_signal(self.C, axis=0)*2) - 1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

def GetHamK3(N=1000):
	"""GetHamK3: Get modulation and demodulation functions for the coding scheme
		HamK3 - Sq16Sq50.	
	Args:
		N (int): N
	Returns:
		modFs: NxK matrix
		demodFs: NxK matrix
	"""
	#### Set some parameters
	K = 3
	maxInstantPowerFactor = 6.
	dt = float(1.0) / float(N)
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	#### Prepare modulation functions
	modDuty = 1./maxInstantPowerFactor
	for i in range(0,K):
		modFs[0:math.floor(modDuty*N),i] = maxInstantPowerFactor*1.0
	#### Prepare demodulation functions
	## Make shape of function
	demodDuty = 1./2.
	for i in range(0,K):
		demodFs[0:math.floor(demodDuty*N),i] = 1.
	## Apply necessary phase shift
	shifts = [0, (1./3.)*N, (2./3.)*N]
	demodFs = ApplyKPhaseShifts(demodFs,shifts)
	return (modFs, demodFs)


def GetHamK4(N=1000):
	"""GetHamK4: Get modulation and demodulation functions for the coding scheme HamK4	
	Args:
		N (int): N
	Returns:
		modFs: NxK matrix
		demodFs: NxK matrix
	"""
	#### Set some parameters
	K = 4
	maxInstantPowerFactor=12.
	dt = float(1.0) / float(N)
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	#### Prepare modulation functions
	modDuty = 1./maxInstantPowerFactor
	for i in range(0,K):
		modFs[0:math.floor(modDuty*N),i] = maxInstantPowerFactor*1.0
	#### Prepare demodulation functions
	## Make shape of function
	demodDuty1 = np.array([6./12.,6./12.])
	shift1 = 5./12.
	demodDuty2 = np.array([6./12.,6./12.])
	shift2 = 2./12.
	demodDuty3 = np.array([3./12.,4./12.,3./12.,2./12.])
	shift3 = 0./12.
	demodDuty4 = np.array([2./12.,3./12,4./12.,3./12.])
	shift4 = 4./12.
	shifts = [shift1*N, shift2*N, shift3*N, shift4*N]
	demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4]
	for i in range(0,K):
		demodDuty = demodDutys[i]
		startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
		endIndeces = startIndeces + np.floor(demodDuty*N) - 1
		for j in range(len(demodDuty)):
			if((j%2) == 0):
				demodFs[int(startIndeces[j]):int(endIndeces[j]),i] = 1.
	## Apply necessary phase shift
	demodFs = ApplyKPhaseShifts(demodFs,shifts)

	return (modFs, demodFs)


def GetHamK5(N=1000):
	"""GetHamK5: Get modulation and demodulation functions for the coding scheme HamK5.	
	Args:
		N (int): N
	Returns:
		modFs: NxK matrix
		demodFs: NxK matrix
	"""
	#### Set some parameters
	K = 5
	maxInstantPowerFactor=30.
	dt = float(1.0) / float(N)
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	#### Prepare modulation functions
	modDuty = 1./maxInstantPowerFactor
	for i in range(0,K):
		modFs[0:math.floor(modDuty*N),i] = maxInstantPowerFactor*1.0
	#### Prepare demodulation functions
	## Make shape of function
	demodDuty1 = np.array([15./30.,15./30.])
	shift1 = 15./30.
	demodDuty2 = np.array([15./30.,15./30.])
	shift2 = 7./30.
	demodDuty3 = np.array([8./30.,8./30.,7./30.,7./30.])
	shift3 = 3./30.
	demodDuty4 = np.array([4./30.,4./30.,4./30.,4./30.,3./30.,4./30.,4./30.,3./30.])
	shift4 = 1./30.
	demodDuty5 = np.array([2./30.,2./30.,2./30.,2./30.,2./30.,2./30.,2./30.,
							3./30.,2./30.,2./30.,2./30.,2./30.,3./30.,2./30])
	shift5 = 4./30.
	shifts = [shift1*N, shift2*N, shift3*N, shift4*N, shift5*N]
	demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4, demodDuty5]
	for i in range(0,K):
		demodDuty = demodDutys[i]
		startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
		endIndeces = startIndeces + np.floor(demodDuty*N) - 1
		for j in range(len(demodDuty)):
			if((j%2) == 0):
				demodFs[int(startIndeces[j]):int(endIndeces[j]),i] = 1.

	## Apply necessary phase shift
	demodFs = ApplyKPhaseShifts(demodFs,shifts)
	return (modFs, demodFs)

def ApplyKPhaseShifts(x, shifts):
	"""ApplyPhaseShifts: Apply phase shift to each vector in x. 
	
	Args:
		x (np.array): NxK matrix
		shifts (np.array): Array of dimension K.
	
	Returns:
		np.array: Return matrix x where each column has been phase shifted according to shifts. 
	"""
	K = 0
	if(type(shifts) == np.ndarray): K = shifts.size
	elif(type(shifts) == list): K = len(shifts) 
	else: K = 1
	for i in range(0,K):
		x[:,i] = np.roll(x[:,i], int(round(shifts[i])))
	return x

class KTapTriangleCoding(Coding):
	'''
		
	'''
	def __init__(self, n_maxres, freq_idx=[0, 1], k=3, **kwargs):
		self.k=k
		self.set_coding_mat(n_maxres, freq_idx)
		super().__init__(**kwargs)

	def init_coding_mat(self, n_maxres, freq_idx):
		'''
		'''
		# Init some params
		self.n_maxres = n_maxres
		self.n_maxfreqs = self.n_maxres // 2
		self.freq_idx = to_nparray(freq_idx)
		self.n_freqs = self.freq_idx.size
		self.n_codes = self.k*self.n_freqs
		# Check input args
		assert(self.freq_idx.ndim == 1), "Number of dimensions for freq_idx should be 1"
		assert(self.n_freqs <= (self.n_maxres // 2)), "Number of frequencies cannot exceed the number of points at the max resolution"
		assert(np.max(self.freq_idx) <= (self.n_maxres // 2)), "No input frequency should be larger"
		# Initialize and populate the matrix with zero mean sinusoids
		self.C = np.zeros((self.n_maxres, self.n_codes))

	def set_coding_mat(self, n_maxres, freq_idx):
		'''
		Initialize all frequencies
		'''
		self.init_coding_mat(n_maxres, freq_idx)
		domain = np.arange(0, self.n_maxres)*(TWOPI / self.n_maxres)
		for i in range(self.n_freqs):
			start_idx = i*self.k
			for j in range(self.k):
				self.C[:, start_idx+j] = signal.sawtooth(domain*self.freq_idx[i] - (j*TWOPI / self.k), width=0.5)
		return self.C

	def are_freq_idx_consecutive(self):
		diff = (self.freq_idx[1:] - self.freq_idx[0:-1])
		return np.sum(diff-1) == 0

class TruncatedKTapTriangleCoding(KTapTriangleCoding):
	'''
	'''
	def __init__(self, n_maxres, n_freqs=1, include_zeroth_harmonic=True, k=3, **kwargs):
		freq_idx = np.arange(0, n_freqs+1)
		# Remove zeroth harmonic if needed.
		if(not include_zeroth_harmonic): freq_idx = freq_idx[1:]
		self.include_zeroth_harmonic = include_zeroth_harmonic
		super().__init__(n_maxres, freq_idx=freq_idx, k=k, **kwargs)

class GatedFourierCoding(Coding):
	'''
		class for GatedFourier coding
	'''
	def __init__(self, n_maxres, freq_idx=[0, 1], n_gates=None, **kwargs):
		self.gated_coding = GatedCoding(n_maxres, n_gates)
		self.gate_len = self.gated_coding.gate_len
		self.n_gates = self.gated_coding.n_gates
		self.fourier_coding = FourierCoding(self.gate_len, freq_idx)
		self.n_freqs = self.fourier_coding.n_freqs
		self.set_coding_mat(n_maxres)
		super().__init__(**kwargs)

	def set_coding_mat(self, n_maxres):
		'''
		Initialize all frequencies
		'''
		self.C = np.zeros((n_maxres, self.gated_coding.n_codes*self.fourier_coding.n_codes), dtype=self.fourier_coding.C.dtype)
		for i in range(self.n_gates):
			start_gate_idx = i*self.gate_len
			end_gate_idx = start_gate_idx + self.gate_len
			start_code_idx = i*self.fourier_coding.n_codes
			end_code_idx = start_code_idx + self.fourier_coding.n_codes
			self.C[start_gate_idx:end_gate_idx, start_code_idx:end_code_idx] = self.fourier_coding.C
		return self.C

class FourierCoding(Coding):
	'''
		class for Fourier coding
	'''
	def __init__(self, n_maxres, freq_idx=[0, 1], n_codes=None, **kwargs):
		self.n_codes = n_codes
		self.set_coding_mat(n_maxres, freq_idx)
		super().__init__(**kwargs)
		self.lres_n_freqs = self.lres_n // 2

	def get_n_maxfreqs(self):
		if(self.lres_mode): return self.lres_n_freqs
		else: return self.n_maxfreqs

	def init_coding_mat(self, n_maxres, freq_idx):
		'''
			Shared initialization for all FourierCoding objects
				* k=2 means that there is 2 sinusoids per frequency
				* some derived classes may use k>2
		'''
		# Init some params
		self.n_maxres = n_maxres
		self.n_maxfreqs = self.n_maxres // 2
		self.freq_idx = to_nparray(freq_idx)
		self.n_freqs = self.freq_idx.size
		self.max_n_sinusoid_codes = self.k*self.n_freqs
		if(self.n_codes is None): self.n_sinusoid_codes = self.max_n_sinusoid_codes
		else:  
			if(self.n_codes > self.max_n_sinusoid_codes): warnings.warn("self.n_codes is larger than max_n_sinusoid_codes, truncating number of codes to max_n_sinusoid_codes")
			self.n_sinusoid_codes = np.min([self.max_n_sinusoid_codes, self.n_codes])
		# Check input args
		assert(self.freq_idx.ndim == 1), "Number of dimensions for freq_idx should be 1"
		assert(self.n_freqs <= (self.n_maxres // 2)), "Number of frequencies cannot exceed the number of points at the max resolution"
		# Initialize and populate the matrix with zero mean sinusoids
		self.C = np.zeros((self.n_maxres, self.n_sinusoid_codes))

	def set_coding_mat(self, n_maxres, freq_idx):
		'''
		Initialize all frequencies
		'''
		self.k = 2
		self.init_coding_mat(n_maxres, freq_idx)
		domain = np.arange(0, self.n_maxres)*(TWOPI / self.n_maxres)
		fourier_mat = signalproc_ops.get_fourier_mat(n=self.n_maxres, freq_idx=self.freq_idx)
		for i in range(self.n_sinusoid_codes):
			if((i % 2) == 0):
				self.C[:, i] = fourier_mat[:, i // 2].real
			else:
				self.C[:, i] = fourier_mat[:, i // 2].imag
		# self.C[:, 0::2] = fourier_mat.real
		# self.C[:, 1::2] = fourier_mat.imag
		return self.C

	def are_freq_idx_consecutive(self):
		diff = (self.freq_idx[1:] - self.freq_idx[0:-1])
		return np.sum(diff-1) == 0

	def has_kth_harmonic(self, k): return k in self.freq_idx
	def has_zeroth_harmonic(self): return self.has_kth_harmonic(k=0)
	def has_first_harmonic(self): return self.has_kth_harmonic(k=1)
	def remove_zeroth_harmonic(self, cmpx_c_vec): return cmpx_c_vec[..., self.freq_idx != 0]

	def ifft_reconstruction(self, c_vec):
		'''
		Use ZNCC to approximately reconstruct the signal encoded by c_vec
		'''
		self.verify_input_c_vec(c_vec)
		fft_transient = self.construct_fft_transient(c_vec)
		# Finally return the IFT
		return np.fft.irfft(fft_transient, axis=-1, n=self.get_n_maxres())
		
	def circmean_reconstruction(self, c_vec):
		'''
			Take phase of the first harmonic and output the depth for that phase
		'''
		n_bins = self.get_n_maxres()
		assert(self.has_first_harmonic()), "First harmonic is required for cirmean calculation"
		circ_mean_phase = self.decode_phase(c_vec, query_freq=1)
		circ_mean_index = np_utils.domain2index(circ_mean_phase, TWOPI, n_bins)
		reconstruction = np.zeros(c_vec.shape[0:-1] + (n_bins,))
		np.put_along_axis(reconstruction, indices=circ_mean_index[..., np.newaxis], values=1, axis=-1)
		return reconstruction

	# def GS1991_reconstruction(self, c_vec):
	# 	'''
	# 		Implementation of Gushov & Soldkin (1991) multi-frequency phase unwrapping algorithm
	# 	'''
	# 	cmpx_c_vec = self.construct_phasor(c_vec).conjugate()
	# 	cmpx_c_vec = self.remove_zeroth_harmonic(cmpx_c_vec)
	# 	nonzero_freqs = self.freq_idx[self.freq_idx != 0]
	# 	m = np.prod(nonzero_freqs)
	# 	M_i = m / nonzero_freqs
	# 	phase = linearize_phase(np.angle(cmpx_c_vec))

	def mese_reconstruction(self, c_vec):
		''' Maximum entropy spectral estimate method
		Calculates the impulse response, that given the fourier coefficients, minimized the burg entropy.
		See paper: http://momentsingraphics.de/Media/SiggraphAsia2015/FastTransientImaging.pdf
		'''
		assert(self.has_zeroth_harmonic()), "MESE Reconstruction requires zeroth harmonic"
		assert(self.are_freq_idx_consecutive()), "MESE Reconstruction requires frequency indeces to be consecutive"
		c_vec = c_vec.squeeze()
		n_bins = self.get_n_maxres()
		# Use conjugate. If we don't conjugate, the reconstructed signal will be flipped. 
		cmpx_c_vec = self.construct_phasor(c_vec).conjugate()
		# Vectorize the tensor
		(cmpx_c_vec, cmpx_c_vec_orig_shape) = np_utils.vectorize_tensor(cmpx_c_vec, axis=-1)
		# Scale zeroth harmonic a bit. It Improves numerical stability. Makes solution slightly less sparse
		# TODO: Go over Trig Moments Paper and review why this helps
		# ambient_estimate = np.abs(cmpx_c_vec[...,0] - 0.5*np.abs(cmpx_c_vec[..., 1]))
		# cmpx_c_vec[..., 0] -= ambient_estimate
		cmpx_c_vec[..., 0] *= 1.1
		# Set constants
		e0 = np.eye(cmpx_c_vec.shape[-1], 1)
		S = signalproc_ops.get_fourier_mat(n=n_bins, freq_idx=self.freq_idx).transpose()
		# Construct toeplitz matrix. We have our own implementation where the toeplitz matrix is constructed the last dimension
		# So if cmpx_c_vec is NxMxK we will output B as a NxMxKxK, where the last 2 dims are the toeplitz for each MxN element
		B = signalproc_ops.broadcast_toeplitz(cmpx_c_vec)
		invertible_B_mask = (np.linalg.matrix_rank(B, hermitian=True) == B.shape[-1])
		reconstruction = np.zeros(cmpx_c_vec.shape[0:-1] + (n_bins,))
		# Try to solve
		try:
			# Only invert for the pixels for which B is invertible
			Binv = np.linalg.inv(B[invertible_B_mask, :])
			e0t_dot_Binv = np.matmul(e0.transpose(), Binv)
			numerator = np.matmul( e0t_dot_Binv, e0 )
			denominator = TWOPI*np.square(np.abs( np.matmul(e0t_dot_Binv, S) ))
			# Only solve for the impulse response at valid pixels
			reconstruction[invertible_B_mask,:] = (np.real(numerator) / denominator).squeeze()
		except np.linalg.linalg.LinAlgError as exception_error:
			print(exception_error.args)
			print("WARNING! We should never arrive here because we only invert valid Bmat matrices!")
		# If the above fails then we return a matrix with all zeros
		return reconstruction.reshape(cmpx_c_vec_orig_shape[0:-1] + (n_bins,))

	def pizarenko_reconstruction(self, c_vec):
		''' Pizarenko Estimate
		K-sparse reconstruction .
		See paper: http://momentsingraphics.de/Media/SiggraphAsia2015/FastTransientImaging.pdf
		'''
		n_bins = self.get_n_maxres()
		c_vec = c_vec.squeeze()
		assert(self.are_freq_idx_consecutive()), "MESE Reconstruction requires frequency indeces to be consecutive"
		# Use conjugate. If we don't conjugate, the reconstructed signal will be flipped. 
		cmpx_c_vec = self.construct_phasor(c_vec).conjugate()
		# Vectorize the tensor
		(cmpx_c_vec, cmpx_c_vec_orig_shape) = np_utils.vectorize_tensor(cmpx_c_vec, axis=-1)
		# Set zeroth harmonic to 0
		if(self.has_zeroth_harmonic()):
			cmpx_c_vec[..., 0] = 0.
		else:
			cmpx_c_vec = np.concatenate(np.zeros((cmpx_c_vec_orig_shape[0:-1][...,np.newaxis])), cmpx_c_vec, axis=-1)
		(n_elems, n_freqs) = cmpx_c_vec.shape
		n_moments = n_freqs - 1
		reconstruction = np.zeros(cmpx_c_vec.shape[0:-1] + (n_bins,))
		# Set constants
		e0 = np.eye(cmpx_c_vec.shape[-1], 1)
		# Construct toeplitz matrix. We have our own implementation where the toeplitz matrix is constructed the last dimension
		# So if cmpx_c_vec is NxMxK we will output B as a NxMxKxK, where the last 2 dims are the toeplitz for each MxN element
		B = signalproc_ops.broadcast_toeplitz(cmpx_c_vec)
		eig_vals,eig_vecs=np.linalg.eigh(B)
		min_eig_val_indeces=np.argmin(eig_vals, axis=-1)
		# Construct the polynomials with the eig_vecs with smallest eig_vals
		# Then compute the roots of that polynomial which tell you the location of the delta peaks
		dirac_delta_locs = np.zeros((n_elems, n_moments), dtype=eig_vecs.dtype)
		exponents = np.arange(1, n_moments+1)[:,np.newaxis].repeat(n_moments,1) 
		for i in range(n_elems):
			polynomial = eig_vecs[i, :, min_eig_val_indeces[i]]
			roots = np.roots(np.conj(polynomial[::-1]))
			n_roots = len(roots)
			# print("nroots = {}".format(n_roots))
			dirac_delta_locs[i, 0:n_roots] = roots[0:n_roots]
			curr_dirac_delta_locs = dirac_delta_locs[i, :] 
			# Compute the weights via lstsq system
			vandermonde = curr_dirac_delta_locs[np.newaxis,:].repeat(n_moments, axis=0) ** (exponents) 
			(weights, residuals, rank, s) = np.linalg.lstsq(vandermonde, cmpx_c_vec[i, 1:], rcond=None)
			weights = weights.real
			angles = np.angle(curr_dirac_delta_locs)
			angles = linearize_phase(angles)
			indeces = np_utils.domain2index(angles, TWOPI, n_bins)
			reconstruction[i, indeces] = weights
		return reconstruction.reshape(cmpx_c_vec_orig_shape[0:-1] + (n_bins,))

	def construct_phasor(self, c_vec):
		return c_vec[..., 0::2] - 1j*c_vec[..., 1::2]

	def construct_fft_transient(self, c_vec):
		fft_transient = np.zeros(c_vec.shape[0:-1] + (self.get_n_maxfreqs(),), dtype=np.complex64)
		# Set the correct frequencies to the correct value
		fft_transient[..., self.freq_idx] = self.construct_phasor(c_vec)
		return fft_transient

	def decode_phase(self, c_vec, query_freq=1):
		assert(query_freq in self.freq_idx), "Input query frequency not available"
		assert(isinstance(query_freq, int)), "input query frequency should be an int"
		phasor = self.construct_phasor(c_vec)[..., query_freq == self.freq_idx].squeeze(-1).conjugate()
		return linearize_phase(np.angle(phasor))

	def circmean_decoding(self, c_vec):
		assert(self.has_first_harmonic()), "First harmonic is required for cirmean calculation"
		circ_mean_phase = self.decode_phase(c_vec, query_freq=1)
		return (circ_mean_phase / TWOPI)*self.n_maxres

	def get_scheme_id(self):
		base_scheme_id = super().get_scheme_id()
		if(0 in self.freq_idx):
			return base_scheme_id + '_withzerothfreq'
		return base_scheme_id



class KTapSinusoidCoding(FourierCoding):
	'''
		Class for KTap Sinusoid Coding that is commonly used in iToF cameras
	'''
	def __init__(self, n_maxres, freq_idx=[0, 1], k=4, **kwargs):
		self.k=k
		super().__init__( n_maxres, freq_idx=freq_idx, **kwargs )

	def set_coding_mat(self, n_maxres, freq_idx):
		'''
		Initialize all frequencies
		'''
		# Check input args
		assert(self.k >= 3), "Number of phase shifts per frequency should be at least 2"
		self.init_coding_mat(n_maxres, freq_idx)
		domain = np.arange(0, self.n_maxres)*(TWOPI / self.n_maxres)
		self.phase_shifts = np.arange(0, self.k)*(TWOPI / self.k)
		for i in range(self.n_freqs):
			start_idx = i*self.k
			for j in range(self.k):
				# self.C[:, start_idx+j] = (0.5*np.cos(self.freq_idx[i]*domain - self.phase_shifts[j])) + 0.5 
				self.C[:, start_idx+j] = np.cos(self.freq_idx[i]*domain - self.phase_shifts[j])
			# self.C[:, cos_idx+2] = np.cos(self.freq_idx[i]*domain - PI)
			# self.C[:, sin_idx+2] = np.sin(self.freq_idx[i]*domain - PI)
		return self.C

	def construct_phasor(self, c_vec):
		# Vectorize c_vec. Some linalg ops do not work if c_vec has more than 2 dims
		(c_vec, c_vec_orig_shape) = np_utils.vectorize_tensor(c_vec)
		cmpx_c_vec = np.zeros(c_vec.shape[0:-1] + (self.n_freqs,), dtype=np.complex64)
		# Allocate Known matrix
		A = np.ones((self.k, 3))
		# For each frequency, construct a matrix of knowns, measurements, and unknowns
		# And solve Ax = b for:
		#		A = [ 1 cos(phase_shifts[j]) sin(phase_shifts[j]]
		#		x = [ offset Amp*cos(phi) Amp*sin(phi)]
		#		b = c_vec for current frequency
		for i in range(self.n_freqs):
			start_idx = i*self.k
			end_idx = start_idx + self.k
			b = np.moveaxis(c_vec[..., start_idx:end_idx], -1, 0) # Make sure that the code axis is the first dim
			# Populate A matrix
			for j in range(self.k):
				A[j, 1] = np.cos(self.phase_shifts[j]) 
				A[j, 2] = np.sin(self.phase_shifts[j])
			(x, residuals, rank, s) = np.linalg.lstsq(A, b, rcond=None)
			x = np.moveaxis(x, 0, -1) # Move result axis to the last dim to match ff_transient dims
			# No need to compute amp and phase since x is directly the real and imag of the fft
			cmpx_c_vec[..., i] = x[..., 1] - 1j*x[..., 2]
			# fft_transient[..., self.freq_idx[i]] = 
			# Amp = np.sqrt(np.square(x[..., 1]) + np.square(x[..., 2]))
			# phi = np.arctan2(x[..., 2] / x[..., 1])
		# Return to the original shape
		return cmpx_c_vec.reshape(c_vec_orig_shape[0:-1] + (self.n_freqs,))

class TruncatedFourierCoding(FourierCoding):
	'''
		Abstract class for linear coding
	'''
	def __init__(self, n_maxres, n_freqs=1, n_codes=None, include_zeroth_harmonic=True, **kwargs):
		if(not (n_codes is None) and (n_codes > 1)): n_freqs = np.ceil(n_codes / 2)
		freq_idx = np.arange(0, n_freqs+1)
		# Remove zeroth harmonic if needed.
		if(not include_zeroth_harmonic): freq_idx = freq_idx[1:]
		self.include_zeroth_harmonic = include_zeroth_harmonic
		super().__init__(n_maxres, freq_idx=freq_idx, n_codes=n_codes, **kwargs)

	def ifft_reconstruction(self, c_vec):
		'''
		Use ifft to reconstruction transient.
		Here since we know that we only have the first K harmonics we can simply apply ifft directly to the phasors
		This rec method is more efficient than the one implemented in FourierCoding
		'''
		self.verify_input_c_vec(c_vec)
		phasors = self.construct_phasor(c_vec)
		# if not available append zeroth harmonic
		if(not self.include_zeroth_harmonic):
			phasors = np.concatenate((np.zeros(phasors.shape[0:-1] + (1,),dtype=phasors.dtype), phasors), axis=-1)
		# Finally return the IFT
		return np.fft.irfft(phasors, axis=-1, n=self.get_n_maxres())

class HighFreqFourierCoding(FourierCoding):
	'''
		Abstract class for High frequency Fourier coding
		This is the type of strategy used in the Micro Phase Shifting paper by Gupta et al., 2015 ACM ToG
	'''
	def __init__(self, n_maxres, n_high_freqs=1, start_high_freq=40, **kwargs):
		freq_idx = np.arange(start_high_freq, start_high_freq+n_high_freqs)
		super().__init__(n_maxres, freq_idx=freq_idx, **kwargs)

class PSeriesFourierCoding(FourierCoding):
	'''
		Abstract class for linear coding
	'''
	def __init__(self, n_maxres, n_freqs=1, include_zeroth_harmonic=False, **kwargs):
		n_power_freqs = int(np.floor(np.log2(n_maxres // 2)))
		power_freq_idx = 2**np.arange(0,n_power_freqs)
		freq_idx = -1*np.ones((n_freqs,))
		freq_idx_candidates = np.arange(0, n_freqs+1)
		# Remove zeroth harmonic if needed.
		self.include_zeroth_harmonic = include_zeroth_harmonic
		if(not include_zeroth_harmonic): freq_idx_candidates = freq_idx_candidates[1:]
		# Add extra frequencies if the power freqs was not enough
		if(n_freqs <= n_power_freqs):
			freq_idx = power_freq_idx[0:n_freqs]
		else:
			freq_idx[0:n_power_freqs] = power_freq_idx
			i = 0
			j = n_power_freqs
			while(j < n_freqs):
				curr_freq_candidate = freq_idx_candidates[i]
				if(not (curr_freq_candidate in freq_idx)):
					freq_idx[j] = curr_freq_candidate
					j+=1
				i+=1
		super().__init__(n_maxres, freq_idx=freq_idx, **kwargs)

class PSeriesBinaryFourierCoding(Coding):
	'''
		class for PSeriesBinaryFourierCoding coding
	'''
	def __init__(self, n_maxres, n_codes=2, **kwargs):
		self.set_coding_mat(n_maxres, n_codes)
		n_power_freqs = int(np.floor(np.log2(n_maxres // 2)))
		if(n_codes <= 2*n_power_freqs):
			print("PSeriesBinary may have ambiguities if there is not enough codes")
		super().__init__(**kwargs)

	def set_coding_mat(self, n_maxres, n_codes):
		'''
		Initialize all frequencies
		'''
		n_freqs = (n_codes // 2) + 1
		self.pseries_fourier_coding = PSeriesFourierCoding(n_maxres, n_freqs=n_freqs)
		## Set C to the same as PSeries
		self.C = np.array(self.pseries_fourier_coding.C[:,0:n_codes])
		## binarize the matrix
		self.C[self.C < 0] = -1 
		self.C[self.C >= 0] = 1 
		return self.C

class RandomFourierCoding(FourierCoding):
	'''
		Abstract class for linear coding
	'''
	def __init__(self, n_maxres, n_rand_freqs=1, include_first_harmonic=True, **kwargs):
		self.include_first_harmonic = include_first_harmonic
		max_freq_idx = n_maxres//2.
		if(include_first_harmonic):
			possible_freq_idx = np.arange(2, max_freq_idx)
			freq_idx = np.random.choice(possible_freq_idx, size=n_rand_freqs-1, replace=False)
			# Always include the first harmonic
			freq_idx = np.append(freq_idx, 1)
		else:
			possible_freq_idx = np.arange(2, max_freq_idx)
			freq_idx = np.random.choice(possible_freq_idx, size=n_rand_freqs, replace=False)
		# sort for better visualization
		freq_idx.sort()
		super().__init__(n_maxres, freq_idx=freq_idx, **kwargs)

class SingleFourierCoding(TruncatedFourierCoding):
	'''
		Fourier coding with only the first harmonic
	'''
	def __init__(self, n_maxres, **kwargs):
		super().__init__(n_maxres=n_maxres, n_freqs=1, include_zeroth_harmonic=False, **kwargs)

class TruncatedKTapSinusoidCoding(KTapSinusoidCoding):
	'''
		Abstract class for linear coding
	'''
	def __init__(self, n_maxres, n_freqs=1, include_zeroth_harmonic=True, k=4, **kwargs):
		freq_idx = np.arange(0, n_freqs+1)
		# Remove zeroth harmonic if needed.
		if(not include_zeroth_harmonic): freq_idx = freq_idx[1:]
		self.include_zeroth_harmonic = include_zeroth_harmonic
		super().__init__(n_maxres, freq_idx=freq_idx, k=k, **kwargs)

	def ifft_reconstruction(self, c_vec):
		'''
		Use ifft to reconstruction transient.
		Here since we know that we only have the first K harmonics we can simply apply ifft directly to the phasors
		This rec method is more efficient than the one implemented in FourierCoding
		'''
		self.verify_input_c_vec(c_vec)
		phasors = self.construct_phasor(c_vec)
		# if not available append zeroth harmonic
		if(not self.include_zeroth_harmonic):
			phasors = np.concatenate(np.zeros(phasors.shape[0:-1],dtype=phasors.dtype), phasors, axis=-1)
		# Finally return the IFT
		return np.fft.irfft(phasors, axis=-1, n=self.get_n_maxres())

class PSeriesGrayBasedFourierCoding(FourierCoding):
	'''
		Gray based fourier coding
		Sample Fourier frequencies in the following way
		- 1, 2, 4, 8, 16 ...
		- 3, 6, 12, 24, 48, ... == 3 * (1, 2, 4, 8, 16, ...)
		- 5, 10, 20, 40, 80, ... == 5 * (1, 2, 4, 8, 16, ...)
		- 7, 14, 28, 56, 112,... == 7 * (1, 2, 4, 8, 16, ...)
		- 9, 18, 36, 72, 144, ... == 9 * (1, 2, 4, 8, 16, ...)
		- 11, 22, 44, 88
		- 13, 26, 52, ...
		- 17, 34, 51
		- 19, 38, 76,
		- 23, 46, 92
	'''
	def __init__(self, n_maxres, n_codes=1, include_zeroth_harmonic=False, filter_freqs=True, valid_freq_thresh=0.1, h_irf=None, account_irf=False):
		n_freqs = (n_codes // 2) + 1
		self.abs_max_freq_idx = n_maxres // 2
		## If we know the IRF we can make sure to only include frequencies with sufficient magnitude
		if(filter_freqs and (not (h_irf is None))):
			(self.low_confidence_freq_idx, self.low_confidence_freqs) = signalproc_ops.get_low_confidence_freqs(h_irf, valid_freq_thresh=valid_freq_thresh)
		else:
			self.low_confidence_freqs = np.zeros((self.abs_max_freq_idx+1,)).astype(bool)
			self.low_confidence_freq_idx = [] 	
		self.n_power_freqs = int(np.floor(np.log2(self.abs_max_freq_idx)))
		self.two_power_freq_idx = 2**np.arange(0,self.n_power_freqs+1)
		freq_idx = -1*np.ones((n_freqs,))
		freq_idx[:] = np.nan
		n_remaining_freqs = n_freqs 
		curr_base_freq = 0
		curr_idx = 0
		if(include_zeroth_harmonic):
			freq_idx[0] = 0
			n_remaining_freqs -= 1
		while(n_remaining_freqs > 0):
			curr_base_freq += 1
			if(curr_base_freq > self.abs_max_freq_idx):
				print("Ran out of freqs.. Adding low confidence frequencies")
				break;
			# Continue to next if curr_base freq is already in
			if(curr_base_freq in freq_idx):
				# print("Skipping curr_base_freq = {} because it is already".format(curr_base_freq))
				continue;
			for i in range(np.min([len(self.two_power_freq_idx), n_remaining_freqs])):
				curr_freq_idx = curr_base_freq*self.two_power_freq_idx[i]
				if((not (curr_freq_idx in freq_idx)) and (curr_freq_idx <= self.abs_max_freq_idx)):
					if(self.low_confidence_freqs[curr_freq_idx]):
						# print("Skipping curr_freq_idx = {} because it is low confidence".format(curr_freq_idx))
						continue;
					else:
						curr_idx = n_freqs - n_remaining_freqs 
						freq_idx[curr_idx] = curr_freq_idx
						n_remaining_freqs -= 1
		## If we ran out of high confidence frequencies add the low confidence ones
		print(n_remaining_freqs)
		for i in range(n_remaining_freqs):
			curr_idx = n_freqs - n_remaining_freqs + i
			print(curr_idx)
			freq_idx[curr_idx] = self.low_confidence_freq_idx[i]

		super().__init__(n_maxres, freq_idx=freq_idx, n_codes=n_codes, h_irf=h_irf, account_irf=account_irf)

class HybridGrayBasedFourierCoding(FourierCoding):
	'''
		Gray based fourier coding
		Sample Fourier frequencies in the following way
		- 1, 2, 4, 8, 16, 32, 64 ...
		If we reach the maximum frequency, then sample the remaining frequencies from lowest to highest, i.e.,
		- 3, 5, 6, 7, 9, 10, 11, 12, ....
	'''
	def __init__(self, n_maxres, n_codes=1, include_zeroth_harmonic=False, filter_freqs=True, valid_freq_thresh=0.1, h_irf=None, account_irf=False):
		n_freqs = (n_codes // 2) + 1
		self.abs_max_freq_idx = n_maxres // 2
		## If we know the IRF we can make sure to only include frequencies with sufficient magnitude
		if(filter_freqs and (not (h_irf is None))):
			(self.low_confidence_freq_idx, self.low_confidence_freqs) = signalproc_ops.get_low_confidence_freqs(h_irf, valid_freq_thresh=valid_freq_thresh)
		else:
			self.low_confidence_freqs = np.zeros((self.abs_max_freq_idx+1,)).astype(bool)
			self.low_confidence_freq_idx = [] 	
		self.n_power_freqs = int(np.floor(np.log2(self.abs_max_freq_idx)))
		self.two_power_freq_idx = 2**np.arange(0,self.n_power_freqs+1)
		freq_idx = -1*np.ones((n_freqs,))
		freq_idx[:] = np.nan
		n_remaining_freqs = n_freqs 
		curr_idx = 0
		if(include_zeroth_harmonic):
			freq_idx[0] = 0
			n_remaining_freqs -= 1
		## Add power of two freqs
		for i in range(len(self.two_power_freq_idx)):
			if(n_remaining_freqs == 0): break;
			curr_freq_idx = self.two_power_freq_idx[i]
			if(self.low_confidence_freqs[curr_freq_idx]):
				# print("Skipping curr_freq_idx = {} because it is low confidence".format(curr_freq_idx))
				continue;
			else:
				curr_idx = n_freqs - n_remaining_freqs 
				freq_idx[curr_idx] = curr_freq_idx
				n_remaining_freqs -= 1
		## 
		curr_freq_idx = 0
		while(n_remaining_freqs > 0):
			curr_freq_idx += 1 
			if(curr_freq_idx > self.abs_max_freq_idx):
				print("Ran out of freqs.. Adding low confidence frequencies")
				break;
			if((not self.low_confidence_freqs[curr_freq_idx]) and (not (curr_freq_idx in freq_idx))):
				curr_idx = n_freqs - n_remaining_freqs 
				freq_idx[curr_idx] = curr_freq_idx
				n_remaining_freqs -= 1
		## If we ran out of high confidence frequencies add the low confidence ones
		print(n_remaining_freqs)
		for i in range(n_remaining_freqs):
			curr_idx = n_freqs - n_remaining_freqs + i
			print(curr_idx)
			freq_idx[curr_idx] = self.low_confidence_freq_idx[i]
		
		super().__init__(n_maxres, freq_idx=freq_idx, n_codes=n_codes, h_irf=h_irf, account_irf=account_irf)

class HybridFourierBasedGrayCoding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.set_coding_mat(n_maxres, n_codes)
		super().__init__( **kwargs )
	
	def set_coding_mat(self, n_maxres, n_codes):
		self.gray_coding_obj = ComplementaryGrayCoding(n_maxres, n_codes)
		self.hybridgrayfourier_coding_obj = HybridGrayBasedFourierCoding(n_maxres, n_codes=n_codes, filter_freqs=False )
		self.gray_C = np.array(self.gray_coding_obj.C)
		n_gray_codes = self.gray_C.shape[-1] 
		## If we have codes left start adding binary codes
		if(n_codes > n_gray_codes):
			self.C = np.array(self.hybridgrayfourier_coding_obj.C)
			self.C[self.C >= 1e-7] = 1
			self.C[self.C < -1e-7] = -1
		else:
			self.C = self.gray_C
		return self.C

class PSeriesFourierBasedGrayCoding(Coding):
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.set_coding_mat(n_maxres, n_codes)
		super().__init__( **kwargs )
	
	def set_coding_mat(self, n_maxres, n_codes):
		self.gray_coding_obj = ComplementaryGrayCoding(n_maxres, n_codes)
		self.psergrayfourier_coding_obj = PSeriesGrayBasedFourierCoding(n_maxres, n_codes=n_codes, filter_freqs=False )
		self.gray_C = np.array(self.gray_coding_obj.C)
		n_gray_codes = self.gray_C.shape[-1] 
		## If we have codes left start adding binary codes
		if(n_codes > n_gray_codes):
			self.C = np.array(self.psergrayfourier_coding_obj.C)
			self.C[self.C >= 1e-7] = 1
			self.C[self.C < -1e-7] = -1
		else:
			self.C = self.gray_C
		return self.C