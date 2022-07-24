'''
	Base class for temporal coding schemes
'''
## Standard Library Imports
from abc import ABC, abstractmethod
import math 
import os
from random import gauss
import warnings

## Library Imports
import numpy as np
from scipy import signal
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

if __name__=='__main__':
    import matplotlib.pyplot as plt

    # Load some data
    transient_img = np.load('./sample_data/scene_irf_data/vgroove_transient_img.npz')['arr_0'].astype(np.float)
    rgb_img = np.load('./sample_data/rgb_data/vgroove.npy')
    depth_img = np.load('./sample_data/depth_data/vgroove.npy')

    # Number of rows, cols, and discrete time bins
    (nr, nc, nt) = transient_img.shape
    t_domain = np.arange(0, nt) 
    # Sinusoid coding parameters
    ktaps = 3
    freqs = [2,5]
    # Set IRF
    delta_irf = np.zeros((nt,))
    delta_irf[0] = 1
    from research_utils.signalproc_ops import gaussian_pulse
    gauss_irf = gaussian_pulse(time_domain=t_domain, mu=0, width=2, circ_shifted=True)

    # Init coding obj
    c_obj = KTapSinusoidCoding(nt, freq_idx=freqs, k=ktaps, h_irf=delta_irf)

    ## Single-Pixel Simulate
    n_total_photons = 500
    n_ambient_photons = 1000
    transient_px = transient_img[nr//4, nc//4, :]

    # 1. Scale transient and add ambient
    transient_px = (transient_px / transient_px.sum())*n_total_photons 
    transient_px += (n_ambient_photons / nt)
    true_time_shift = transient_px.argmax()
    # 
    plt.clf()
    plt.plot(transient_px, label='True Transient')

    # 2. Add poisson noise
    from toflib.tof_utils import add_poisson_noise
    transient_px = add_poisson_noise(transient_px, n_mc_samples=1).squeeze()
    plt.plot(transient_px, label='Incident Transient')
    plt.legend()

    # 3. Capture ToF data
    b_vals = c_obj.encode(transient_px)

    # 4. Add read noise
    read_noise_stddev = 20
    b_vals += np.random.normal(0, read_noise_stddev)

    # 5. estimate depths
    est_time_shift = c_obj.zncc_depth_decoding(b_vals)

    print("True Time Shift (in time bins): {}".format(true_time_shift))
    print("Est. Time Shift (in time bins): {}".format(est_time_shift))




