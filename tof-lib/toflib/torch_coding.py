'''
	temporal coding schemes implemented in pytorch
'''
## Standard Library Imports
from abc import ABC, abstractmethod
import math 

## Library Imports
import numpy as np
import torch
import torch.nn as nn
import torch.fft as torchfft
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.np_utils import to_nparray
from research_utils.shared_constants import *
from research_utils import signalproc_ops, np_utils, py_utils
from toflib.torch_tof_utils import norm_t, zero_norm_t, linearize_phase
from toflib import coding 


class CodingLayer(nn.Module):
	'''
		Class for coding class based on input data
	'''
	def __init__(self, C=None, n_maxres=1024, n_codes=4, domain=None, init_id='random'):
		super(CodingLayer, self).__init__()
		self.dtype = torch.float32
		self.softmax = nn.Softmax(dim=-1)
		self.tanh = nn.Tanh()
		# initialize C
		# set to random zero mean matrix if none
		if(C is None):
			self.init_id = init_id
			if(init_id == 'truncfourier'):
				n_freqs = int(np.ceil(float(n_codes) / 2.))
				coding_obj = coding.TruncatedFourierCoding(n_maxres, n_freqs=n_freqs, include_zeroth_harmonic=False)
				C = coding_obj.C[:, 0:n_codes]
			elif(init_id == 'gray'):
				C = (np.random.rand(n_maxres, n_codes)*2) - 1
				coding_obj = coding.GrayCoding(n_maxres, n_bits=n_codes)
				C[:, 0:coding_obj.n_codes] = coding_obj.C
			else:
				C = ((np.random.rand(n_maxres, n_codes)*2) - 1)*0.01
				# C = (np.random.rand(n_maxres, n_codes)*0.1) + 0.45
				# C = (np.random.rand(n_maxres, n_codes)*0.01)
				C = C - C.mean(axis=-2, keepdims=True)
		self.C_unconstrained = nn.Parameter(torch.tensor(C).type(self.dtype), requires_grad=True)
		# if domain is not none set n_maxres according to it
		if(not (domain is None)):
			self.domain = nn.Parameter(torch.tensor(domain).type(self.dtype), requires_grad=False)
		# Update C parameters given new C
		self.update_C()
		# Store how many codes there are
		(self.n_maxres, self.n_codes) = (self.C.shape[-2], self.C.shape[-1])
		assert(self.n_codes <= self.n_maxres), "n_codes ({}) should not be larger than n_maxres ({})".format(self.n_codes, self.n_maxres)
		self.update_base_coding()
		# Set domains
		self.discrete_domain = nn.Parameter(torch.arange(0, self.n_maxres).type(self.dtype), requires_grad=False)
		if(domain is None):
			self.domain = nn.Parameter(torch.arange(0, self.n_maxres).type(self.dtype), requires_grad=False)
		else:
			assert(self.domain.numel() == self.n_maxres), "invalid input domain"
	
	def normalize_domain(self, domain):
		return (2*domain*(1./self.n_maxres)) - 1

	def forward(self, x):
		return self.encode(x)

	def update_C(self):
		# if(not (C is None)): self.C = C
		# assert((self.C_unconstrained.shape[-2] == self.n_maxres) or (self.C_unconstrained.shape[-1] == self.n_codes)), "Can't resize C"
		# make C between -1 and 1
		## Bandlimit + tanh
		n_maxres = self.C_unconstrained.shape[0]
		self.C_bandlimited = torchfft.irfft(torchfft.rfft(self.C_unconstrained, dim=0)[0:n_maxres//4,:], dim=0, n=n_maxres)
		self.C_constrained = self.C_bandlimited
		## tanh
		# self.C_constrained = torch.tanh(self.C_unconstrained)
		# ## Nothing
		# self.C_constrained = self.C_unconstrained
		## Zero mean
		self.C = self.C_constrained - torch.mean(self.C_constrained, dim=0, keepdim=True)
		# Re-compute some useful quantities
		self.zero_norm_C = zero_norm_t(self.C)
		self.norm_C = norm_t(self.C)
		# Set normalized domain

	def update_base_coding(self): self.base_coding_obj = coding.DataCoding(C = self.C.detach().data.cpu().numpy())
	
	def verify_input_c_vec(self, c_vec):
		assert(c_vec.shape[-1] == self.n_codes), "Input c_vec does not have the correct dimensions"

	def get_rec_algo_func(self, rec_algo_id):
		# Check if rec algorithm exists
		rec_algo_func_name = '{}_reconstruction'.format(rec_algo_id)
		rec_algo_function = getattr(self, rec_algo_func_name, None)
		assert(rec_algo_function is not None), "Reconstruction algorithm {} is NOT available. Please choose from the following algos: {}".format(rec_algo_func_name, self.rec_algos_avail)
		# # Apply rec algo
		# print("Running reconstruction algorithm {}".format(rec_algo_func_name))
		return rec_algo_function

	def get_input_C(self, input_C=None, C_mode=None):
		if(input_C is None):
			if(C_mode == 'norm'): input_C = self.norm_C
			elif(C_mode == 'zeronorm'): input_C = self.zero_norm_C
			else: input_C = self.C
		self.verify_input_c_vec(input_C) # Last dim should be the codes
		return input_C

	def encode(self, transient_img):
		'''
		Encode the transient image using the n_codes inside the self.C matrix
		'''
		self.update_C()
		transient_img=self.to_torch_tensor(transient_img)
		assert(transient_img.shape[-1] == self.n_maxres), "Input tensor does not have the correct dimensions"
		return torch.matmul(transient_img.unsqueeze(-2), self.C).squeeze(-2)

	def xcorr1D(self, Cmat, c_vec):
		assert(Cmat.shape[-1] == c_vec.shape[-1]), 'last dim should match'
		return torch.matmul(Cmat, c_vec.unsqueeze(-1)).squeeze(-1)

	def to_torch_tensor(self, input_tensor):
		if(not torch.is_tensor(input_tensor)): return torch.tensor(input_tensor, dtype=self.dtype)
		else: return input_tensor

	def ncc_reconstruction(self, c_vec, input_C=None, c_vec_is_norm=False):
		'''
		NCC Reconstruction: Works for any arbitrary set of zero-mean codes
		'''
		self.verify_input_c_vec(c_vec)
		# Make c_vec zero norm if needed
		if(not c_vec_is_norm): norm_c_vec = norm_t(c_vec, axis=-1)
		else: norm_c_vec = c_vec
		# If no input_C is provided use one of the existing ones
		input_C = self.get_input_C(input_C, C_mode='norm')
		# Perform zncc
		return self.xcorr1D(input_C, norm_c_vec)

	def zncc_reconstruction(self, c_vec, input_C=None, c_vec_is_norm=False):
		'''
		NCC Reconstruction: Works for any arbitrary set of zero-mean codes
		'''
		self.verify_input_c_vec(c_vec)
		# Make c_vec zero norm if needed
		if(not c_vec_is_norm): zero_norm_c_vec = zero_norm_t(c_vec, axis=-1)
		else: zero_norm_c_vec = c_vec
		# If no input_C is provided use one of the existing ones
		input_C = self.get_input_C(input_C, C_mode='zero_norm')
		# Perform zncc
		return self.xcorr1D(input_C, zero_norm_c_vec)

	def softncc_reconstruction(self, c_vec, input_C=None, c_vec_is_norm=False, beta=100):
		return self.softmax(self.ncc_reconstruction(c_vec, input_C=None, c_vec_is_norm=False)*beta)

	def reconstruction(self, c_vec, rec_algo_id='ncc', **kwargs):
		c_vec=self.to_torch_tensor(c_vec)
		rec_algo_function = self.get_rec_algo_func(rec_algo_id)
		lookup = rec_algo_function(c_vec, **kwargs)
		return lookup

	def max_peak_decoding(self, c_vec, rec_algo_id='ncc', **kwargs):
		'''
			Perform max peak decoding using the specified reconstruction algorithm
			kwargs (key-work arguments) will depend on the chosen reconstruction algorithm 
		'''
		c_vec=self.to_torch_tensor(c_vec)
		lookup = self.reconstruction(c_vec, rec_algo_id, **kwargs)
		return torch.argmax(lookup, axis=-1)

	def softmax_peak_decoding(self, c_vec, rec_algo_id='ncc', beta=100, use_norm_domain=False, **kwargs):
		'''
			Perform max peak decoding using the specified reconstruction algorithm
			kwargs (key-work arguments) will depend on the chosen reconstruction algorithm 
		'''
		c_vec = self.to_torch_tensor(c_vec)
		lookup = self.reconstruction(c_vec, rec_algo_id, **kwargs)
		if(use_norm_domain): domain = self.normalize_domain(self.discrete_domain)
		else: domain = self.discrete_domain
		return torch.matmul(self.softmax(beta*lookup), domain.unsqueeze(-1)).squeeze(-1)

	def get_pretty_C(self):
		self.update_base_coding()
		return self.base_coding_obj.get_pretty_C()


class HybridFourierCodingLayer(CodingLayer):
	def __init__(self, n_maxres=1024, n_codes=2):
		self.n_maxres = n_maxres
		self.n_codes_total = n_codes
		self.n_opt_codes = int(np.ceil(float(n_codes)/2.))
		# self.n_opt_codes = 10
		self.n_fourier_codes = self.n_codes_total - self.n_opt_codes
		self.fourier_C = None
		super(HybridFourierCodingLayer, self).__init__(C=None, n_maxres=self.n_maxres, n_codes=self.n_opt_codes)
		if(self.n_fourier_codes >= 1):
			n_freqs = int(np.ceil(float(self.n_fourier_codes) / 2.))
			coding_obj = coding.TruncatedFourierCoding(n_maxres, n_freqs=2*n_freqs, include_zeroth_harmonic=False)
			# coding_obj = coding.GrayCoding(n_maxres, n_bits=4)
			self.fourier_C = nn.Parameter(torch.tensor(coding_obj.C[:, 0:self.n_fourier_codes], dtype=torch.float32), requires_grad=False)
			# self.fourier_C = nn.Parameter(torch.tensor(coding_obj.C[:, -self.n_fourier_codes:], dtype=torch.float32), requires_grad=False)

	def update_C(self):
		super().update_C()
		## 
		if(self.fourier_C is None):
			self.hybrid_C = self.C
		else:
			self.hybrid_C = torch.cat((self.fourier_C, self.C), dim=-1)
		# Re-compute some useful quantities
		self.zero_norm_C = zero_norm_t(self.hybrid_C)
		self.norm_C = norm_t(self.hybrid_C)

	def verify_input_c_vec(self, c_vec):
		assert(c_vec.shape[-1] == self.n_codes_total), "Input c_vec does not have the correct dimensions"

	def get_input_C(self, input_C=None, C_mode=None):
		if(input_C is None):
			if(C_mode == 'norm'): input_C = self.norm_C
			elif(C_mode == 'zeronorm'): input_C = self.zero_norm_C
			else: input_C = self.hybrid_C
		self.verify_input_c_vec(input_C) # Last dim should be the codes
		return input_C

	def encode(self, transient_img):
		'''
		Encode the transient image using the n_codes inside the self.C matrix
		'''
		self.update_C()
		transient_img=self.to_torch_tensor(transient_img)
		assert(transient_img.shape[-1] == self.n_maxres), "Input tensor does not have the correct dimensions"
		return torch.matmul(transient_img.unsqueeze(-2), self.hybrid_C).squeeze(-2)

class FourierCodingLayer(CodingLayer):
	'''
		Class for coding class based on input data
	'''
	def __init__(self, n_maxres=1024, n_freqs=2):
		self.n_freqs = n_freqs
		self.n_maxres = n_maxres
		self.freqs = np.arange(0, n_freqs).astype(np.float32) + 1
		# self.freqs = np.random.rand(n_freqs).astype(np.float32)*(self.n_maxres//8)
		self.phase_domain = np.arange(0, self.n_maxres)*(TWOPI / self.n_maxres)
		super(FourierCodingLayer, self).__init__(C=self.freqs, n_maxres=self.n_maxres, n_codes=self.n_freqs*2, domain=self.phase_domain)

	def update_C(self):
		## Nothing
		self.phase = torch.outer(self.domain, self.C_unconstrained)
		## Zero mean
		self.C_constrained = torch.cat((torch.cos(self.phase), torch.sin(self.phase)), dim=-1)
		self.C = self.C_constrained
		# Re-compute some useful quantities
		self.zero_norm_C = zero_norm_t(self.C)
		self.norm_C = norm_t(self.C)
		# Set normalized domain

