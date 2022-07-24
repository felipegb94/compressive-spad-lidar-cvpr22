'''
	Base class for temporal coding schemes based on ECC coding
'''
## Standard Library Imports
from abc import ABC, abstractmethod
import math 
import os

## Library Imports
import numpy as np
import scipy
import pywt
from scipy import signal, interpolate
from scipy.special import softmax
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from toflib.coding import Coding
from research_utils.shared_constants import *


def generate_continuous_wavelet_codes(wavelet, n_maxres, level=1):
	n_codes = 2**(level-1)
	codes = np.zeros((n_maxres, n_codes))
	wavelet_len = int(np.floor(n_maxres / n_codes))
	(phi, x) = wavelet.wavefun(length=wavelet_len)
	for i in range(n_codes):
		start_idx = i*wavelet_len
		end_idx = np.min((start_idx+wavelet_len, n_maxres))
		codes[start_idx:end_idx, i] = phi
	return codes


class WaveletCoding(Coding):
	'''
		Gray coding class. 
	'''
	def __init__(self, n_maxres, n_codes, **kwargs):
		self.n_maxres = n_maxres
		self.n_codes = n_codes
		self.set_coding_mat(self.n_maxres, self.n_codes)
		super().__init__(**kwargs)

	def set_coding_mat(self, n_maxres, n_codes):
		self.C = np.zeros((n_maxres, n_codes))
		## Get wavelet
		gauss_wavelet = pywt.ContinuousWavelet('mexh')
		n_codes_remaining = self.n_codes
		curr_level = 1
		curr_start_code_idx = 0
		while(n_codes_remaining > 0):
			curr_codes = generate_continuous_wavelet_codes(gauss_wavelet, self.n_maxres, level=curr_level)
			curr_n_codes = curr_codes.shape[-1]
			end_code_idx = np.min([curr_start_code_idx + curr_n_codes, n_codes])
			self.C[:, curr_start_code_idx:end_code_idx] = curr_codes[:,0:end_code_idx-curr_start_code_idx] 
			# Update counters
			curr_level += 1
			curr_start_code_idx = end_code_idx
			n_codes_remaining -= curr_n_codes
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

if __name__=='__main__':
	import matplotlib.pyplot as plt

	nt = 1250
	n_codes = 16
	gauss_wavelet = pywt.ContinuousWavelet('gaus2')


	wavelet_coding_obj = WaveletCoding(nt, n_codes)

	plt.clf()
	# plt.plot(wavelet_coding_obj.C)

	plt.imshow(wavelet_coding_obj.get_pretty_C())

	# codes = np.zeros((nt, n_codes))
	# n_codes_remaining = n_codes
	# curr_level = 1
	# curr_start_code_idx = 0
	# while(n_codes_remaining > 0):
	# 	curr_codes = generate_continuous_wavelet_codes(gauss_wavelet, nt, level=curr_level)
	# 	curr_n_codes = curr_codes.shape[-1]
	# 	end_code_idx = np.min([curr_start_code_idx + curr_n_codes, n_codes])
	# 	codes[:, curr_start_code_idx:end_code_idx] = curr_codes[:,0:end_code_idx-curr_start_code_idx] 
	# 	# Update counters
	# 	curr_level += 1
	# 	curr_start_code_idx = end_code_idx
	# 	n_codes_remaining -= curr_n_codes
