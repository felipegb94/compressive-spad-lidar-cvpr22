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
from scipy import signal, interpolate
from scipy.special import softmax
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from toflib.coding import Coding
from research_utils.np_utils import to_nparray
from research_utils.shared_constants import *
from research_utils import signalproc_ops, np_utils, py_utils



# function decimalToVector
# input: numbers n and r (0 <= n<2**r)
# output: a string v of r bits representing n
def decimalToVector(n, r):
	v = []
	for s in range(r):
		v.insert(0, n % 2)
		n //= 2
	return v

def hammingGeneratorMatrix(r):
	n = 2 ** r - 1
	# construct permutation pi
	pi = []
	for i in range(r):
		pi.append(2 ** (r - i - 1))
	for j in range(1, r):
		for k in range(2 ** j + 1, 2 ** (j + 1)):
			pi.append(k)
	# construct rho = pi^(-1)
	rho = []
	for i in range(n):
		rho.append(pi.index(i + 1))
	# construct H'
	H = []
	for i in range(r, n):
		H.append(decimalToVector(pi[i], r))
	# construct G'
	GG = [list(i) for i in zip(*H)]
	for i in range(n - r):
		GG.append(decimalToVector(2 ** (n - r - i - 1), n - r))
	# apply rho to get Gtranpose
	G = []
	for i in range(n):
		G.append(GG[rho[i]])
	# transpose
	G = [list(i) for i in zip(*G)]
	return np.array(G)

class HammingCoding(Coding):
	'''
		Hamming coding class. 
	'''
	def __init__(self, n_maxres, n_parity_bits=None, **kwargs):
		if(n_parity_bits is None): n_parity_bits=3
		self.n_maxres = n_maxres
		self.n_parity_bits = n_parity_bits 
		self.set_coding_mat(self.n_maxres, self.n_parity_bits)
		super().__init__(**kwargs)

	def set_coding_mat(self, n_maxres, n_parity_bits):
		self.hamming_codes = hammingGeneratorMatrix(n_parity_bits).transpose()
		self.n_codes = self.hamming_codes.shape[-1]
		self.C = np.zeros((n_maxres, self.n_codes))
		self.min_hamming_code_length = self.hamming_codes.shape[0]
		assert(n_maxres >= self.min_hamming_code_length), "n_maxres is not large enough to encode the gray code"
		if((n_maxres % self.min_hamming_code_length) != 0):
			print("WARNING: Hamming codes where the n_maxres is not a multiple of the hamming code length, may have some small ambiguous regions")
		self.x_fullres = np.arange(0, n_maxres) * (1. / n_maxres)
		self.x_lowres = np.arange(0, self.min_hamming_code_length) * (1. / self.min_hamming_code_length)
		ext_x_lowres = np.arange(-1, self.min_hamming_code_length+1) * (1. / self.min_hamming_code_length)
		ext_hamming_codes = np.concatenate((self.hamming_codes[-1,:][np.newaxis, :], self.hamming_codes, self.hamming_codes[0,:][np.newaxis,:]), axis=0)
		f = interpolate.interp1d(ext_x_lowres, ext_hamming_codes, axis=0, kind='linear')
		self.C = f(self.x_fullres)
		self.C = (self.C*2)-1
		self.C = self.C - self.C.mean(axis=-2, keepdims=True)

