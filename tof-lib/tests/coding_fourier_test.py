'''
	Test functionality of toflib.coding: FourierCoding
	Example run command:
		run tests/coding_fourier_test.py -freq_idx 0 1 2 3 4 5 6 7 8 9 10 -n_tbins 1000
'''
#### Standard Library Imports
import argparse
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from toflib import coding
from toflib import tof_utils, input_args_utils
from research_utils.signalproc_ops import standardize_signal
from research_utils.np_utils import vectorize_tensor
from tirf_modelbased_test import generate_model_tirf

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for scene_tirf_test.')
	parser = input_args_utils.add_fourier_coding_args(parser)
	args = parser.parse_args()
	# Parse input args
	freq_idx = args.freq_idx
	n_tbins = args.n_tbins

	# initialize fourier coding
	fourier_coding = coding.FourierCoding(n_tbins, freq_idx=freq_idx)
	ktap = 4
	sinusoid_coding = coding.KTapSinusoidCoding(n_tbins, freq_idx=freq_idx, k=ktap)
	fourier_coding.lres_mode = False
	sinusoid_coding.lres_mode = False
	domain = fourier_coding.get_domain()
	gt_domain = fourier_coding.domain

	# get some random transient 
	## The following produces good examples to test batch processing. 
	# n_samples = 2
	# (gauss_tirf1, expgauss_tirf1) = generate_model_tirf(n_tbins, n_samples=n_samples)
	# (g_transient1, expg_transient1) = (gauss_tirf1.tirf, expgauss_tirf1.tirf)
	# (gauss_tirf2, expgauss_tirf2) = generate_model_tirf(n_tbins, n_samples=n_samples)
	# (g_transient2, expg_transient2) = (gauss_tirf2.tirf, expgauss_tirf2.tirf)
	# g_transient = np.zeros((2, n_samples, n_tbins))
	# expg_transient = np.zeros((2, n_samples, n_tbins))
	# g_transient[0,:] = g_transient1
	# g_transient[1,:] = g_transient2
	# expg_transient[0,:] = expg_transient1
	# expg_transient[1,:] = expg_transient2
	## The following produces good examples to test implemented algorithms. 
	n_samples = 1
	(gauss_tirf, expgauss_tirf) = generate_model_tirf(n_tbins, n_samples=n_samples)
	(g_transient, expg_transient) = (gauss_tirf.tirf, expgauss_tirf.tirf)	
	if(g_transient.ndim > 1):
		g_transient = g_transient.sum(axis=0) 
		expg_transient = expg_transient.sum(axis=0) 

	# Get true fft encoding and decoding
	fft_g_transient = np.fft.rfft(g_transient, axis=-1)
	fft_expg_transient = np.fft.rfft(expg_transient, axis=-1)
	# Zero out all elements that are not in freq_idx
	for i in range(fft_g_transient.shape[-1]):
		if (not (i in freq_idx)):
			fft_g_transient[..., i] = 0
			fft_expg_transient[..., i] = 0
		else:
			print("Keeping freq {}".format(i))
	gt_g_transient_ifft_rec = np.fft.irfft(fft_g_transient, axis=-1)
	gt_expg_transient_ifft_rec = np.fft.irfft(fft_expg_transient, axis=-1)

	# Encode the transient signals
	g_c_vec = fourier_coding.encode(g_transient)
	expg_c_vec = fourier_coding.encode(expg_transient)

	sinusoid_g_c_vec = sinusoid_coding.encode(g_transient)
	sinusoid_expg_c_vec = sinusoid_coding.encode(expg_transient)

	## Reconstruct signal using different sinusoid models
	# ZNCC Rec
	g_transient_zncc_rec = fourier_coding.zncc_reconstruction(g_c_vec)
	expg_transient_zncc_rec = fourier_coding.zncc_reconstruction(expg_c_vec)
	sinusoid_g_transient_zncc_rec = sinusoid_coding.zncc_reconstruction(sinusoid_g_c_vec)
	sinusoid_expg_transient_zncc_rec = sinusoid_coding.zncc_reconstruction(sinusoid_expg_c_vec)
	# IFT Rec
	g_transient_ifft_rec = fourier_coding.ifft_reconstruction(g_c_vec)
	expg_transient_ifft_rec = fourier_coding.ifft_reconstruction(expg_c_vec)
	sinusoid_g_transient_ifft_rec = sinusoid_coding.ifft_reconstruction(sinusoid_g_c_vec)
	sinusoid_expg_transient_ifft_rec = sinusoid_coding.ifft_reconstruction(sinusoid_expg_c_vec)
	# Basis Rec
	g_transient_basis_rec = fourier_coding.basis_reconstruction(g_c_vec)
	expg_transient_basis_rec = fourier_coding.basis_reconstruction(expg_c_vec)
	sinusoid_g_transient_basis_rec = sinusoid_coding.basis_reconstruction(sinusoid_g_c_vec)
	sinusoid_expg_transient_basis_rec = sinusoid_coding.basis_reconstruction(sinusoid_expg_c_vec)
	# Circular Mean Rec
	g_transient_circmean_rec = fourier_coding.circmean_reconstruction(g_c_vec)
	expg_transient_circmean_rec = fourier_coding.circmean_reconstruction(expg_c_vec)
	sinusoid_g_transient_circmean_rec = sinusoid_coding.circmean_reconstruction(sinusoid_g_c_vec)
	sinusoid_expg_transient_circmean_rec = sinusoid_coding.circmean_reconstruction(sinusoid_expg_c_vec)
	# # MESE Rec
	g_transient_mese_rec = fourier_coding.mese_reconstruction(g_c_vec)
	expg_transient_mese_rec = fourier_coding.mese_reconstruction(expg_c_vec)
	sinusoid_g_transient_mese_rec = sinusoid_coding.mese_reconstruction(sinusoid_g_c_vec)
	sinusoid_expg_transient_mese_rec = sinusoid_coding.mese_reconstruction(sinusoid_expg_c_vec)
	# # Pizarenko Rec
	g_transient_piza_rec = fourier_coding.pizarenko_reconstruction(g_c_vec)
	expg_transient_piza_rec = fourier_coding.pizarenko_reconstruction(expg_c_vec)
	sinusoid_g_transient_piza_rec = sinusoid_coding.pizarenko_reconstruction(sinusoid_g_c_vec)
	sinusoid_expg_transient_piza_rec = sinusoid_coding.pizarenko_reconstruction(sinusoid_expg_c_vec)

	plt.clf()
	## Plot Sinusoid codes
	plt.subplot(3,2,1)
	plt.plot(domain, fourier_coding.get_input_C()[:,0::2], linewidth=2)
	plt.title("Cosine Codes", fontsize=14)
	plt.subplot(3,2,2)
	plt.plot(domain, fourier_coding.get_input_zn_C()[:,0::2], linewidth=2)
	plt.title("Zero-Norm Cosine Codes", fontsize=14)

	## Plot transient and reconstructed transient
	plt.subplot(3,1,2)
	plt.plot(gt_domain, vectorize_tensor(standardize_signal(g_transient))[0].transpose(), linewidth=2, label='GT Transient (0-1 scaled)')
	# plt.plot(domain, vectorize_tensor(standardize_signal(gt_g_transient_ifft_rec))[0].transpose(), '-', linewidth=4, label='GT Fourier IFFT Rec')
	plt.plot(domain, vectorize_tensor(standardize_signal(g_transient_zncc_rec))[0].transpose(), linewidth=2, label='Fourier ZNCC Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(g_transient_ifft_rec))[0].transpose(), linewidth=2, label='Fourier IFFT Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(g_transient_basis_rec))[0].transpose(), linewidth=2, label='Fourier Basis Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(g_transient_mese_rec))[0].transpose(), linewidth=2, label='Fourier MESE Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(g_transient_piza_rec))[0].transpose(), linewidth=2, label='Fourier Pizarenko Rec (0-1 scaled)')
	# plt.plot(domain, vectorize_tensor(standardize_signal(g_transient_circmean_rec))[0].transpose(), linewidth=2, label='Fourier Circ-Mean Rec (0-1 scaled)')
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_g_transient_zncc_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid ZNCC Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_g_transient_ifft_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid IFFT Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_g_transient_basis_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid Basis Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_g_transient_mese_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid MESE Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_g_transient_piza_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid Pizarenko Rec (0-1 scaled)'.format(ktap))
	plt.legend()
	plt.title("Gaussian Reconstruction", fontsize=14)

	plt.subplot(3,1,3)
	plt.plot(gt_domain, vectorize_tensor(standardize_signal(expg_transient))[0].transpose(), linewidth=2, label='GT Transient (0-1 scaled)')
	# plt.plot(domain, vectorize_tensor(standardize_signal(gt_expg_transient_ifft_rec))[0].transpose(), '-', linewidth=4, label='GT Fourier IFFT Rec')
	plt.plot(domain, vectorize_tensor(standardize_signal(expg_transient_zncc_rec))[0].transpose(), linewidth=2, label='Fourier ZNCC Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(expg_transient_ifft_rec))[0].transpose(), linewidth=2, label='Fourier IFFT Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(expg_transient_basis_rec))[0].transpose(), linewidth=2, label='Fourier Basis Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(expg_transient_mese_rec))[0].transpose(), linewidth=2, label='Fourier MESE Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(expg_transient_piza_rec))[0].transpose(), linewidth=2, label='Fourier Pizarenko Rec (0-1 scaled)')
	plt.plot(domain, vectorize_tensor(standardize_signal(expg_transient_circmean_rec))[0].transpose(), linewidth=2, label='Fourier Circ-Mean Rec (0-1 scaled)')
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_expg_transient_zncc_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid ZNCC Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_expg_transient_ifft_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid IFFT Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_expg_transient_basis_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid Basis Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_expg_transient_mese_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid MESE Rec (0-1 scaled)'.format(ktap))
	# plt.plot(domain, vectorize_tensor(standardize_signal(sinusoid_expg_transient_piza_rec))[0].transpose(), '--', linewidth=2, label='{}-Tap Sinusoid Pizarenko Rec (0-1 scaled)'.format(ktap))
	plt.legend()
	plt.title("Exp-Modified Gaussian Reconstruction", fontsize=14)