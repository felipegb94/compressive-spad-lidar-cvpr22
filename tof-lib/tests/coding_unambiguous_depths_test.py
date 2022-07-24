'''
	Test functionality of toflib.coding.py:DataCoding Class
	Sample Run command: 

	Fourier-Domain Testing Commands (That should produce 0 error):
		run tests/coding_unambiguous_depths_test.py -coding Fourier -rec ifft -freq_idx 1
		run tests/coding_unambiguous_depths_test.py -coding Fourier -rec ncc -freq_idx 1

	Fourier-Domain Testing Commands (That should fail, after running with random domains multiple times):
		run tests/coding_unambiguous_depths_test.py -coding Fourier -rec ifft -freq_idx 2
		run tests/coding_unambiguous_depths_test.py -coding Fourier -rec ifft -freq_idx 3
		run tests/coding_unambiguous_depths_test.py -coding Fourier -rec ifft -freq_idx 4
		run tests/coding_unambiguous_depths_test.py -coding Fourier -rec ifft -freq_idx 2 4

	Truncated Fourier-Domain Testing Commands (That should produce 0 error):
		run tests/coding_unambiguous_depths_test.py -coding TruncatedFourier -rec ncc -n_freqs 1 --account_irf

	Gray Codes Testing Commands:
		run tests/coding_unambiguous_depths_test.py -coding Gray -rec ncc --account_irf -n_bits 4
		run tests/coding_unambiguous_depths_test.py -coding Gray -rec ncc --account_irf -n_bits 4 -pw_factors 20 
		run tests/coding_unambiguous_depths_test.py -coding Gray -rec ncc --account_irf -n_bits 4 -pw_factors 20 -irf_fpath ../scan_data_scripts/system_irf/unimodal-irf_tres-8ps_tlen-9152ps.npy

	Hamiltonian Codes Testing Commands (That should almost 0 error, there is a small bug in Ham Codes generation):
		run tests/coding_unambiguous_depths_test.py -coding HamiltonianK3 -rec zncc
		run tests/coding_unambiguous_depths_test.py -coding HamiltonianK4 -rec zncc
		run tests/coding_unambiguous_depths_test.py -coding HamiltonianK5 -rec zncc

	Timestamps Coding (That should almost 0 error, there is a small bug in Ham Codes generation):
		run tests/coding_unambiguous_depths_test.py -coding Timestamp -rec linear -n_timestamps 16

	Identity Coding (That should almost 0 error, there is a small bug in Ham Codes generation):
		run tests/coding_unambiguous_depths_test.py -coding Identity -rec linear
		run tests/coding_unambiguous_depths_test.py -coding Identity -rec matchfilt

	If you want to test some of the functionality of torch_coding simply add the flag --torch_coding
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
from research_utils.timer import Timer
from research_utils import plot_utils, np_utils, io_ops
from research_utils.signalproc_ops import standardize_signal
from research_utils.shared_constants import *
from toflib import tirf, coding, coding_ecc, coding_utils
from toflib import torch_coding
from toflib import tof_utils, input_args_utils

def calc_decoding_errors(coding_obj, simulated_pulses, rec_algo, gt_shifts):
	# breakpoint()
	if(isinstance(coding_obj, coding.Coding) or isinstance(coding_obj, coding_ecc.HammingCoding) ):
		# If numpy coding
		c_vals = coding_obj.encode(simulated_pulses)
		lookup = coding_obj.reconstruction(c_vals, rec_algo_id=rec_algo)
		decoded_shifts = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo)
	else:
		# If pytorch coding
		c_vals = coding_obj.encode(simulated_pulses).detach().cpu().numpy()
		lookup = coding_obj.reconstruction(c_vals, rec_algo_id=rec_algo).detach().cpu().numpy()
		decoded_shifts = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo).detach().cpu().numpy()
		# decoded_shifts = coding_obj.softmax_peak_decoding(c_vals, rec_algo_id=rec_algo).cpu().numpy()
	errors = 100.*np.abs(decoded_shifts - gt_shifts) / float(coding_obj.n_maxres)
	errors_median = np.median(errors)
	errors_max = np.max(errors)
	return (errors, errors_median, errors_max, lookup)

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for coding_unambiguous_depths_test.')
	parser = input_args_utils.add_eval_coding_args(parser)
	parser.add_argument('--torch_coding', action='store_true', help='Use TorchCoding class to eval scheme')
	args = parser.parse_args()
	colors = plot_utils.get_color_cycle()
	# Set tbins to a random number. 
	# Depending on the sampling ambiguous coding can appear to be unambiguous, so all coding schemes need to be tested at different tbins
	n_tbins = np.random.randint(low=512, high=1500)
	# n_tbins = 581
	# n_tbins = 512
	# Get coding ids and reconstruction algos and verify their lengths
	coding_ids = args.coding
	rec_algos_ids = args.rec
	pw_factors = np_utils.to_nparray(args.pw_factors)
	n_coding_schemes = len(coding_ids)
	# If only one rec algo is given, use that same algo for all coding
	if(len(rec_algos_ids) == 1): rec_algos_ids = [rec_algos_ids[0]]*n_coding_schemes
	# If only one pulse width is given, use that same pulse width for all coding
	if(len(pw_factors) == 1): pw_factors = np_utils.to_nparray([pw_factors[0]]*n_coding_schemes)
	# pair all coding and rec algos
	coding_scheme_ids = ['{}-{}-pw-{}'.format(coding_ids[i], rec_algos_ids[i], pw_factors[i]) for i in range(n_coding_schemes) ]
	# Verify input arguments and issue warnings
	assert(len(set(coding_scheme_ids)) == len(coding_scheme_ids)), "Input coding ids need to be unique. Current script does not support simulating the same coding with different parameters in a single run"

	## Create IRF
	if(args.irf_fpath is None):
		irf_obj = tirf.GaussianTIRF(n_tbins, mu=0, sigma=pw_factors[0])
		irf = irf_obj.tirf
	else:
		irf_data = np.load(args.irf_fpath)
		irf_fit = np_utils.circular_signal_fit(irf_data)
		t = np.arange(0, n_tbins) * (1./n_tbins)
		irf = irf_fit(t)
		irf[irf < 0] = 0.
		irf /= irf.sum()

	# initialize coding strategies
	coding_list = []
	for i in range(n_coding_schemes):
		curr_coding_id = coding_ids[i]
		if(curr_coding_id == 'Hamming'):
			curr_coding = coding_ecc.HammingCoding(n_tbins, 5, h_irf=irf)
		else:
			curr_coding = coding_utils.create_coding_obj(curr_coding_id, n_tbins, args, h_irf=irf)
		if(args.torch_coding):
			curr_coding = torch_coding.CodingLayer(C=curr_coding.C, h_irf=irf)
		coding_list.append(curr_coding)

	## Set ground truth depth and time shifts
	gt_shifts = np.arange(5, n_tbins - 5)
	n_pulses = gt_shifts.size
	## Generate all pulses by shifting the IRF
	pulses = np.zeros((n_pulses, n_tbins))
	for i in range(n_pulses):
		pulses[i,:] = np.roll(irf, gt_shifts[i])
	## Create a tirf object
	pulses_obj = tirf.TemporalIRF(pulses)
	
	plt.clf()
	fig = plt.gcf()
	ax1_1=fig.add_subplot(3,2,1)
	ax1_2=fig.add_subplot(3,2,2)
	ax2_1=fig.add_subplot(3,2,3)
	ax2_2=fig.add_subplot(3,2,4)
	ax3_1=fig.add_subplot(3,2,5)
	ax3_2=fig.add_subplot(3,2,6)

	for k in range(n_coding_schemes):
		print('-------------------------------------------------------------------------------')
		coding_obj = coding_list[k]
		rec_algo = rec_algos_ids[k]
		ID = coding_scheme_ids[k] + '_ncodes-{}'.format(int(coding_obj.n_codes))
		print('Evaluating {} Coding - {}'.format(coding_ids[k], ID))

		img = ax1_1.imshow(coding_obj.get_pretty_C(), cmap='gray', vmin=-1, vmax=1)
		ax1_1.set_title("Coding Mat | " + ID, fontsize=13)
		plot_utils.remove_ticks(ax1_1)

		img = ax1_2.imshow(coding_obj.get_pretty_decoding_C(), cmap='gray', vmin=-1, vmax=1)
		ax1_2.set_title("Decoding Mat | " + ID, fontsize=13)
		plot_utils.remove_ticks(ax1_2)

		# calc errors with pulses with very little noise
		pulses_obj.set_sbr(1.0)
		# simulated_pulses = pulses_obj.simulate_n_signal_photons(n_photons=5000).squeeze()
		simulated_pulses = pulses_obj.simulate_n_photons(n_photons=5000).squeeze()
		(errors, errors_median, errors_max, rec_lookup) = calc_decoding_errors(coding_obj, simulated_pulses, rec_algo, gt_shifts)
		print('Shift Errors (VERY High SNR) \n	Median Error = {:.2f} % \n	Max Error = {:.2f} %'.format(errors_median, errors_max)) 

		ax2_1.plot(errors, label=ID)
		ax2_1.set_title("Shift Errors (VERY High SNR)", fontsize=13)
		ax2_1.legend(fontsize=13)
		ax2_1.set_ylabel('Percent Error (Error / Range)', fontsize=13)
		ax2_1.set_ylim([-1, 101])
		n_rand_pulses = 2
		rand_indeces = np.random.randint(0, simulated_pulses.shape[0]-1, size=(n_rand_pulses,))
		for idx in range(n_rand_pulses):
			ax2_2.plot(standardize_signal(simulated_pulses[rand_indeces[idx], :]), color=colors[idx])
			ax2_2.plot(standardize_signal(rec_lookup[rand_indeces[idx], :]), '--*', color=colors[idx], alpha=0.5, linewidth=2)
		ax2_2.set_title("Example Test Pulses & Lookup Tables", fontsize=13)
		
		# generate random scaling factor and offsets
		scaling_factors = np.random.rand(n_pulses)*np.random.randint(100) + 0.1 # Make sure there is a small signal at least
		constant_offsets = np.random.rand(n_pulses)*np.random.randint(100) 
		# calc errors with pulses with random offsets
		simulated_pulses = (np.multiply(pulses, scaling_factors[..., np.newaxis])) + constant_offsets[..., np.newaxis] 
		(errors, errors_median, errors_max, rec_lookup) = calc_decoding_errors(coding_obj, simulated_pulses, rec_algo, gt_shifts)
		print('Shift Errors (Random Scaling & Offset, No Noise) \n	Median Error = {:.2f} % \n	Max Error = {:.2f} %'.format(errors_median, errors_max)) 

		ax3_1.plot(errors, label=ID)
		ax3_1.set_title("Errors (Random Scaling and Offsets, No Noise)", fontsize=13)
		ax3_1.legend(fontsize=13)
		ax3_1.set_ylabel('Percent Error (Error / Range)', fontsize=13)
		ax3_1.set_ylim([-1, 101])
		rand_indeces = np.random.randint(n_pulses, size=(n_rand_pulses,))
		for idx in range(n_rand_pulses):
			ax3_2.plot(simulated_pulses[rand_indeces[idx], :], color=colors[idx])
			ax3_2.plot(rec_lookup[rand_indeces[idx], :], '--*', color=colors[idx], alpha=0.5, linewidth=2)
		ax3_2.set_title("Example Test Pulses & Lookup Tables", fontsize=13)

