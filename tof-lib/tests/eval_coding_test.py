'''
	Test functionality of toflib.coding.py:DataCoding Class
	Sample Run command: 
run tests/eval_coding_test.py \
	-coding OptC TruncatedFourier Hamiltonian Gated Identity \
	-rec zncc ifft zncc linear linear \
	-n_freqs 3 \
	-n_tri_freqs 3 \
	-pw_factors 1 1 1 8 1 \
	-n_tbins 1024 \
	-n_signal_lvls 5 \
	-n_sbr_lvls 5 \
	-n_mc_samples 1000 \
	-min_max_sbr_exp 0 1 \
	-min_max_signal_exp 1.0 2.5 \
	-n_depths 20 \
	-rep_freq 1e7 -n_gates 128

run tests/eval_coding_test.py \
	-coding TruncatedFourier Gray OptC \
	-rec ifft zncc zncc \
	-n_freqs 4 \
	-n_bits 8 \
	-n_lvls 3 \
	-pw_factors 1 1 1 \
	-n_tbins 1024 \
	-n_signal_lvls 5 \
	-n_sbr_lvls 5 \
	-n_mc_samples 750 \
	-min_max_sbr_exp 0 1 \
	-min_max_signal_exp 1.25 2.0 \
	-n_depths 15 \
	-rep_freq 1e7 -n_gates 32
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
from toflib import coding
from toflib import tirf
from toflib import tof_utils, input_args_utils

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for eval_coding_test.')
	parser = input_args_utils.add_tbins_arg(parser)
	parser = input_args_utils.add_eval_coding_args(parser)
	args = parser.parse_args()
	# Parse input args
	n_tbins = args.n_tbins
	n_signal_lvls = args.n_signal_lvls
	n_sbr_lvls = args.n_sbr_lvls
	n_mc_samples = args.n_mc_samples
	n_depths = args.n_depths
	rep_freq = args.rep_freq # In Hz
	rep_tau = 1. / rep_freq
	max_depth = tof_utils.time2depth(rep_tau)
	depth_padding = 0.02 # Skip the depths at the boundaries
	colors = plot_utils.get_color_cycle()

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
	assert(len(set(coding_scheme_ids)) == len(coding_scheme_ids)), "Input coding ids need to be unique. Current script does not support simulating the same coding with different parameters in a single run"

	# Verify input arguments and issue warnings
	if(n_mc_samples < 1000): print("Warning: n_mc_samples < 1000 will lead to high variance MAE calculations")

	# Get number of time bins, their location, resoluon, and bounds
	(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils.calc_tof_domain_params(n_tbins, rep_tau=rep_tau)

	# Set signal, sbr, and depths at which the MAE will be calculated at
	# (min_signal_exp, max_signal_exp) = (0.5, 2.5)
	# (min_sbr_exp, max_sbr_exp) = (-1, 1)
	(min_signal_exp, max_signal_exp) = (args.min_max_signal_exp[0], args.min_max_signal_exp[1])
	(min_sbr_exp, max_sbr_exp) = (args.min_max_sbr_exp[0], args.min_max_sbr_exp[1])
	photon_levels = np.round(np.power(10, np.linspace(min_signal_exp, max_signal_exp, n_signal_lvls)))
	sbr_levels = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, n_sbr_lvls))
	assert(np.all(photon_levels >= 0)), "All photon levels should be >= 0"
	assert(np.all(sbr_levels > 0)), "All photon levels should be > 0"

	# Set ground truth depth and time shifts
	gt_depths = np.linspace(depth_padding*max_depth, max_depth-(depth_padding*max_depth), n_depths)
	gt_tshifts = tof_utils.depth2time(gt_depths)

	# initialize coding strategies
	coding_list = coding.init_coding_list(coding_ids, n_tbins, args)

	# Create GT gaussian pulses for each coding. Different coding may use different pulse widths
	pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors*tbin_res, mu=gt_tshifts, t_domain=t_domain)

	# plt.clf()
	(X, Y) = np.meshgrid(sbr_levels, photon_levels)
	for k in range(n_coding_schemes):
		print('Evaluating {} Coding'.format(coding_ids[k]))
		expected_depth_mae = np.zeros((n_signal_lvls, n_sbr_lvls))
		results_dict = {}
		results_dict['X_label'] = 'Signal-Background Ratio'
		results_dict['Y_label'] = 'N Signal Photons'
		results_dict['Z_label'] = 'Expected Depth MAE'
		results_dict['X'] = X.tolist()
		results_dict['Y'] = Y.tolist()
		results_dict['Z'] = np.zeros_like(X)
		pulses = pulses_list[k]
		coding_obj = coding_list[k]
		rec_algo = rec_algos_ids[k]
		results_dict['ID'] = coding_scheme_ids[k] + '_ncodes-{}'.format(int(coding_obj.n_codes))
		results_filename = 'mae_{}.json'.format(coding_scheme_ids[k]) 		
		for i in range(n_signal_lvls):
			for j in range(n_sbr_lvls):
				# curr_n_photons = photon_levels[i]
				# curr_sbr = sbr_levels[j]
				curr_sbr = X[i, j]
				curr_n_photons = Y[i, j]
				# Set SBR
				pulses.set_sbr(curr_sbr)
				# simulate
				simulated_pulses = pulses.simulate_n_signal_photons(n_photons=curr_n_photons, n_mc_samples=n_mc_samples)
				# Encode vals
				# with Timer("Encoding Time:"):
				c_vals = coding_obj.encode(simulated_pulses)
				# Estimate depths
				# with Timer("Decoding Time:"):
				if(coding_ids[k] == 'OptC'):
					decoded_depths = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo)*tbin_depth_res
				elif((coding_ids[k] == 'Gated') or (coding_ids[k] == 'Identity')):
					print('MaxGauss: {}'.format(pw_factors[k]))
					decoded_depths = coding_obj.maxgauss_peak_decoding(c_vals, gauss_sigma=pw_factors[k], rec_algo_id=rec_algo)*tbin_depth_res
				elif(coding_ids[k] == 'SingleFourier'):
					decoded_depths = coding_obj.circmean_decoding(c_vals)*tbin_depth_res
				else:
					decoded_depths = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo)*tbin_depth_res
				# Calc errors
				depth_errors = np.abs(decoded_depths - gt_depths[np.newaxis,:])*1000
				depthwise_mae = depth_errors.mean(axis=0).astype(int)
				mae = depthwise_mae.mean().astype(int)
				print("n_photons = {} | sbr = {} ".format(curr_n_photons, curr_sbr))
				print("    MAE = {}".format(mae))
				results_dict['Z'][i,j] = mae
		results_dict['Z'] = results_dict['Z'].tolist() 
		io_ops.write_json('./tmp/'+results_filename, results_dict)

		# plot surface plot
		fig = plt.gcf()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(np.log10(X), np.log10(Y), np.array(results_dict['Z']), label=results_dict['ID'], linewidth=1, facecolor=colors[k], alpha=0.7)
		surf._facecolors2d=surf._facecolors3d
		surf._edgecolors2d=surf._edgecolors3d
		ax.legend(fontsize=16)
		ax.set_xlabel('Log '+results_dict['X_label'])
		ax.set_ylabel('Log '+results_dict['Y_label'])
		# 		# if(i == 2 and j ==1):
		# 		# 	n_plots = np.min([3, n_depths])
		# 		# 	for k in range(n_depths):
		# 		# 		plt.subplot(n_plots,1,k+1)
		# 		# 		depth_idx = 0
		# 		# 		if(k==1): depth_idx = n_depths // 2
		# 		# 		if(k==2): depth_idx = n_depths -1
		# 		# 		plt.plot(simulated_pulses[0, depth_idx, :] / simulated_pulses[0, depth_idx, :].max(), label='data', alpha=0.5)
		# 		# 		ifft = fourier_coding.ifft_reconstruction(c_vals[0,depth_idx,:])
		# 		# 		mese = fourier_coding.mese_reconstruction(c_vals[0,depth_idx,:])
		# 		# 		piza = fourier_coding.pizarenko_reconstruction(c_vals[0,depth_idx,:])
		# 		# 		plt.plot(ifft / ifft.max(), label='ifft')
		# 		# 		plt.plot(mese / mese.max(), label='mese')
		# 		# 		plt.plot(piza / piza.max(), label='piza')
		# 		# 		plt.plot(gt_pulses[depth_idx, :] / gt_pulses[depth_idx, :].max(), label='GT')
		# 		# 		plt.legend()