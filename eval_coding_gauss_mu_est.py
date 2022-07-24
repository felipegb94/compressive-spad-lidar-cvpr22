'''
Description:
	This script evaluates the MAE for a given coding scheme through a range of SBR and nphotons levels. To compute the MAE it runs a monte carlo simulate where many histograms are simulated for a single configuration of hyperparameters, and the depth errors are computed and aggregated.

	The simulate assumes that the laser pulse is a Gaussian pulse with a pulse width that is specified with the `pw_factors` parameter

	For a description of all the input parameters run `python eval_coding_gauss_mu_est.py --help`


Example Commands to generate Gaussian Pulse Results:

	- TruncatedFourierCoding (for paper)
	python eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 500 \
		-min_max_sbr_exp -2 0 -min_max_nphotons_exp 2 4 -n_sbr_lvls 12 -n_nphotons_lvls 12  \
		-coding TruncatedFourier -n_freqs 4 --account_irf --save_results

	- Gray Coding
	run eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 500 -min_max_sbr_exp -2 0 -min_max_nphotons_exp 2 4 -n_sbr_lvls 12 -n_nphotons_lvls 12  -coding Gray -n_bits 8 --account_irf --save_results

	- Full Resolution Histogram:
	run eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 500 -min_max_sbr_exp -2 0 -min_max_nphotons_exp 2 4 -n_sbr_lvls 6 -n_nphotons_lvls 6  -coding Identity --account_irf --save_results -rec matchfilt

	- GatedCoding (Narrow Pulse - Quantization Limited)
	run eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 500 -min_max_sbr_exp 0 1 -n_sbr_lvls 4 -min_max_nphotons_exp 1 2 -n_nphotons_lvls 3 -coding Gated -pw_factors 1 -rec linear -n_gates 16

	- GatedCoding (Narrow Pulse - Quantization Limited)
	run eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 500 -min_max_sbr_exp 0 1 -n_sbr_lvls 4 -min_max_nphotons_exp 1 2 -n_nphotons_lvls 3 -coding Gated -pw_factors 16 -rec linear -n_gates 16

	- Timestamp Codinng
	run eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 1000 -min_max_sbr_exp -1 1 -min_max_nphotons_exp 0 2 -n_sbr_lvls 5 -n_nphotons_lvls 5 -coding Timestamp -rec linear 
	run eval_coding_gauss_mu_est.py -n_tbins 1024 -n_depths 32 -n_mc_samples 500 -nphotons 100 -sbr 1.0 -coding Timestamp -rec matchfilt -n_timestamps 32

'''

#### Standard Library Imports
import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from toflib.input_args_utils import add_eval_coding_args, add_tbins_arg
from toflib import tof_utils, tirf, coding, coding_utils
from research_utils import plot_utils, np_utils, io_ops, timer
import eval_coding_utils

def compose_fname(coding_id, n_codes, rec_algo, pw_factor):
	return eval_coding_utils.compose_coding_params_str(coding_id, n_codes, rec_algo, pw_factor)+'.npz'

def get_out_rel_dirpath(n_tbins, sbr_levels, nphotons_levels):
	return 'final_coding_gauss_mu_est/ntbins-{}_logsbr-{:.1f}-{:.1f}_lognp-{:.1f}-{:.1f}'.format(n_tbins, np.log10(sbr_levels[0]), np.log10(sbr_levels[-1]), np.log10(nphotons_levels[0]), np.log10(nphotons_levels[-1]))

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for flash lidar simulation.')
	add_tbins_arg(parser)
	add_eval_coding_args(parser)
	parser.add_argument('--save_results', default=False, action='store_true', help='Save results.')
	args = parser.parse_args()
	# Parse input args
	n_tbins = args.n_tbins
	n_mc_samples = args.n_mc_samples
	colors = plot_utils.get_color_cycle()

	# Get dirpaths for data
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')

	## Set rep frequency depending on the domain of the simulated transient
	(_, _, tbin_res, t_domain, _, _) = tof_utils.calc_tof_domain_params(n_tbins, rep_tau=n_tbins)

	## Get coding ids and reconstruction algos and verify their lengths
	coding_ids = args.coding
	rec_algos_ids = args.rec
	pw_factors = np_utils.to_nparray(args.pw_factors)
	n_coding_schemes = len(coding_ids)
	(coding_scheme_ids, rec_algos_ids, pw_factors) = eval_coding_utils.generate_coding_scheme_ids(coding_ids, rec_algos_ids, pw_factors)

	## Set signal and sbr levels at which the MAE will be calculated at
	(_, sbr_levels, nphotons_levels) = eval_coding_utils.parse_signalandsbr_params(args)
	(X_sbr_levels, Y_nphotons_levels) = np.meshgrid(sbr_levels, nphotons_levels)
	n_nphotons_lvls = len(nphotons_levels)
	n_sbr_lvls = len(sbr_levels)

	## Create GT gaussian pulses for each coding. Different coding may use different pulse widths
	n_shifts = args.n_depths
	padding = 0.02 # Skip the time bins at the boundaries
	gt_shifts = np.linspace(padding*n_tbins, t_domain[-1] - (padding*n_tbins), n_shifts)
	pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors*tbin_res, mu=gt_shifts, t_domain=t_domain)

	## Create irf list
	irf_mus = np.zeros_like(pw_factors)
	irf_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors*tbin_res, mu=irf_mus, t_domain=t_domain)

	## initialize coding strategies
	coding_list = coding_utils.init_coding_list(coding_ids, n_tbins, args, irf_list)
	## For each coding scheme, signal/sbr lvl combination, compute shifts and shift errros
	# plt.clf()
	for i in range(n_coding_schemes):
		coding_id = coding_ids[i]
		pw_factor = pw_factors[i]
		rec_algo = rec_algos_ids[i]
		pulses = pulses_list[i]
		coding_obj = coding_list[i]
		n_codes = coding_obj.n_codes
		if(isinstance(coding_obj, coding.TimestampCoding)): n_codes = args.n_timestamps
		## Initialize arrays to store computed error metrics
		metric_mae = np.zeros_like(X_sbr_levels)
		metric_medae = np.zeros_like(X_sbr_levels)
		metric_1_tol_errs = np.zeros_like(X_sbr_levels)
		metric_percentile_mae = np.zeros(X_sbr_levels.shape + (4,)).astype(X_sbr_levels.dtype)
		for j in range(n_nphotons_lvls):
			for k in range(n_sbr_lvls):
				## Simulate a dtof image
				curr_sbr = X_sbr_levels[j, k]
				curr_nphotons = Y_nphotons_levels[j, k]
				# Set SBR
				pulses.set_sbr(curr_sbr)
				# simulate
				with timer.Timer("Simulate pulses"):
					simulated_pulses = pulses.simulate_n_photons(n_photons=curr_nphotons, n_mc_samples=n_mc_samples)
				# Check for pulses with 0 photons, to not include them in error calc.
				nosignal_mask = simulated_pulses.sum(axis=-1) < 1.
				validsignal_mask = np.logical_not(nosignal_mask)
				# If more than 75 percent had no signal, throw error (this should never happen for the photon levels we study)
				assert(nosignal_mask.sum() <= 0.75*nosignal_mask.size), "Not enough pixels with a signal larger than 1"
				# If more than 50 percent had no signal, simulate again and if it happens again throw error (this should never happen)
				# more than 50 percent pulses with no signal is very rare for the flux levels we are interested in
				if(nosignal_mask.sum() > 0.5*nosignal_mask.size):
					print("WARNING: More than 50 percent of pulses have no signal, simulating again")
					print("percent pulses with no signal = {}".format(nosignal_mask.sum() / nosignal_mask.size))
					simulated_pulses = pulses.simulate_n_photons(n_photons=curr_nphotons, n_mc_samples=n_mc_samples)
					nosignal_mask = simulated_pulses.sum(axis=-1) < 1.
					validsignal_mask = np.logical_not(nosignal_mask)
					assert(nosignal_mask.sum() <= 0.5*nosignal_mask.size), "Not enough pixels with a signal larger than 1"
				## Encode
				with timer.Timer("Encoding"):
					c_vals = coding_obj.encode(simulated_pulses)
				# Estimate gauss pulse shifts
				with timer.Timer("decoding"):
					decoded_shifts = eval_coding_utils.decode_peak(coding_obj, c_vals, coding_id, rec_algo, pw_factor)
				## Calc error metrics
				errors = decoded_shifts.squeeze() - gt_shifts
				# mask errors for simulated pixels that had 0 photons detected
				errors_masked = errors[validsignal_mask]
				abs_error_metrics = np_utils.calc_error_metrics(np.abs(errors_masked), delta_eps = 1.)
				## Store error metrics
				metric_mae[j,k] = abs_error_metrics['mae']
				metric_medae[j,k] = abs_error_metrics['medae']
				metric_1_tol_errs[j,k] = abs_error_metrics['1_tol_errs']
				metric_percentile_mae[j,k] = abs_error_metrics['percentile_mae']
				print("Photon Level = {} photons, SBR = {}".format(curr_nphotons, curr_sbr))
				np_utils.print_error_metrics(abs_error_metrics, '    ')
				plt.hist(errors.flatten(),bins=200,range=(-50,50), alpha=0.6)
		## Save computed errors metrics
		if(args.save_results):
			rel_dirpath = get_out_rel_dirpath(n_tbins, sbr_levels, nphotons_levels)
			out_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_data'], rel_dirpath)
			os.makedirs(out_dirpath, exist_ok=True)
			if(args.account_irf):
				out_fname = compose_fname(coding_id, n_codes, rec_algo+'-irf', pw_factor)
			else:
				out_fname = compose_fname(coding_id, n_codes, rec_algo, pw_factor)
			out_fpath = os.path.join(out_dirpath, out_fname)
			print(out_fpath)
			np.savez(out_fpath 
							, X_sbr_levels=X_sbr_levels
							, Y_nphotons_levels=Y_nphotons_levels
							, sbr_levels=sbr_levels
							, nphotons_levels=nphotons_levels
							, metric_mae=metric_mae
							, metric_medae=metric_medae
							, metric_1_tol_errs=metric_1_tol_errs
							, metric_percentile_mae=metric_percentile_mae
							, percentiles=abs_error_metrics['percentiles']
							, n_codes=n_codes
							, coding_id=coding_id
							, rec_algo=rec_algo
							, pw_factor=pw_factor
							, n_mc_samples=n_mc_samples
							, n_shifts_per_snr=n_shifts
							, n_tbins=n_tbins
							, coding_mat=coding_obj.C
							, decoding_mat=coding_obj.decoding_C
			)
			# save as matlab file too
			from research_utils.scipy_utils import npz2mat
			npz2mat(out_fpath)