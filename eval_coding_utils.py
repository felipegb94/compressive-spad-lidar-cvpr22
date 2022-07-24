#### Standard Library Imports
import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from utils.input_args_parser import add_flash_lidar_scene_args
from toflib.input_args_utils import add_eval_coding_args
from toflib import tof_utils, tirf, tirf_scene, coding
from research_utils import plot_utils, np_utils, io_ops, improc_ops

def compose_coding_params_str(coding_id, n_codes, rec_algo, pw_factor, account_irf=False):
	if(account_irf): 
		return '{}_ncodes-{}_rec-{}-irf_pw-{:.1f}'.format(coding_id, n_codes, rec_algo, pw_factor)
	else:
		return '{}_ncodes-{}_rec-{}_pw-{:.1f}'.format(coding_id, n_codes, rec_algo, pw_factor)

def parse_signalandsbr_params(args):
	## Set nphotons, signal and sbr levels at which the MAE will be calculated at
	# If we input a single nphotons/sbr level then we use that
	if(not (args.sbr is None)): sbr_levels = np_utils.to_nparray(args.sbr)
	elif(args.min_max_sbr_exp is None): sbr_levels = np_utils.to_nparray(10.)
	else:
		(min_sbr_exp, max_sbr_exp) = (args.min_max_sbr_exp[0], args.min_max_sbr_exp[1])
		sbr_levels = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, args.n_sbr_lvls))
	if(not (args.signal is None)): signal_levels = np_utils.to_nparray(args.signal)
	elif(args.min_max_signal_exp is None): signal_levels = np_utils.to_nparray(1000.)
	else:
		(min_signal_exp, max_signal_exp) = (args.min_max_signal_exp[0], args.min_max_signal_exp[1])
		signal_levels = np.power(10, np.linspace(min_signal_exp, max_signal_exp, args.n_signal_lvls))
	if(not (args.nphotons is None)): nphotons_levels = np_utils.to_nparray(args.nphotons)
	elif(args.min_max_nphotons_exp is None): nphotons_levels = np_utils.to_nparray(1000.)
	else:
		(min_nphotons_exp, max_nphotons_exp) = (args.min_max_nphotons_exp[0], args.min_max_nphotons_exp[1])
		nphotons_levels = np.power(10, np.linspace(min_nphotons_exp, max_nphotons_exp, args.n_nphotons_lvls))
	assert(np.all(sbr_levels > 0)), "All sbr levels should be > 0"
	assert(np.all(signal_levels > 0)), "All signal_levels levels should be > 0"
	assert(np.all(nphotons_levels > 0)), "All signal_levels levels should be > 0"
	return (signal_levels, sbr_levels, nphotons_levels)

def generate_coding_scheme_ids(coding_ids, rec_algos_ids, pw_factors):
	n_coding_schemes = len(coding_ids)
	# If only one rec algo is given, use that same algo for all coding
	if(len(rec_algos_ids) == 1): rec_algos_ids = [rec_algos_ids[0]]*n_coding_schemes
	# If only one pulse width is given, use that same pulse width for all coding
	if(len(pw_factors) == 1): pw_factors = np_utils.to_nparray([pw_factors[0]]*n_coding_schemes)
	# pair all coding and rec algos
	coding_scheme_ids = ['{}-{}-pw-{}'.format(coding_ids[i], rec_algos_ids[i], pw_factors[i]) for i in range(n_coding_schemes) ]
	assert(len(set(coding_scheme_ids)) == len(coding_scheme_ids)), "Input coding ids need to be unique. Current script does not support simulating the same coding with different parameters in a single run"
	return (coding_scheme_ids, rec_algos_ids, pw_factors)

def decode_peak(coding_obj, c_vals, coding_id, rec_algo, pw_factor):
	'''
		Decoding peak assume gaussian for Gated, Identity and Timestamp coding
	'''
	print("Decoding peak.. Assuming Gaussian for Gated, Identity and Timestamp")
	if((coding_id == 'Gated') or (coding_id == 'Identity') or (coding_id == 'Timestamp')):
		# print('MaxGauss: {}'.format(pw_factor))
		decoded_depths = coding_obj.maxgauss_peak_decoding(c_vals, gauss_sigma=pw_factor, rec_algo_id=rec_algo)
		# decoded_depths = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo)
	elif(coding_id == 'SingleFourier'):
		decoded_depths = coding_obj.circmean_decoding(c_vals)
	else:
		decoded_depths = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo)
	return decoded_depths
