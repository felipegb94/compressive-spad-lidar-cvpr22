'''
	Contains the code that puts together the parsers for the different scripts
'''
## Standard Library Imports
import argparse

## Library Imports
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports

def add_tbins_arg( parser ):
	parser.add_argument('-n_tbins', type=int, default=512, help='Number of time bins to use when generating the model-based tirf')
	return parser

def add_data_tirf_args( parser ):
	parser.add_argument('-data_dirpath', type=str, default='./sample_data', help='Dirpath of folder containing a scene_irf_data folder')
	parser.add_argument('-scene_id', type=str, default='vgroove', help='ID of the scene we are loading from scene_irf_data folder')
	return parser

def add_model_tirf_args( parser ):
	add_tbins_arg( parser )
	return parser

def add_tirf_scene_args( parser ):
	parser = add_data_tirf_args( parser )
	return parser

def add_data_coding_args( parser ):
	add_data_tirf_args(parser)
	parser.add_argument('-c_data_fpath', type=str, default='./sample_data/sample_corrfs/keatof_2MHz_ham-k3_min.npy', help='Filepath to numpy array with coding data')
	parser.add_argument('-rep_freq', type=float, default=15e6, help='Rep frequency. Needed to interpret the depth map correctly')
	return parser

def add_fourier_coding_args( parser ):
	add_tbins_arg( parser )
	add_coding_args(parser, coding_id='Fourier')
	return parser

def add_coding_args( parser, coding_id=None ):
	# If no coding Id is given add arguments for all coding
	add_all = (coding_id is None) 
	if(add_all): coding_id = ''
	parser.add_argument('--account_irf', default=False, action='store_true', help='Account for irf during decoding.')
	parser.add_argument('-n_codes', type=int, default=2, help='Number of codes to use (only some coding schemes use this param).')
	# Fourier and KTap Sinusoid coding args
	if(add_all or ('Fourier' in coding_id) or ('KTapSinusoid' in coding_id) ):
		parser.add_argument('-freq_idx', type=int, nargs="+", default=[0, 1], help='(Fourier Coding Args) Harmonics to include in the coding')
		parser.add_argument('-n_freqs', type=int, default=2, help='(Fourier Coding Args) Number of frequencies to sample. For TruncatedFourierCoding these are the first K frequencies')
		if(('KTapSinusoid' in coding_id) or add_all):
			# K-Tap Sinusoid Coding
			parser.add_argument('-ktaps', type=int, default=4, help='Number of phase shift per sinusoid')
	# RandomFourier Coding Arguments
	if(add_all or ('RandomFourier' == coding_id)):
		parser.add_argument('-n_rand_freqs', type=int, default=1, help='(RandomFourier Coding Args) Number of frequencies to sample')
	if(add_all or ('HighFreqFourier' == coding_id)):
		parser.add_argument('-n_high_freqs', type=int, default=1, help='(HighFreqFourier Coding Args) Number of frequencies to sample')
		parser.add_argument('-start_high_freq', type=int, default=40, help='(HighFreqFourier Coding Args) Frequency to start sampling at')
	# Gated Coding Arguments
	if(add_all or ('Gated' == coding_id)):
		parser.add_argument('-n_gates', type=int, default=None, help='(Gated Coding args) Number of gates')
	if(add_all or ('Timestamp' == coding_id)):
		parser.add_argument('-n_timestamps', type=int, default=4, help='(Timestamp Coding args) Number of timestamps to simulate')
	# Hamming Coding Arguments
	if(add_all or ('Hamming' == coding_id)):
		parser.add_argument('-n_parity_bits', type=int, default=None, help='(Hamming Coding args) Number of bits to use')
	# WalshHadamard Coding Arguments
	if(add_all or ('WalshHadamard' == coding_id)):
		parser.add_argument('-n_wh_codes', type=int, default=4, help='(WalshHadamard Coding args) Number of codes to use')
	# Gray Coding Arguments
	if(add_all or ('Gray' == coding_id)):
		parser.add_argument('-n_bits', type=int, default=2, help='(Gray Coding args) Number of bits to use')
	# Gray Coding Arguments
	if(add_all or ('PSeriesGray' == coding_id)):
		parser.add_argument('-n_psergray_codes', type=int, default=2, help='(PSeriesGray Coding args) Number of codes to use')
	# Gray Coding Arguments
	if(add_all or ('Random' == coding_id)):
		parser.add_argument('-n_random_codes', type=int, default=2, help='(Random Coding args) Number of codes to use')
	# PSeriesBinaryFourier Coding Arguments
	if(add_all or ('PSeriesBinaryFourier' == coding_id)):
		parser.add_argument('-n_pserbinfourier_codes', type=int, default=2, help='(PSeriesBinaryFourier Coding args) Number of codes to use')
	# GrayTruncatedFourier Coding Arguments
	if(add_all or ('GrayTruncatedFourier' == coding_id)):
		parser.add_argument('-n_grayfourier_codes', type=int, default=None, help='(GrayTruncatedFourier Coding args) Number of codes to use')
	elif(add_all or ('GrayTruncatedFourierV2' == coding_id)):
		parser.add_argument('-n_grayfourier_codes', type=int, default=None, help='(GrayTruncatedFourierV2 Coding args) Number of codes to use')
	elif(add_all or ('GrayEquispaced3Fourier' == coding_id)):
		parser.add_argument('-n_grayfourier_codes', type=int, default=None, help='(GrayEquispaced3Fourier Coding args) Number of codes to use')
	# Haar Coding Arguments
	if(add_all or ('Haar' == coding_id)):
		parser.add_argument('-n_lvls', type=int, default=None, help='(Haar Coding args) Number of levels in the Haar Coding Matrix')
	if(add_all or ('PCA' in coding_id)):
		parser.add_argument('-pca_fpath', type=str, default='./pca/masks/pca_mask_on_gt.npy', help='Filepath to pca codes')
		parser.add_argument('-n_pca_codes', type=int, default=2, help='Number of pca codes to use')
	if(add_all or ('OptC' in coding_id)):
		parser.add_argument('-opt_codes_dirpath', type=str, default='./tof-lib/opt_codes_data', help='Dirpath to opt codes data')
	# OptCL1 Coding Arguments
	if(add_all or ('OptCL1' == coding_id)):
		parser.add_argument('-n_codes_l1', type=int, default=None, help='(OptCL1 Coding args) Number of Codes')
	# OptC1Tol Coding Arguments
	if(add_all or ('OptC1Tol' == coding_id)):
		parser.add_argument('-n_codes_1tol', type=int, default=None, help='(OptC1Tol Coding args) Number of Codes')
	# Triangular Coding args
	if(add_all or ('Triangle' in coding_id) ):
		parser.add_argument('-tri_freq_idx', type=int, nargs="+", default=[0, 1], help='(Triangle Coding Args) Frequencies to include in the coding')
		parser.add_argument('-n_tri_freqs', type=int, default=2, help='(Triangle Coding Args)  Number of frequencies to sample. For TriangleFourierCoding these are the first K frequencies')
	return parser

def add_tofsim_args( parser ):
	parser.add_argument('-rep_freq', type=float, default=1e7, help='Repetition Frequency to use in Hz. Determines depth rang')
	# parser.add_argument('-sbr', type=float, default=10, help='SBR level to simulate (sometimes used instead of min_max_sbr_lvls)')
	# parser.add_argument('-n_photons', type=float, default=100, help='Number of signal photons (sometimes used instead of min_max_signal_lvls)')
	# if we only want to simulate a single flux, signal or sbr level
	parser.add_argument('-sbr', type=float, default=None, help='single sbr level to evaluate')
	parser.add_argument('-nphotons', type=float, default=None, help='single total photons level to evaluate')
	parser.add_argument('-signal', type=float, default=None, help='single signal level to evaluate')
	return parser

def add_eval_coding_args( parser ):
	# Add the arguments for all possible coding
	add_coding_args( parser )
	# Add coding scheme ID, reconstruction algo ID
	parser.add_argument('-coding', type=str, nargs="+", default=['Fourier'], help='ID of coding scheme. See coding.py for all possible ')
	parser.add_argument('-rec', type=str, nargs="+", default=['ncc'], help='ID of coding scheme. See coding.py for all possible ')
	# Add arguments for tofsimulation
	add_tofsim_args( parser )
	# Add args for gaussian pulse
	parser.add_argument('-pw_factors', type=float, nargs="+", default=[1.0], help='Pulse width factors to use for each coding scheme. If only one is given, the same pulse width is used for all')
	# IRF fpath
	parser.add_argument('-irf_fpath', type=str, default=None, help='Filepath to irf')
	# Add additional arg
	parser.add_argument('-n_mc_samples', type=int, default=100, help='Number of monte carlo simulations to perform per depth to determine error')
	parser.add_argument('-n_depths', type=int, default=3, help='Number of depths to simulate evenly spread across the depth range')
	# 
	parser.add_argument('-n_signal_lvls', type=int, default=1, help='Number of signal levels to simulate')
	parser.add_argument('-n_nphotons_lvls', type=int, default=1, help='Number of total nphotons levels to simulate')
	parser.add_argument('-n_sbr_lvls', type=int, default=1, help='Number of sbr levels to simulate')
	parser.add_argument('-min_max_sbr_exp', type=float, nargs="+", default=None, help='min and max sbr level to simulate')
	parser.add_argument('-min_max_signal_exp', type=float, nargs="+", default=None, help='min and max signal level to simulate')
	parser.add_argument('-min_max_nphotons_exp', type=float, nargs="+", default=None, help='min and max total nphotons level to simulate')
	return parser

def add_torch_code_opt_args( parser ):
	parser.add_argument('-n_opt_codes', type=int, default=4, help='Number of codes to optimize')
	# parser.add_argument('-n_samples', type=int, default=2048, help='Number of training samples to generate')
	parser.add_argument('-epochs', type=int, default=1000, help='Number of epochs')
	parser.add_argument('-loss_id', type=str, default='l1', help='Loss function to use: l1, 1tol, 0tol')
	parser.add_argument('-init_id', type=str, default='random', help='initialization id for the codes. options: random, truncfourier')
	add_eval_coding_args(parser)
	return parser