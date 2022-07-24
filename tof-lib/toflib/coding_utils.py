'''
	Base class for temporal coding schemes
'''
## Standard Library Imports

## Library Imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.shared_constants import *
from toflib.coding import *
from toflib.coding_ecc import *


def init_coding_list(coding_ids, n_tbins, args, pulses_list=None):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''
	coding_list = []
	for i in range(len(coding_ids)):
		curr_coding_id = coding_ids[i]
		h_irf = None
		if(not (pulses_list is None)): h_irf = pulses_list[i].tirf.squeeze()
		curr_coding = create_coding_obj(curr_coding_id, n_tbins, args, h_irf)
		coding_list.append(curr_coding)
	return coding_list

def create_coding_obj(coding_id, n_tbins, args, h_irf=None):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''
	coding_obj = None
	if(coding_id == 'TruncatedFourier'):
		coding_obj = TruncatedFourierCoding(n_tbins, n_freqs=args.n_freqs, include_zeroth_harmonic=False, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesGrayBasedFourier'):
		coding_obj = PSeriesGrayBasedFourierCoding(n_tbins, n_codes=args.n_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'HybridGrayBasedFourier'):
		coding_obj = HybridGrayBasedFourierCoding(n_tbins, n_codes=args.n_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'RandomFourier'):
		coding_obj = RandomFourierCoding(n_tbins, n_rand_freqs=args.n_rand_freqs, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'HighFreqFourier'):
		coding_obj = HighFreqFourierCoding(n_tbins, n_high_freqs=args.n_high_freqs, start_high_freq=args.start_high_freq, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesFourier'):
		coding_obj = PSeriesFourierCoding(n_tbins, n_freqs=args.n_freqs, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesBinaryFourier'):
		coding_obj = PSeriesBinaryFourierCoding(n_tbins, n_codes=args.n_pserbinfourier_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Fourier'):
		coding_obj = FourierCoding(n_tbins, freq_idx=args.freq_idx, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'SingleFourier'):
		coding_obj = SingleFourierCoding(n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'KTapSinusoid'):
		coding_obj = KTapSinusoidCoding(n_maxres=n_tbins, freq_idx=args.freq_idx, k=args.ktaps, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Hamiltonian'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=4, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'HamiltonianK3'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=3, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'HamiltonianK4'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=4, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'HamiltonianK5'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=5, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Hamming'):
		coding_obj = HammingCoding(n_maxres=n_tbins, n_parity_bits=args.n_parity_bits, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'WalshHadamard'):
		coding_obj = WalshHadamardCoding(n_maxres=n_tbins, n_codes=args.n_wh_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Gray'):
		coding_obj = GrayCoding(n_maxres=n_tbins, n_bits=args.n_bits, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesGray'):
		coding_obj = PSeriesGrayCoding(n_maxres=n_tbins, n_codes=args.n_psergray_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Random'):
		coding_obj = RandomCoding(n_maxres=n_tbins, n_codes=args.n_random_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'GrayTruncatedFourier'):
		coding_obj = GrayTruncatedFourierCoding(n_maxres=n_tbins, n_codes=args.n_grayfourier_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'GrayTruncatedFourierV2'):
		coding_obj = GrayTruncatedFourierV2Coding(n_maxres=n_tbins, n_codes=args.n_grayfourier_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'GrayEquispaced3Fourier'):
		coding_obj = GrayEquispaced3FourierCoding(n_maxres=n_tbins, n_codes=args.n_grayfourier_codes, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'ReflectedGray'):
		coding_obj = ReflectedGrayCoding(n_maxres=n_tbins, n_bits=(args.n_bits//2), account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Haar'):
		coding_obj = HaarCoding(n_maxres=n_tbins, n_lvls=args.n_lvls, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'TruncatedKTapTriangle'):
		coding_obj = TruncatedKTapTriangleCoding(n_tbins, n_freqs=args.n_tri_freqs, include_zeroth_harmonic=False, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'KTapTriangle'):
		coding_obj = KTapTriangleCoding(n_tbins, freq_idx=args.tri_freq_idx, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Gated'):
		coding_obj = GatedCoding(n_maxres=n_tbins, n_gates=args.n_gates, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'GatedFourier'):
		coding_obj = GatedFourierCoding(n_maxres=n_tbins, n_gates=args.n_gates, freq_idx=args.freq_idx, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'GatedFourier-F-1'):
		coding_obj = GatedFourierCoding(n_maxres=n_tbins, n_gates=args.n_gates, freq_idx=[1], account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'GatedFourier-F-1-10'):
		coding_obj = GatedFourierCoding(n_maxres=n_tbins, n_gates=args.n_gates, freq_idx=[1, 10], account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Identity'):
		coding_obj = IdentityCoding(n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'Timestamp'):
		coding_obj = TimestampCoding(n_maxres=n_tbins, n_timestamps=args.n_timestamps, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PCA'):
		pca_data = np.load(args.pca_fpath).transpose()[:, 0:args.n_pca_codes]
		assert(pca_data.shape[0] == n_tbins), "pca_data is only implemented for 1024 tbins for now"
		coding_obj = DataCoding(C=pca_data, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PCAOnGT'):
		pca_data = np.load(args.pca_fpath).transpose()[:, 0:args.n_pca_codes]
		assert(pca_data.shape[0] == n_tbins), "pca_data is only implemented for 1024 tbins for now"
		coding_obj = DataCoding(C=pca_data, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PCAFlux10000'):
		pca_data = np.load(args.pca_fpath).transpose()[:, 0:args.n_pca_codes]
		assert(pca_data.shape[0] == n_tbins), "pca_data is only implemented for 1024 tbins for now"
		coding_obj = DataCoding(C=pca_data, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'PCACDFFlux10000'):
		pca_data = np.load(args.pca_fpath).transpose()[:, 0:args.n_pca_codes]
		assert(pca_data.shape[0] == n_tbins), "pca_data is only implemented for 1024 tbins for now"
		coding_obj = DataCoding(C=pca_data, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'HamK4'):
		hamk4_data = np.load('sample_data/sample_corrfs/keatof_2MHz_ham-k4_min.npy').transpose()
		coding_obj = DataCoding(C=hamk4_data, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'OptCL1'):
		opt_codes_fname = 'opt_C_loss-l1_ncodes-{}_ntbins-{}_signal-0-2_sbr--1-1'.format(args.n_codes_l1, n_tbins)
		opt_codes = np.load(os.path.join(args.opt_codes_dirpath, '{}.npy'.format(opt_codes_fname)))
		coding_obj = DataCoding(C=opt_codes, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'OptCL1-HFlux'):
		opt_codes_fname = 'opt_C_loss-l1_ncodes-{}_ntbins-{}_signal-1.0-2.0_sbr--2.0-0.0'.format(args.n_codes_l1, n_tbins)
		opt_codes = np.load(os.path.join(args.opt_codes_dirpath, '{}.npy'.format(opt_codes_fname)))
		coding_obj = DataCoding(C=opt_codes, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	elif(coding_id == 'OptC1Tol'):
		opt_codes_fname = 'opt_C_loss-1tol_ncodes-{}_ntbins-{}_signal-0-2_sbr--1-1'.format(args.n_codes_1tol, n_tbins)
		opt_codes = np.load(os.path.join(args.opt_codes_dirpath, '{}.npy'.format(opt_codes_fname)))
		coding_obj = DataCoding(C=opt_codes, n_maxres=n_tbins, account_irf=args.account_irf, h_irf=h_irf)
	# elif(coding_id == 'OptC'):
	# 	opt_codes = np.load('sample_data/sample_corrfs/optC_n-{}_k-8.npy'.format(n_tbins))
	# 	coding_obj = DataCoding(C=opt_codes, n_maxres=n_tbins)
	else: assert(False), "Invalid input coding for evaluation ({})".format(coding_id)
	return coding_obj

def create_basic_coding_obj(coding_id, n_tbins, n_codes, h_irf=None, account_irf=False):
	'''
		Only supports coding objects that can be easily created if we know the number of codes and no additional params are needed.
	'''
	coding_obj = None
	if(h_irf is None): account_irf = False
	if(coding_id == 'TruncatedFourier'):
		coding_obj = TruncatedFourierCoding(n_tbins, n_freqs=n_codes//2, include_zeroth_harmonic=False, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'HybridGrayBasedFourier'):
		coding_obj = HybridGrayBasedFourierCoding(n_tbins, n_codes=n_codes, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesGrayBasedFourier'):
		coding_obj = PSeriesGrayBasedFourierCoding(n_tbins, n_codes=n_codes, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'RandomFourier'):
		coding_obj = RandomFourierCoding(n_tbins, n_rand_freqs=n_codes//2, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesFourier'):
		coding_obj = PSeriesFourierCoding(n_tbins, n_freqs=n_codes//2, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesBinaryFourier'):
		coding_obj = PSeriesBinaryFourierCoding(n_tbins, n_codes=n_codes, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'Gray'):
		coding_obj = GrayCoding(n_maxres=n_tbins, n_bits=n_codes, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'PSeriesGray'):
		coding_obj = PSeriesGrayCoding(n_maxres=n_tbins, n_codes=n_codes, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'Gated'):
		coding_obj = GatedCoding(n_maxres=n_tbins, n_gates=n_codes, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'GatedFourier-F-1'):
		coding_obj = GatedFourierCoding(n_maxres=n_tbins, n_gates=n_codes//2, freq_idx=[1], account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'Identity'):
		coding_obj = IdentityCoding(n_maxres=n_tbins, account_irf=account_irf, h_irf=h_irf)
	elif(coding_id == 'Timestamp'):
		coding_obj = TimestampCoding(n_maxres=n_tbins, n_timestamps=n_codes, account_irf=account_irf, h_irf=h_irf)
	return coding_obj
