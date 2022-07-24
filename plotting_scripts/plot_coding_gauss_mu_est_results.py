## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils import plot_utils, np_utils, io_ops
from eval_coding_gauss_mu_est import compose_fname

def parse_results_data_fname(fname):
	fname_split = fname.split('_')
	coding_id = fname_split[0]
	n_codes = fname_split[1].split('-')[-1]
	rec_algo_id = fname_split[2].split('-')[-1]
	pw_factor = fname_split[3].split('-')[-1]
	return (coding_id, n_codes, rec_algo_id, pw_factor)

def finalize_plot(ax, title=None, ylabel=None):
	plot_utils.set_ticks(ax, fontsize=14)
	ax.grid(linestyle='--', alpha=0.5)
	plot_utils.set_legend(ax=ax, fontsize=14)
	# if(not (title is None)): ax.set_title(title, fontsize=14)
	# if(not (ylabel is None)): ax.set_ylabel(ylabel, fontsize=14)
	# ax.set_xlabel("Number of Codes", fontsize=14)

def compose_dict_key(coding_id, rec_algo):
	# return "{} Coding".format(coding_id)
	return "C: {}, Rec: {}".format(coding_id, rec_algo)

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	# plt.style.use('dark_background')

	n_tbins=1024
	n_codes_list=np.array([8, 16, 32, 64])
	n_code_configs = len(n_codes_list)
	coding_ids = ['TruncatedFourier', 'HybridGrayBasedFourier', 'PSeriesGrayBasedFourier']
	rec_algos = ['ncc-irf','ncc-irf','ncc-irf']
	# coding_ids = ['TruncatedFourier', 'PSeriesGray']
	# rec_algos = ['ncc-irf','ncc-irf']

	### OLD
	# coding_ids = ['TruncatedFourier', 'Gray', 'OptCL1', 'Timestamp', 'Gated', 'GatedWide']
	# rec_algos = ['ncc', 'ncc', 'ncc', 'matchfilt', 'linear', 'linear']
	# coding_ids = ['TruncatedFourier', 'Gray', 'Timestamp']
	# rec_algos = ['ncc', 'ncc', 'ncc', 'matchfilt']
	# coding_ids = ['TruncatedFourier', 'Gray', 'OptCL1', 'Timestamp']
	# rec_algos = ['ncc', 'ncc', 'ncc', 'matchfilt']
	# coding_ids = ['TruncatedFourier', 'Gray', 'Gated', 'GatedWide', 'Timestamp']
	# rec_algos = ['ncc', 'ncc', 'linear', 'linear', 'matchfilt']
	# coding_ids = ['GrayTruncatedFourier', 'TruncatedFourier','Gray', 'Gated', 'GatedWide', 'Timestamp']
	# rec_algos = ['ncc', 'ncc', 'ncc', 'linear', 'linear', 'matchfilt']
	# coding_ids = ['RandomFourier', 'GatedFourier-F-1', 'GatedFourier-F-1-10', 'GrayEquispaced3Fourier', 'GrayTruncatedFourier', 'TruncatedFourier','Gray', 'Gated', 'GatedWide']
	# rec_algos = ['ncc', 'ncc', 'ncc', 'ncc', 'ncc', 'ncc', 'ncc', 'linear', 'linear']
	# coding_ids = ['Hamming', 'TruncatedFourier', 'Gray', 'Gated', 'Timestamp']
	# rec_algos = ['ncc', 'ncc', 'ncc', 'linear', 'matchfilt']
	# coding_ids = ['Gray', 'TruncatedFourier', 'Gray', 'Gated', 'Timestamp']
	# rec_algos = ['ncc-irf', 'ncc', 'ncc', 'linear', 'matchfilt']
	# coding_ids = ['Gray', 'Gray',]
	# rec_algos = ['ncc-irf', 'ncc']
	# coding_ids = ['Gray', 'PCAOnGT']
	# coding_ids = ['TruncatedFourier','PCAOnGT','PCAFlux10000','PCACDFFlux10000']
	# rec_algos = ['ncc','ncc','ncc','ncc']

	# Set params
	min_max_sbr_levels = (0.01,1)
	min_max_photons_levels = (100,10000)
	sbr_level = 0.1 # 0.1, 1.29154967,3.59381366, 10.0
	nphotons_level = 10000.0 # 1, 5, 22, 100
	pw_factor = 1.0

	n_code_configs = len(n_codes_list)
	n_coding_configs = len(coding_ids)

	# Get dirpath from params
	from eval_coding_gauss_mu_est import get_out_rel_dirpath
	rel_dirpath = get_out_rel_dirpath(n_tbins, min_max_sbr_levels, min_max_photons_levels)
	# in_dirpath = os.path.join(io_dirpaths['results_data'], rel_dirpath)
	in_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_data'], rel_dirpath)

	# Init dict that will store the data
	results_dict = {}
	for i in range(n_coding_configs):
		coding_id = coding_ids[i]
		# coding_dict_key = '{}_rec-{}'.format(coding_id, rec_algos[i])
		coding_dict_key = compose_dict_key(coding_id, rec_algos[i])
		results_dict[coding_dict_key] = {}
		results_dict[coding_dict_key]['mae'] = np.zeros((n_code_configs,))
		results_dict[coding_dict_key]['medae'] = np.zeros((n_code_configs,))
		results_dict[coding_dict_key]['1_tol_errs'] = np.zeros((n_code_configs,))
		results_dict[coding_dict_key]['coding_mats'] = []
		results_dict[coding_dict_key]['decoding_mats'] = []

	for i in range(n_code_configs):
		n_codes = n_codes_list[i]
		# filenames = io_ops.get_filepaths_in_dir(in_dirpath, '*_ncodes-{}_*'.format(n_codes), only_filenames=True, keep_ext=False)
		# for fname in filenames:
		#     (coding_id, n_codes, rec_algo_id, pw_factor) = parse_results_data_fname(fname)
		#     if(coding_id in coding_ids):
		for j in range(n_coding_configs):
			coding_id = coding_ids[j]
			# If we are using gated with a wide pulse width use different pw_width
			if(coding_id == 'GatedWide'):
				fname = compose_fname('Gated', n_codes, rec_algos[j], pw_factor=n_tbins / n_codes)
			else:
				fname = compose_fname(coding_id, n_codes, rec_algos[j], pw_factor=pw_factor)
			fpath = os.path.join(in_dirpath, fname)
			# coding_dict_key = '{}_rec-{}'.format(coding_id, rec_algos[j])
			coding_dict_key = compose_dict_key(coding_id, rec_algos[j])
			# skip if result does not exist for this number of codes
			if(os.path.exists(fpath)):
				results_data = np.load(fpath)
				X_sbr_levels = results_data['X_sbr_levels']
				Y_nphotons_levels = results_data['Y_nphotons_levels']
				# Get index for the input signal levels
				sbr_mask = np.around(sbr_level, 2) == np.around(X_sbr_levels, 2)
				nphotons_mask = np.around(nphotons_level, 2) == np.around(Y_nphotons_levels, 2) 
				(row, col) = np.where(np.logical_and(sbr_mask, nphotons_mask))
				assert(len(row) > 0), "sbr or signal level not found"
				assert(len(row) == 1), "something went wrong. there should only by one matching sbr and signal lvl"
				(row, col) = (row[0], col[0])
				# Store result in dict
				results_dict[coding_dict_key]['mae'][i] = results_data['metric_mae'][row,col]
				results_dict[coding_dict_key]['medae'][i] = results_data['metric_medae'][row,col]
				results_dict[coding_dict_key]['1_tol_errs'][i] = results_data['metric_1_tol_errs'][row,col]
				results_dict[coding_dict_key]['coding_mats'].append(results_data['coding_mat'])
				results_dict[coding_dict_key]['decoding_mats'].append(results_data['decoding_mat'])
			else:
				print("WARNING: {} DOES NOT EXIST!".format(coding_dict_key))
				results_dict[coding_dict_key]['mae'][i] = np.nan
				results_dict[coding_dict_key]['medae'][i] = np.nan
				results_dict[coding_dict_key]['1_tol_errs'][i] = np.nan

	# Load identity coding results if available
	# fname_linear = compose_fname('Identity', n_tbins, 'linear', 1.0)
	# fpath_linear = os.path.join(in_dirpath, fname_linear)
	# if(os.path.exists(fpath_linear)):
	#     results_data = np.load(fpath_linear)
	#     coding_dict_key = 'Full Histogram (linear)'
	#     results_dict[coding_dict_key] = {}
	#     results_dict[coding_dict_key]['mae'] = results_data['metric_mae'][row,col]*np.ones((n_code_configs,))
	#     results_dict[coding_dict_key]['medae'] = results_data['metric_medae'][row,col]*np.ones((n_code_configs,))
	#     results_dict[coding_dict_key]['1_tol_errs'] = results_data['metric_1_tol_errs'][row,col]*np.ones((n_code_configs,))
	fname_matchfilt = compose_fname('Identity', n_tbins, 'matchfilt-irf', pw_factor)
	fpath_matchfilt = os.path.join(in_dirpath, fname_matchfilt)
	if(os.path.exists(fpath_matchfilt)):
		results_data = np.load(fpath_matchfilt)
		# coding_dict_key = 'Full Histogram (matchfilt)'
		coding_dict_key = 'Max-Res Histogram'
		results_dict[coding_dict_key] = {}
		results_dict[coding_dict_key]['mae'] = results_data['metric_mae'][row,col]*np.ones((n_code_configs,))
		results_dict[coding_dict_key]['medae'] = results_data['metric_medae'][row,col]*np.ones((n_code_configs,))
		results_dict[coding_dict_key]['1_tol_errs'] = results_data['metric_1_tol_errs'][row,col]*np.ones((n_code_configs,))
	np.set_printoptions(suppress=True)
	print("Available nphotons and sbr settings:")
	print("    nphotons_levels: {}".format(results_data['nphotons_levels'].round(decimals=3)))
	print("    sbr_levels: {}".format(results_data['sbr_levels'].round(decimals=3)))

	## Polished Plots
	plt.clf()
	# plot_utils.update_fig_size(height=6, width=8)
	ax1 = plt.subplot(2,1,1)
	ax2 = plt.subplot(2,1,2)
	# plt.close('all')
	# plt.figure(); ax1 = plt.subplot(1,1,1)
	# plt.figure(); ax2 = plt.subplot(1,1,1)
	# plt.figure(); ax3 = plt.subplot(1,1,1)
	for key in results_dict.keys():
		indeces = np.logical_not(np.isnan(results_dict[key]['mae']))
		label=key
		if(key == 'Gated Coding'): label = 'Low-Res Histo'
		elif(key == 'GatedWide Coding'): label = 'Low-Res Histo (Wide)'
		# if('Full Histogram' in key):
		if('Max-Res Histogram' in key):
			ax1.plot(n_codes_list[indeces], results_dict[key]['mae'][indeces], '--', linewidth=3, label=key)
			ax2.plot(n_codes_list[indeces], results_dict[key]['medae'][indeces], '--', linewidth=3, label=key)
			# ax3.plot(n_codes_list[indeces], results_dict[key]['1_tol_errs'][indeces], '--', linewidth=3, label=key)        
		else:
			ax1.plot(n_codes_list[indeces], results_dict[key]['mae'][indeces], '-o', linewidth=3, label=label)
			ax2.plot(n_codes_list[indeces], results_dict[key]['medae'][indeces], '-o', linewidth=3, label=label)
			# ax3.plot(n_codes_list[indeces], results_dict[key]['1_tol_errs'][indeces], '-o', linewidth=3, label=label)
	title_string = 'Total Photons = {:.1f} | SBR = {}'.format(float(nphotons_level), sbr_level)
	finalize_plot(ax1, title=title_string, ylabel='MAE (Time Bins)')
	finalize_plot(ax2, title=title_string, ylabel='Median AE (Time Bins)')
	# finalize_plot(ax3, title=title_string, ylabel='1-Tolerance Errors (Percent)')
	# Save plots
	out_dirpath = os.path.join(io_dirpaths['results_dirpath'], 'gauss_mu_est')
	out_fname_shared = 'ntbins-{}_nphotons-{:.2f}_sbr-{:.2f}_pw-{}'.format(n_tbins, float(nphotons_level), float(sbr_level), int(pw_factor))

	# set title 
	ax1.set_title('Mean Abs. Error, nphotons: {:.2f}, sbr: {:.2f}, pw: {}'.format(float(nphotons_level), float(sbr_level), int(pw_factor)),fontsize=15)
	# ax1.set_xlabel("Number of Codes",fontsize=15)
	# ax1.set_ylim(0, max_tbin_error)

	ax2.set_title('Median Abs. Error, nphotons: {:.2f}, sbr: {:.2f}, pw: {}'.format(float(nphotons_level), float(sbr_level), int(pw_factor)),fontsize=15)
	ax2.set_xlabel("Number of Codes",fontsize=15)

	plt.pause(0.1) # Pause to avoid tkinter error due to plots not being rendered
	plot_utils.save_ax_png(ax1, dirpath=out_dirpath, filename='mae_'+out_fname_shared+'.png')
	# plot_utils.save_ax_png(ax2, dirpath=out_dirpath, filename='medae_'+out_fname_shared+'.png')
	# plot_utils.save_ax_png(ax3, dirpath=out_dirpath, filename='1tolerrs_'+out_fname_shared+'.png')




