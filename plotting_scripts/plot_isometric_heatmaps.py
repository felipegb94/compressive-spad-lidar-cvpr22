'''
	plots isometric contours of a given coding approach with respect to the ideal setting where you have the full histogram available
'''
## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib as mpl
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

def err2relerr(errs, maxval):
	return 100*(errs/maxval)

def get_mask(X, min_val, max_val):
	X_min = X >= min_val
	X_max = X <= max_val
	return np.logical_not(np.logical_and(X_min, X_max))

def get_contour_lvls(): return np.array([1e-16, 1e-2, 0.1, 1, 10, 25, 40])

def plot_isometric_heatmap(data, target_data, metric_id='mae', ax=None):
	n_tbins = target_data['n_tbins']
	## Create a heatmap from the difference
	if(metric_id == 'mae'):
		target_metric = err2relerr(target_data['metric_mae'], n_tbins)
		metric = err2relerr(data['metric_mae'], n_tbins)
	elif(metric_id == 'medae'):
		target_metric = err2relerr(target_data['metric_medae'], n_tbins)
		metric = err2relerr(data['metric_medae'], n_tbins)
	elif(metric_id == 'medae'):
		target_metric = err2relerr(target_data['metric_1tol'], n_tbins)
		metric = err2relerr(data['metric_1tol'], n_tbins)
	else:
		assert(False), "metric not available"	
	## NOTE: For a hist of 1000 tbins, a rel error of 1 tbin == 0.1
	metric_diff = metric - target_metric
	if(np.any(metric_diff < -0.4)):
		print("coding {} significantly outperforming target".format(fname))
	metric_diff[metric_diff <= 0.0] = 1e-4 # there are cases 
	# metric_diff[metric_diff < -0.1] = np.nan # there are cases 
	metric_abs_diff = np.abs(metric_diff)
	(fig, ax) = (plt.gcf(), plt.gca()) 
	##### Other attempts to plot that did not work
	# img = ax.contourf(data['X_sbr_levels'], data['Y_nphotons_levels'], np.log10(metric_abs_diff), 
	# 					cmap="YlGnBu", levels=1000, vmin=np.log10(0.1), vmax=np.log10(10))
	# img = ax.contourf(data['X_sbr_levels'], data['Y_nphotons_levels'], metric_abs_diff, 
	# 					cmap="YlGnBu", levels=1000, norm=mpl.colors.LogNorm())
	#### Contour plots worked best
	contour_lvls = get_contour_lvls()
	if(ax is None):
		ax = plt.gca()
	## Set log scale
	ax.set_xlim(data['X_sbr_levels'].min(), data['X_sbr_levels'].max())
	ax.set_xlim(data['X_sbr_levels'].min(), data['X_sbr_levels'].max())
	ax.set_xscale('log')
	ax.set_yscale('log')
	img = ax.contourf(data['X_sbr_levels'], data['Y_nphotons_levels'], metric_abs_diff,
						cmap="YlGnBu", levels=contour_lvls, norm=mpl.colors.LogNorm(vmin=contour_lvls[1], vmax=contour_lvls[-1]))
	# breakpoint()
	ax.set_aspect('equal')
	return (ax, img)


if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['results_data'])

	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"


	## Get dirpaths and set params
	n_tbins=1024
	absmin_logsbr = -2 
	absmax_logsbr = 0
	absmin_lognphotons = 2 
	absmax_lognphotons = 4
	rel_dirpath = 'final_coding_gauss_mu_est/ntbins-{}_logsbr-{:.1f}-{:.1f}_lognp-{:.1f}-{:.1f}'.format(n_tbins, absmin_logsbr, absmax_logsbr, absmin_lognphotons, absmax_lognphotons)
	data_dirpath = os.path.join(base_dirpath, rel_dirpath)
	out_dirpath = os.path.join(out_base_dirpath, 'IsometricHeatmaps')

	## Set parameter
	metric_id = 'mae' #options: mae, medae, 1tol 

	## Set coding schemes we want to plot
	n_codes = 64
	pw_factor = 1
	coding_ids = ['TruncatedFourier', 'PSeriesGray', 'PSeriesFourier', 'Gated', 'GatedWide', 'GatedFourier-F-1', 'Timestamp']
	rec_algo_ids = ['ncc-irf','ncc-irf','ncc-irf','linear-irf','linear-irf','ncc-irf','matchfilt-irf']
	coding_ids = ['TruncatedFourier', 'PSeriesGray', 'PSeriesFourier', 'Gated', 'GatedWide', 'Timestamp']
	rec_algo_ids = ['ncc-irf','ncc-irf','ncc-irf','linear-irf','linear-irf','matchfilt-irf']
	# coding_ids = ['TruncatedFourier']
	# rec_algo_ids = ['ncc-irf']
	# coding_ids = ['PSeriesFourier']
	# rec_algo_ids = ['ncc-irf']

	# idx_to_plot = [0]
	idx_to_plot = np.arange(0,len(coding_ids))

	## Load target data
	target_coding_id = 'Identity'
	target_rec_algo_id = 'matchfilt-irf'
	target_fname = compose_fname(target_coding_id, n_tbins, target_rec_algo_id, pw_factor)
	target_data = np.load(os.path.join(data_dirpath, target_fname))
	
	## extract variables 
	X_sbr_levels = target_data['X_sbr_levels']
	Y_nphotons_levels = target_data['Y_nphotons_levels']

	# ## Mask some SBR and photon count regions (NOT USED ANYMORE)
	# (min_sbr_lvl, max_sbr_lvl) = (0.1, 1.0)
	# (min_nphotons_lvl, max_nphotons_lvl) = (500, 10000)
	# X_sbr_mask = get_mask(X_sbr_levels, min_sbr_lvl, max_sbr_lvl)
	# Y_nphotons_mask = get_mask(Y_nphotons_levels, min_nphotons_lvl, max_nphotons_lvl)
	# mask = np.logical_and(X_sbr_mask, Y_nphotons_mask)

	for idx in idx_to_plot:
		## Load data from a coding scheme
		if(coding_ids[idx] == 'GatedWide'):
			fname = compose_fname('Gated', n_codes, rec_algo_ids[idx], pw_factor=n_tbins/n_codes)
			data = np.load(os.path.join(data_dirpath, fname))		
		else:
			fname = compose_fname(coding_ids[idx], n_codes, rec_algo_ids[idx], pw_factor)
			data = np.load(os.path.join(data_dirpath, fname))
		out_fname = 'mae-isometric_{}'.format(fname.split('.npz')[0])
		
		plt.clf()
		contour_lvls = get_contour_lvls()
		(ax, img) = plot_isometric_heatmap(data, target_data, metric_id='mae')
		fig = plt.gcf()
		## Set font sizes
		plot_utils.set_ticks(fontsize=20)
		plot_utils.set_axis_linewidth(width=1.5)
		ax.tick_params(
			axis='both',          # changes apply to the x-axis and y-axis
			which='major',      # both major and minor ticks are affected
			length=5,
			width=1.5,
			) # labels along the bottom edge are off
		ax.tick_params(
			axis='both',          # changes apply to the x-axis and y-axis
			which='minor',      # both major and minor ticks are affected
			length=3,
			width=1.,
			) # labels along the bottom edge are off
		plot_utils.save_currfig(dirpath = out_dirpath, filename = out_fname, file_ext = 'svg')
		## The cbar ticks should match the contour levels
		cbar_ticks = contour_lvls[1:]
		cbar = fig.colorbar(img, ticks=cbar_ticks, pad=0.01, aspect=15)
		cbar.ax.set_yticklabels(cbar_ticks)  # vertically oriented colorbar
		cbar.ax.tick_params(labelsize=20)
		plot_utils.save_currfig(dirpath = out_dirpath, filename = out_fname + '_withcbar', file_ext = 'svg')






