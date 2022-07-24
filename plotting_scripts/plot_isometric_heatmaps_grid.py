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
from mpl_toolkits.axes_grid1 import ImageGrid
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils import plot_utils, np_utils, io_ops
from eval_coding_gauss_mu_est import compose_fname
from plotting_scripts.plot_isometric_heatmaps import *

def get_rec_algo_id(coding_id, account_irf=True):
	rec_algo_id = 'ncc'
	if('Gated' in coding_id): rec_algo_id =  'linear'
	elif('GatedZNCC' in coding_id): rec_algo_id =  'zncc'
	elif('Timestamp' == coding_id): rec_algo_id =  'matchfilt'
	elif('Identity' == coding_id): rec_algo_id =  'matchfilt'
	if(account_irf): rec_algo_id+='-irf'
	return rec_algo_id

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	# plt.style.use('dark_background')

	## Get dirpaths and set params
	n_tbins=1024
	absmin_logsbr = -2 
	absmax_logsbr = 0
	absmin_lognphotons = 2 
	absmax_lognphotons = 4
	base_dirpath = io_dirpaths['results_data']
	rel_dirpath = 'final_coding_gauss_mu_est/ntbins-{}_logsbr-{:.1f}-{:.1f}_lognp-{:.1f}-{:.1f}'.format(n_tbins, absmin_logsbr, absmax_logsbr, absmin_lognphotons, absmax_lognphotons)
	data_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], base_dirpath, rel_dirpath)
	out_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'], 'IsometricHeatmaps')
	## Set parameter
	metric_id = 'mae' #options: mae, medae, 1tol
	n_codes = 8
	pw_factor = 1
	## Set Grid Size and coding schemes we will plot
	grid_size = (1, 2)
	coding_ids = ['TruncatedFourier', 'PSeriesGray']
	n_coding_ids = len(coding_ids)

	## Load target data
	target_coding_id = 'Identity'
	target_rec_algo_id = get_rec_algo_id(target_coding_id)
	target_fname = compose_fname(target_coding_id, n_tbins, target_rec_algo_id, pw_factor)
	target_data = np.load(os.path.join(data_dirpath, target_fname))
	
	## extract variables 
	X_sbr_levels = target_data['X_sbr_levels']
	Y_nphotons_levels = target_data['Y_nphotons_levels']

	plt.clf()
	fig = plt.gcf()
	# axs = fig.subplots(1,2,sharex=True, sharey=True)
	# fig, axs = plt.subplots(1,2)
	# grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.1)
	# for ax in axs:
	# 	ax.set_xscale('log')
	# 	ax.set_yscale('log')

	for coding_id, i in zip(coding_ids, np.arange(0,n_coding_ids)):

		# curr_ax = axs[i]
		rec_algo_id = get_rec_algo_id(coding_id)
		## Load data from a coding scheme
		if(coding_id == 'GatedWide'):
			fname = compose_fname('Gated', n_codes, rec_algo_id, pw_factor=n_tbins/n_codes)
			data = np.load(os.path.join(data_dirpath, fname))		
		else:
			fname = compose_fname(coding_id, n_codes, rec_algo_id, pw_factor)
			data = np.load(os.path.join(data_dirpath, fname))
		contour_lvls = get_contour_lvls()
		plt.subplot(1,2,i+1)
		(ax, img) = plot_isometric_heatmap(data, target_data, metric_id='mae', ax=plt.gca())
		# ax.contourf(data['metric_mae'])
		# img = ax.contourf(data['X_sbr_levels'], data['Y_nphotons_levels'],np.abs(data['metric_mae']-target_data['metric_mae']),
		# 					cmap="YlGnBu", levels=contour_lvls, norm=mpl.colors.LogNorm(vmin=contour_lvls[1], vmax=contour_lvls[-1]))
		# ax.set_xscale('log')
		# ax.set_yscale('log')

		# hb = ax.hexbin(df.x, df.y, C=df[col], cmap='RdBu_r', vmin=0, vmax=1)
		# ax.set_title(col)
		# ax.set_xlabel('x')
		# ax.set_ylabel('y')


	# # plot_utils.save_currfig(dirpath = out_dirpath, filename = out_fname, file_ext = 'svg')






