'''
	This script generates the raw figures used for the diagrams created for hist formation and compressive hist
'''

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
from research_utils.plot_utils import *
from research_utils import io_ops
from toflib.tirf import GaussianTIRF
from toflib.tof_utils import get_time_domain
from toflib.coding import *


# long_plot_dims = (1.75, 8)
long_plot_dims = (1.5, 6.855)
short_plot_dims = (1.5, 3.5)
colors = get_color_cycle()


def plot_coding_mat(coding_obj, is_long_plot=True, is_decoding_C=False):
	'''
		col2row_ratio = 5 was used for Compressive Histograms Overview Figure
	'''
	plt.clf()
	if(is_long_plot):
		if(is_decoding_C):
			plt.imshow(coding_obj.get_pretty_decoding_C(col2row_ratio=5), cmap='gray', vmin=-1, vmax=1)
		else:
			plt.imshow(coding_obj.get_pretty_C(col2row_ratio=5), cmap='gray', vmin=-1, vmax=1)
		update_fig_size(height=long_plot_dims[0], width=long_plot_dims[1])
	else:
		if(is_decoding_C):
			plt.imshow(coding_obj.get_pretty_decoding_C(col2row_ratio=3), cmap='gray', vmin=-1, vmax=1)
		else:
			plt.imshow(coding_obj.get_pretty_C(col2row_ratio=3), cmap='gray', vmin=-1, vmax=1)
		update_fig_size(height=short_plot_dims[0], width=short_plot_dims[1])
	remove_ticks()

def plot_codes(cmat, indeces=[0,1], is_long_plot=True):
	plt.clf()
	i = 0
	for idx in indeces:
		plt.plot(cmat[:,idx], linewidth=2, color=colors[i])
		i+=1
		i=i%len(colors)
	set_ticks(fontsize=8)
	remove_xticks()
	if(is_long_plot):
		update_fig_size(height=long_plot_dims[0]/4, width=long_plot_dims[1])
	else:
		update_fig_size(height=short_plot_dims[0]/4, width=short_plot_dims[1])

def plot_all_codes(cmat, is_long_plot=True):
	indeces = np.arange(0, cmat.shape[-1])
	plot_codes(cmat, indeces=indeces, is_long_plot=is_long_plot)
	set_xy_box()
	if(is_long_plot):
		update_fig_size(height=long_plot_dims[0]/1.5, width=long_plot_dims[1])
	else:
		update_fig_size(height=short_plot_dims[0]/1.5, width=short_plot_dims[1])

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
		colors = get_color_cycle()

	np.random.seed(0)
	out_base_dirpath = os.path.join(out_base_dirpath, 'diagrams/coding-mats')

	## Set params
	n_tbins_fullres = 1024
	n_tbins_lres = 17
	n_tbins_hres = 128

	## Pre-proc params
	sigma_factor = 0.05
	mu = 0

	## Generate narrow gaussian pulse
	sigma = sigma_factor*n_tbins_fullres
	gauss_irf = GaussianTIRF(n_tbins_fullres, mu=mu, sigma=sigma)
	irf = gauss_irf.tirf.squeeze()
	f_irf = np.fft.rfft(irf, axis=-1)

	plt.close('all')

	plt.clf()
	update_fig_size(height=4, width=7)
	plt.plot(irf, linewidth=4)
	set_ticks(fontsize=15)
	remove_yticks()
	plt.grid(axis='y', which='major',linestyle='--', linewidth=1)
	set_xy_box()
	plt.pause(0.1)
	save_currfig(dirpath=out_base_dirpath, filename='gauss-irf_sigma-{:.3f}'.format(sigma_factor), file_ext='svg')

	plt.clf()
	update_fig_size(height=4, width=7)
	plt.plot(np.abs(f_irf), linewidth=4)
	set_ticks(fontsize=15)
	remove_yticks()
	plt.grid(axis='y', which='major',linestyle='--', linewidth=1)
	set_xy_box()
	plt.pause(0.1)
	save_currfig(dirpath=out_base_dirpath, filename='gauss-freq-irf_sigma-{:.3f}'.format(sigma_factor), file_ext='svg')

	## Set plot parameters 
	ncodes = 12
	is_long_plot = False
	code_indeces = [0, ncodes//2]
	code_indeces_supplement_mats = [0, ncodes//4]

	if(is_long_plot):
		out_dirpath = os.path.join(out_base_dirpath, 'long')
	else:
		out_dirpath = os.path.join(out_base_dirpath, 'medium')

	############# Plot Long Coding Matrix #############

	hybridgrayfourier_coding = HybridGrayBasedFourierCoding(n_tbins_fullres, n_codes=ncodes, account_irf=True, filter_freqs=False, h_irf=irf)
	out_fname = "hybridgrayfourier_ncodes-{}".format(hybridgrayfourier_coding.n_codes)
	plot_coding_mat(hybridgrayfourier_coding, is_long_plot=is_long_plot)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_codes(hybridgrayfourier_coding.C, indeces=[-1,0], is_long_plot=is_long_plot)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	plot_all_codes(hybridgrayfourier_coding.C, is_long_plot=is_long_plot)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')
	out_fname = "hybridgrayfourier_ncodes-{}-irf_sigma-{:.3f}".format(hybridgrayfourier_coding.n_codes, sigma_factor)
	plot_coding_mat(hybridgrayfourier_coding, is_long_plot=is_long_plot, is_decoding_C=True)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_codes(hybridgrayfourier_coding.decoding_C, indeces=[-1,0], is_long_plot=is_long_plot)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	plot_all_codes(hybridgrayfourier_coding.decoding_C, is_long_plot=is_long_plot)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# psergrayfourier_coding = PSeriesGrayBasedFourierCoding(n_tbins_fullres, n_codes=ncodes, account_irf=True, filter_freqs=False, h_irf=irf)
	# out_fname = "psergrayfourier_ncodes-{}".format(psergrayfourier_coding.n_codes)
	# plot_coding_mat(psergrayfourier_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(psergrayfourier_coding.C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(psergrayfourier_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')
	# out_fname = "psergrayfourier_ncodes-{}-irf_sigma-{:.3f}".format(psergrayfourier_coding.n_codes, sigma_factor)
	# plot_coding_mat(psergrayfourier_coding, is_long_plot=is_long_plot, is_decoding_C=True)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(psergrayfourier_coding.decoding_C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(psergrayfourier_coding.decoding_C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# hybridfouriergray_coding = HybridFourierBasedGrayCoding(n_tbins_fullres, n_codes=ncodes, account_irf=True, h_irf=irf)
	# out_fname = "hybridfouriergray_ncodes-{}".format(hybridfouriergray_coding.n_codes)
	# plot_coding_mat(hybridfouriergray_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(hybridfouriergray_coding.C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(hybridfouriergray_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')
	# out_fname = "hybridfouriergray_ncodes-{}-irf_sigma-{:.3f}".format(hybridfouriergray_coding.n_codes, sigma_factor)
	# plot_coding_mat(hybridfouriergray_coding, is_long_plot=is_long_plot, is_decoding_C=True)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(hybridfouriergray_coding.decoding_C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(hybridfouriergray_coding.decoding_C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# pserfouriergray_coding = PSeriesFourierBasedGrayCoding(n_tbins_fullres, n_codes=ncodes, account_irf=True, h_irf=irf)
	# out_fname = "pserfouriergray_ncodes-{}".format(pserfouriergray_coding.n_codes)
	# plot_coding_mat(pserfouriergray_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(pserfouriergray_coding.C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(pserfouriergray_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')
	# out_fname = "pserfouriergray_ncodes-{}-irf_sigma-{:.3f}".format(pserfouriergray_coding.n_codes, sigma_factor)
	# plot_coding_mat(pserfouriergray_coding, is_long_plot=is_long_plot, is_decoding_C=True)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(pserfouriergray_coding.decoding_C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(pserfouriergray_coding.decoding_C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')



	# pserfourier_coding = PSeriesFourierCoding(n_tbins_fullres, n_freqs=ncodes // 2)
	# out_fname = "pserfourier_ncodes-{}".format(pserfourier_coding.n_codes)
	# plot_coding_mat(pserfourier_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(pserfourier_coding.C, indeces=code_indeces, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(pserfourier_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# fourier_coding = TruncatedFourierCoding(n_tbins_fullres, n_freqs=ncodes // 2, include_zeroth_harmonic=False)
	# out_fname = "fourier_ncodes-{}".format(fourier_coding.n_codes)
	# plot_coding_mat(fourier_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(fourier_coding.C, indeces=code_indeces, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(fourier_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# psergray_coding = PSeriesGrayCoding(n_tbins_fullres, n_codes=ncodes, account_irf=True, h_irf=irf)
	# out_fname = "psergray_ncodes-{}".format(psergray_coding.n_codes)
	# plot_coding_mat(psergray_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(psergray_coding.C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(psergray_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')
	# out_fname = "psergray_ncodes-{}-irf_sigma-{:.3f}".format(psergray_coding.n_codes, sigma_factor)
	# plot_coding_mat(psergray_coding, is_long_plot=is_long_plot, is_decoding_C=True)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(psergray_coding.decoding_C, indeces=[-1,0], is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(psergray_coding.decoding_C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# random_coding = RandomCoding(n_tbins_fullres, n_codes=ncodes)
	# out_fname = "random_ncodes-{}".format(random_coding.n_codes)
	# plot_coding_mat(random_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(random_coding.C, indeces=code_indeces_supplement_mats, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(random_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')

	# highfreqfourier_coding = HighFreqFourierCoding(n_tbins_fullres, n_high_freqs=ncodes // 2, start_high_freq=40)
	# out_fname = "highfreqfourier_ncodes-{}".format(highfreqfourier_coding.n_codes)
	# plot_coding_mat(highfreqfourier_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(highfreqfourier_coding.C, indeces=code_indeces, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(highfreqfourier_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')


	# gated_coding = GatedCoding(n_tbins_fullres, n_gates=ncodes)
	# plot_coding_mat(gated_coding, is_long_plot=is_long_plot)
	# out_fname = "gated_ncodes-{}".format(gated_coding.n_codes)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(gated_coding.C, indeces=code_indeces_supplement_mats, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')

	# whb_coding = WalshHadamardCoding(n_tbins_fullres, n_codes=ncodes)
	# out_fname = "wh_ncodes-{}".format(whb_coding.n_codes)
	# plot_coding_mat(whb_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(whb_coding.C, indeces=code_indeces_supplement_mats, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')

	# gatedfour_coding = GatedFourierCoding(n_tbins_fullres, freq_idx=[1], n_gates=ncodes//2)
	# out_fname = "gatedfour-f-1_ncodes-{}".format(gatedfour_coding.n_codes)
	# plot_coding_mat(gatedfour_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(gatedfour_coding.C, indeces=code_indeces_supplement_mats, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')

	# ktaptriangle_coding = TruncatedKTapTriangleCoding(n_tbins_fullres, n_freqs=ncodes // 2, k=2, include_zeroth_harmonic=False)
	# out_fname = "ktaptriangle_ncodes-{}".format(ktaptriangle_coding.n_codes)
	# plot_coding_mat(ktaptriangle_coding, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_codes(ktaptriangle_coding.C, indeces=code_indeces, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_rows', file_ext='svg')
	# plot_all_codes(ktaptriangle_coding.C, is_long_plot=is_long_plot)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_allrows', file_ext='svg')


	# pserbinaryfourier_coding = PSeriesBinaryFourierCoding(n_tbins_fullres, n_codes=ncodes)
	# plot_coding_mat(pserbinaryfourier_coding, is_long_plot=is_long_plot)
	# out_fname = "pserbinaryfourier_ncodes-{}".format(pserbinaryfourier_coding.n_codes)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	# psergray_coding = PSeriesGrayCoding(n_tbins_fullres, n_codes=ncodes)
	# plot_coding_mat(psergray_coding, is_long_plot=is_long_plot)
	# out_fname = "psergray_ncodes-{}".format(psergray_coding.n_codes)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')


	# for i in range(ncodes):
	# 	corr = np.dot(psergray_coding.C[:,24], psergray_coding.C[:,i])
	# 	print("Correlation {} = {}".format(i, corr))

	# complgray_coding = ComplementaryGrayCoding(n_tbins_fullres, n_codes=ncodes)
	# plot_coding_mat(complgray_coding, is_long_plot=is_long_plot)
	# out_fname = "complgray_ncodes-{}".format(complgray_coding.n_codes)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	# hadamard_coding = WalshHadamardCoding(n_tbins_fullres, n_codes=ncodes)
	# plot_coding_mat(hadamard_coding, is_long_plot=is_long_plot)
	# out_fname = "hadamard_ncodes-{}".format(hadamard_coding.n_codes)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

