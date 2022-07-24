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
from toflib.coding import GrayCoding, TruncatedFourierCoding, PSeriesFourierCoding


def prepare_plot_long(tau):
	update_fig_size(height=2, width=8)
	remove_ticks()
	set_xy_box()
	set_axis_linewidth(width=2)
	set_xy_arrow()
	plt.xlim([0,tau])
	plt.ylim([0,1.1])

def prepare_plot(tau):
	update_fig_size(height=3, width=8)
	remove_ticks()
	set_xy_box()
	set_axis_linewidth(width=3)
	# set_xy_arrow()
	plt.xlim([0,tau])
	plt.ylim([0,1.1])

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams')
	colors = get_color_cycle()

	## Set random seed to get the same histograms
	np.random.seed(0)

	# ## Set physical params (Params used for paper diagrams)
	# tau = 1.
	# (mu, sigma) = (0.5*tau, 0.05*tau)
	# n_tbins_fullres = 1024
	# n_tbins_lres = 17
	# n_tbins_hres = 128
	# n_photons = 2000
	# sbr = 0.3

	## Set physical params (Params used for presentaton)
	tau = 1.
	(mu, sigma) = (0.5*tau, 0.03*tau)
	n_tbins_fullres = 1024
	n_tbins_lres = 17
	n_tbins_hres = 128
	n_photons = 1000
	sbr = 0.1

	## Create coding obj
	n_bits = 8
	gray_coding = GrayCoding(n_tbins_fullres, n_bits=n_bits)
	n_freqs = 8
	fourier_coding = TruncatedFourierCoding(n_tbins_fullres, n_freqs=n_freqs, include_zeroth_harmonic=False)
	pserfourier_coding = PSeriesFourierCoding(n_tbins_fullres, n_freqs=n_freqs, include_zeroth_harmonic=False)

	# Create tbins
	(tbins_fullres, tbin_res_fullres, tbin_bounds_fullres) = get_time_domain(tau, n_tbins_fullres)
	(tbins_lres, tbin_res_lres, tbin_bounds_lres) = get_time_domain(tau, n_tbins_lres)
	(tbins_hres, tbin_res_hres, tbin_bounds_hres) = get_time_domain(tau, n_tbins_hres)

	## Generate gaussian pulse
	gauss_obj_fullres = GaussianTIRF(n_tbins_fullres, mu=mu, sigma=sigma, t_domain=tbins_fullres)
	gauss_obj_lres = GaussianTIRF(n_tbins_lres, mu=mu, sigma=sigma, t_domain=tbins_lres)
	gauss_obj_hres = GaussianTIRF(n_tbins_hres, mu=mu, sigma=sigma, t_domain=tbins_hres)
	gauss_obj_fullres.set_sbr(sbr)
	gauss_obj_lres.set_sbr(sbr)
	gauss_obj_hres.set_sbr(sbr)

	## Get waveform and histtograms
	waveform_fullres = gauss_obj_fullres.simulate_n_photons(n_photons=n_photons, add_noise=False).squeeze()
	waveform_lres = gauss_obj_lres.simulate_n_photons(n_photons=n_photons, add_noise=False).squeeze()
	waveform_hres = gauss_obj_hres.simulate_n_photons(n_photons=n_photons, add_noise=False).squeeze()
	hist_fullres = gauss_obj_fullres.simulate_n_photons(n_photons=n_photons, add_noise=True).squeeze()
	hist_lres = gauss_obj_lres.simulate_n_photons(n_photons=n_photons, add_noise=True).squeeze()
	hist_hres = gauss_obj_hres.simulate_n_photons(n_photons=n_photons, add_noise=True).squeeze()

	## Get compressive histogram
	gray_hist = gray_coding.encode(hist_fullres)
	fourier_hist = fourier_coding.encode(hist_fullres)
	pserfourier_hist = pserfourier_coding.encode(hist_fullres)

	############# Plots for Histogram Formation Diagram #############
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams/hist-formation')
	plt.close('all')

	## plot waveform
	plt.clf()
	plt.plot(tbins_fullres, waveform_fullres/ waveform_fullres.max(), linewidth=4)
	prepare_plot_long(tau)
	save_currfig(dirpath=out_dirpath, filename='gauss-waveform_sbr-{:.2f}'.format(sbr), file_ext='svg')


	plt.clf()
	plt.bar(tbins_lres, hist_lres / hist_lres.max(), width=tbin_res_lres, align='center', edgecolor='white', alpha=0.6)
	plt.plot(tbins_fullres, waveform_fullres/ waveform_fullres.max(), linewidth=4)
	prepare_plot_long(tau)
	out_fname = 'gauss-hist_nt-{}_sbr-{:.2f}_nphotons-{}'.format(n_tbins_lres, sbr, n_photons)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')


	############# Plots for Compressive Histograms Overview Diagram
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams/compressive-hist')

	plt.clf()
	plt.bar(tbins_hres, hist_hres / hist_hres.max(), width=tbin_res_hres, align='center', edgecolor='black', alpha=0.7)
	prepare_plot(tau)
	out_fname = 'gauss-hist_nt-{}_sbr-{:.2f}_nphotons-{}'.format(n_tbins_hres, sbr, n_photons)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	plt.clf()
	plt.bar(tbins_lres, hist_lres / hist_lres.max(), width=tbin_res_lres, align='center', edgecolor='black', alpha=0.7)
	prepare_plot(tau)
	out_fname = 'gauss-hist_nt-{}_sbr-{:.2f}_nphotons-{}'.format(n_tbins_lres, sbr, n_photons)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	plt.clf()
	plt.bar(np.arange(0, gray_hist.size), gray_hist / np.abs(gray_hist).max(), width=1, align='center', edgecolor='black', alpha=0.7)
	prepare_plot(tau)
	plt.xlim([-0.5,gray_hist.size])
	plt.ylim([-1,1])
	ax = plt.gca()
	ax.spines['bottom'].set_position('center')
	out_fname = 'gray-gauss-hist_ncodes-{}_nt-{}_sbr-{:.2f}_nphotons-{}'.format(gray_hist.size, n_tbins_lres, sbr, n_photons)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	plt.clf()
	plt.bar(np.arange(0, fourier_hist.size), fourier_hist / np.abs(fourier_hist).max(), width=1, align='center', edgecolor='black', alpha=0.7)
	prepare_plot(tau)
	plt.xlim([-0.5,fourier_hist.size])
	plt.ylim([-1,1])
	ax = plt.gca()
	ax.spines['bottom'].set_position('center')
	out_fname = 'fourier-gauss-hist_ncodes-{}_nt-{}_sbr-{:.2f}_nphotons-{}'.format(fourier_hist.size, n_tbins_lres, sbr, n_photons)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

	plt.clf()
	plt.bar(np.arange(0, pserfourier_hist.size), pserfourier_hist / np.abs(pserfourier_hist).max(), width=1, align='center', edgecolor='black', alpha=0.7)
	prepare_plot(tau)
	plt.xlim([-0.5,pserfourier_hist.size])
	plt.ylim([-1,1])
	ax = plt.gca()
	ax.spines['bottom'].set_position('center')
	out_fname = 'pseriesfourier-gauss-hist_ncodes-{}_nt-{}_sbr-{:.2f}_nphotons-{}'.format(pserfourier_hist.size, n_tbins_lres, sbr, n_photons)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')

