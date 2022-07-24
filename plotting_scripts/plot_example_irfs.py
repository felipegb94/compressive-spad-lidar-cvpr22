'''
	This script generates the raw figures used for the diagrams created for IRF descriptions
'''

## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.plot_utils import *
from research_utils import io_ops
from toflib.tirf import GaussianTIRF
from research_utils import plot_utils, io_ops
from toflib.tof_utils import get_time_domain
from scan_data_scripts.scan_data_utils import get_irf, get_scene_irf

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams/irfs')
	colors = get_color_cycle()

	## Set physical params
	tau = 1.
	n_tbins = 1000

	## Pre-proc params
	narrow_sigma_factor = 0.001
	wide_sigma_factor = 0.006
	roll_factor = 0.02
	end_bin_factor = 0.1
	end_freq_factor = 0.3
	mu = roll_factor*n_tbins

	## Generate narrow gaussian pulse
	narrow_sigma = narrow_sigma_factor*n_tbins
	narrow_gauss_irf = GaussianTIRF(n_tbins, mu=mu, sigma=narrow_sigma)
	narrow_pulse = narrow_gauss_irf.tirf.squeeze()
	f_narrow_pulse = np.fft.rfft(narrow_pulse, axis=-1)

	## Generate wide gaussian pulse
	wide_sigma = wide_sigma_factor*n_tbins
	wide_gauss_irf = GaussianTIRF(n_tbins, mu=mu, sigma=wide_sigma)
	wide_pulse = wide_gauss_irf.tirf.squeeze()
	f_wide_pulse = np.fft.rfft(wide_pulse, axis=-1)

	## Load real data pulse
	# this irf is loaded from scan_data_scripts/system_irf
	real_data_irf = get_scene_irf('20190207_face_scanning_low_mu/free', n=n_tbins, tlen=6656, tres=8, is_unimodal=True)
	real_data_irf /= real_data_irf.sum()
	real_data_irf = np.roll(real_data_irf, int(mu))
	f_real_data_irf = np.fft.rfft(real_data_irf, axis=-1)

	real_data_irf2 = get_scene_irf('20190207_face_scanning_low_mu/ground_truth', n=n_tbins, tlen=6656, tres=8, is_unimodal=True)
	real_data_irf2 /= real_data_irf2.sum()
	real_data_irf2 = np.roll(real_data_irf2, int(mu))
	f_real_data_irf2 = np.fft.rfft(real_data_irf2, axis=-1)

	## Random coding function
	random_code = (np.random.rand(n_tbins)*2)-1

	## Set time domain ticks with Latex fonts
	n_ticks = 5
	time_ticks = np.around(np.linspace(0, n_tbins*end_bin_factor, n_ticks), decimals=1)
	time_ticks_str = []
	for tick_val in time_ticks:
		if(tick_val == 0): time_ticks_str.append('0')
		else: time_ticks_str.append('{:.2f}'.format(tick_val/n_tbins)+r'$\tau$') 
	freq_ticks = np.linspace(0, n_tbins*end_freq_factor, n_ticks).round().astype(int)
	freq_ticks_str = []
	for tick_val in freq_ticks:
		if(tick_val == 0): freq_ticks_str.append('0')
		else: freq_ticks_str.append('{}f'.format(tick_val)+r'$_0$') 


	plt.clf()
	plt.plot(narrow_pulse*0.4, linewidth=4)
	plt.xlim([0, end_bin_factor*n_tbins])
	plot_utils.set_ticks(fontsize=15)
	plot_utils.update_fig_size(height=3, width=5)
	plot_utils.remove_yticks()
	plt.grid(axis='y', which='major',linestyle='--', linewidth=1)
	plot_utils.set_xy_box()
	plt.xticks(time_ticks, time_ticks_str)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='irf-narrow_time')

	plt.plot(wide_pulse, linewidth=4)
	plt.plot(real_data_irf, linewidth=4)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='irf_time')

	plt.clf()
	plt.plot(np.abs(f_narrow_pulse), '*',linewidth=2)
	plt.plot(np.abs(f_wide_pulse), '*',linewidth=2)
	plt.plot(np.abs(f_real_data_irf), '*',linewidth=2)
	plt.xlim([0, end_freq_factor*n_tbins])
	plot_utils.set_ticks(fontsize=15)
	plot_utils.update_fig_size(height=3, width=5)
	plot_utils.set_xy_box()
	plt.grid(axis='y', which='major',linestyle='--', linewidth=1)
	plot_utils.remove_yticks()
	plt.xticks(freq_ticks, freq_ticks_str)
	plot_utils.save_currfig(dirpath=out_dirpath, filename='irf_freq')


	## Test band-limit propoerty
	freq_offset = 150
	freq = np.random.randint(freq_offset,200)
	domain = np.arange(0,n_tbins)/n_tbins
	random_shift = np.pi*np.random.rand()
	sinusoid = np.cos((2*np.pi*freq*domain) - random_shift)
	product_narrow_pulse = np.dot(sinusoid, narrow_pulse) 
	product_wide_pulse = np.dot(sinusoid, wide_pulse) 
	product_real_irf = np.dot(sinusoid, real_data_irf) 
	print("freq = {}, shift = {:.2f}, product_narrow_pulse = {}".format(freq, random_shift, product_narrow_pulse))
	print("freq = {}, shift = {:.2f}, product_wide_pulse = {}".format(freq, random_shift, product_wide_pulse))
	print("freq = {}, shift = {:.2f}, product_real_irf = {}".format(freq, random_shift, product_real_irf))

	# plt.clf()
	# plt.subplot(2,2,1)
	# plt.plot(narrow_pulse, linewidth=2)
	# plt.plot(wide_pulse, linewidth=2)
	# plt.plot(real_data_irf, linewidth=2)
	# plt.plot(real_data_irf2, linewidth=2)
	# # remove_yticks()

	# plt.subplot(2,2,2)
	# plt.plot(narrow_pulse, linewidth=2)
	# plt.plot(wide_pulse, linewidth=2)
	# plt.plot(real_data_irf, linewidth=2)
	# plt.plot(real_data_irf2, linewidth=2)
	# plt.xlim([400,600])
	# # remove_yticks()
	
	# plt.subplot(2,2,3)
	# plt.plot(np.abs(np.fft.rfft(narrow_pulse)), linewidth=3)
	# plt.plot(np.abs(np.fft.rfft(wide_pulse)), linewidth=3)
	# plt.plot(np.abs(np.fft.rfft(real_data_irf)), linewidth=3)
	# plt.plot(np.abs(np.fft.rfft(real_data_irf2)), linewidth=3)
	# # remove_yticks()

	# plt.subplot(2,2,4)
	# plt.plot(np.abs(np.fft.rfft(narrow_pulse)), linewidth=3)
	# plt.plot(np.abs(np.fft.rfft(wide_pulse)), linewidth=3)
	# plt.plot(np.abs(np.fft.rfft(real_data_irf)), linewidth=3)
	# plt.plot(np.abs(np.fft.rfft(real_data_irf2)), linewidth=3)
	# plt.xlim([0,100])
	# # remove_yticks()


	# save_currfig(dirpath=out_dirpath, filename='draft_irfs')


	# plt.subplot(2,2,5)
	# plt.plot(random_code, linewidth=3)
	# plt.subplot(2,2,6)
	# plt.plot(np.abs(np.fft.rfft(random_code)), linewidth=3)

