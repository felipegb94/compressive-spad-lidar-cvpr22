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

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams/tof_principle')
	colors = get_color_cycle()
	# plt.rcParams["font.family"] = "Times New Roman"

	## Set physical params
	tau = 25
	n_tbins = 1024
	(time_domain, tbin_res, tbin_bounds) = get_time_domain(tau, n_tbins)

	## Pre-proc params
	pw_factor = 0.015
	shift_factor1 = 0.05
	shift_factor2 = 0.3
	scale_factor = 0.4
	vertical_shift_factor = 0.2
	end_bin_factor = 0.3
	end_freq_factor = 0.3
	mu1 = shift_factor1*n_tbins
	mu2 = shift_factor2*n_tbins
	pulse_width = pw_factor*n_tbins

	## Generate narrow gaussian pulse
	gauss_irf1 = GaussianTIRF(n_tbins, mu=mu1, sigma=pulse_width)
	gauss_pulse1 = gauss_irf1.tirf.squeeze()

	## Get shifted pulse
	gauss_irf2 = GaussianTIRF(n_tbins, mu=mu2, sigma=pulse_width)
	gauss_pulse2 = gauss_irf2.tirf.squeeze()*scale_factor
	gauss_pulse2 += gauss_pulse1.max()*vertical_shift_factor

	## plot
	out_fname_base = 'gauss-pulse'

	plt.clf()
	plt.plot(time_domain, gauss_pulse1, '--', linewidth=3, color=colors[1])
	update_fig_size(fig=None, height=3, width=6)
	remove_yticks()
	set_xy_box()
	set_axis_linewidth(width=1)
	plt.grid(linestyle='--',alpha=0.35)
	set_ticks(fontsize=16)
	remove_yticks()
	save_currfig( dirpath = out_dirpath, filename = out_fname_base+'_emitted', file_ext = 'svg'  )

	plt.plot(time_domain, gauss_pulse2, '-', linewidth=3, color=colors[0])
	save_currfig( dirpath = out_dirpath, filename = out_fname_base+'_both', file_ext = 'svg'  )

