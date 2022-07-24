'''
	This script generates the raw figures used for coding curve plots in supplement
'''

## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.plot_utils import *
from research_utils import io_ops
from toflib.coding import *
from plotting_scripts.plot_coding_matrices import plot_coding_mat

colors = get_color_cycle()
plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
dark_mode = plot_params['dark_mode']

def plot_coding_curve(cmat, add_points=True):
	plt.clf()
	ax = plt.axes(projection='3d')
	if(dark_mode):
		ax.plot3D(cmat[:,0], cmat[:,1], cmat[:,2], 'yellow', linewidth=4)
		if(add_points):
			# ax.scatter3D(cmat[:,0], cmat[:,1], cmat[:,2], c=domain, linewidth=6)
			# ax.scatter3D(cmat[:,0], cmat[:,1], cmat[:,2], c=domain, cmap='YlOrRd', linewidth=6)
			ax.scatter3D(cmat[:,0], cmat[:,1], cmat[:,2], c=domain, cmap='Wistia', linewidth=6)
		else:
			ax.plot3D(cmat[:,0], cmat[:,1], cmat[:,2], 'yellow', linewidth=5)
		ax.xaxis._axinfo["grid"]['linestyle'] = ":"
		ax.yaxis._axinfo["grid"]['linestyle'] = ":"
		ax.zaxis._axinfo["grid"]['linestyle'] = ":"
	else:
		ax.plot3D(cmat[:,0], cmat[:,1], cmat[:,2], 'black', linewidth=3)
		if(add_points):
			ax.scatter3D(cmat[:,0], cmat[:,1], cmat[:,2], c=domain, cmap='jet', linewidth=5)
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	# ax.set_xlabel('Coding Function 1', fontsize=20)
	# ax.set_ylabel('Coding Function 2', fontsize=20)
	# ax.set_zlabel('Coding Function 3', fontsize=20)
	update_fig_size(height=8, width=8)
	set_ticks(ax, fontsize=20)

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = os.path.join(io_dirpaths['data_base_dirpath'], io_dirpaths['paper_results_dirpath'])
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"
	out_dirpath = os.path.join(out_base_dirpath, 'diagrams/coding-curves')
	np.random.seed(0)

	## Set params
	n_tbins_fullres = 120
	domain = np.arange(0,n_tbins_fullres)*(1/n_tbins_fullres)*2*np.pi

	## Set plot parameters 
	ncodes = 3

	############# Plot Coding Curves #############


	# fourier_coding = TruncatedFourierCoding(n_tbins_fullres, n_freqs=int(np.ceil(ncodes / 2)), include_zeroth_harmonic=False)
	# fourier_coding.update_C(C=fourier_coding.C[:,0:ncodes])
	# cobj = fourier_coding 
	# cmat = fourier_coding.C
	# out_fname = "fourier_ncodes-{}".format(ncodes)
	# plot_coding_curve(cmat)
	# save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# plot_coding_mat(cobj)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_mat', file_ext='svg')

	psergray_coding = PSeriesGrayCoding(n_tbins_fullres, n_codes=ncodes)
	cobj = psergray_coding 
	cmat = psergray_coding.C 
	out_fname = "psergray_ncodes-{}".format(psergray_coding.n_codes)
	plot_coding_curve(cmat)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_coding_curve(cmat, add_points=False)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_nopoints', file_ext='svg')
	# plot_coding_mat(cobj)
	# save_currfig(dirpath=out_dirpath, filename=out_fname+'_mat', file_ext='svg')

	############# Self-intersecting
	gated_coding = GatedCoding(n_tbins_fullres, n_gates=ncodes)
	cobj = gated_coding 
	cmat = gated_coding.C 
	out_fname = "gated_ncodes-{}".format(gated_coding.n_codes)
	plot_coding_curve(cmat)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_coding_mat(cobj)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_mat', file_ext='svg')

	selfintersectingfourier_coding = FourierCoding(n_tbins_fullres, freq_idx=[2,3])
	selfintersectingfourier_coding.update_C(C=selfintersectingfourier_coding.C[:,0:ncodes])
	cobj = selfintersectingfourier_coding 
	cmat = selfintersectingfourier_coding.C
	out_fname = "selfintersectingfourier_ncodes-{}".format(selfintersectingfourier_coding.n_codes)
	plot_coding_curve(cmat)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_coding_mat(cobj)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_mat', file_ext='svg')


	############# non-locality preserving
	dualfreqfourier_coding = FourierCoding(n_tbins_fullres, freq_idx=[10,11])
	dualfreqfourier_coding.update_C(C=dualfreqfourier_coding.C[:,0:ncodes])
	cobj = dualfreqfourier_coding 
	cmat = dualfreqfourier_coding.C 
	out_fname = "dualfreqfourier_ncodes-{}".format(dualfreqfourier_coding.n_codes)
	plot_coding_curve(cmat)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_coding_mat(cobj)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_mat', file_ext='svg')

	random_coding = RandomCoding(n_tbins_fullres, n_codes=3)
	cobj = random_coding 
	cmat = random_coding.C 
	out_fname = "random_ncodes-{}".format(random_coding.n_codes)
	plot_coding_curve(cmat)
	save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plot_coding_mat(cobj)
	save_currfig(dirpath=out_dirpath, filename=out_fname+'_mat', file_ext='svg')

