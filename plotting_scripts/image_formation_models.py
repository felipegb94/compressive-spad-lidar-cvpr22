'''
Plot functions to illustrate different components of the image formation model
'''

## Standard Library Imports
import os
import sys
sys.path.append('../')

## Library Imports
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.artist as artist
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from io_utils import input_args_parser, io_dirpaths
from utils import plot_utils
from utils.plot_utils import *
from utils.shared_constants import *
import coding, decoding
import common

def finalize_plot(legend, fontsize=14):
    set_legend(legend, fontsize=fontsize)
    set_ticks(fontsize=fontsize)
    remove_yticks()

# Set dark plot theme
plt.style.use('dark_background')

## Get output folder
results_dirpath = io_dirpaths.get_dirpath(dirpath_id='results_dirpath', io_dirpath_json_file_dir='../')
results_dirpath = os.path.join(results_dirpath, 'image_formation')

## Time Domain parameters
n_tbins = 129
bin_size = 1
T = bin_size * n_tbins
# Define time domain
t = np.arange(0, n_tbins)*bin_size
n_photons = 1

## System Impulse Response Model (Gaussian)
narrow_pulse_width = 1*bin_size
wide_pulse_width = 5*narrow_pulse_width
mu = wide_pulse_width*2
gauss_pulse_narrow = coding.gaussian_pulse(t, mu=mu, width=narrow_pulse_width, circ_shifted=True)
gauss_pulse_wide = coding.gaussian_pulse(t, mu=mu, width=wide_pulse_width, circ_shifted=True)

plt.clf()
plt.plot(t - mu, gauss_pulse_narrow / gauss_pulse_narrow.max(), linewidth=3)
plt.plot(t - mu, gauss_pulse_wide / gauss_pulse_wide.max(), linewidth=3)
legend = ['Narrow Pulse, Width = {} tbins'.format(int(narrow_pulse_width/bin_size)), 'Wide Pulse, Width = {} tbins'.format(int(wide_pulse_width/bin_size))]
finalize_plot(legend=legend)
save_rgb(dirpath=results_dirpath, filename='system_irf_gauss_model', rm_ticks=False)

## System Impulse Response Model (Exp. Modified Gaussian)
exp_lambda = 1 / (7*bin_size)

expgauss_pulse_narrow = coding.expgaussian_pulse_conv(t, mu = mu, sigma = narrow_pulse_width, exp_lambda = exp_lambda)
expgauss_pulse_wide = coding.expgaussian_pulse_conv(t, mu = mu, sigma = wide_pulse_width, exp_lambda = exp_lambda)

plt.clf()
plt.plot(t - mu, expgauss_pulse_narrow, linewidth=3)
plt.plot(t - mu, expgauss_pulse_wide, linewidth=3)
legend = ['Narrow Pulse, Width = {} tbins'.format(int(narrow_pulse_width/bin_size)), 'Wide Pulse, Width = {} tbins'.format(int(wide_pulse_width/bin_size))]
finalize_plot(legend=legend)
save_rgb(dirpath=results_dirpath, filename='system_irf_expgauss_model', rm_ticks=False)


## Scene Impulse Response Model (Delta)
mu = T / 3.
sigma = 0.01*bin_size

delta_scene_irf = coding.gaussian_pulse(t, mu=mu, width=sigma, circ_shifted=True)
plt.clf()
plt.plot(t / T, delta_scene_irf, linewidth=3)
legend = ['Scene IRF']
finalize_plot(legend=legend, fontsize=16)
save_rgb(dirpath=results_dirpath, filename='scene_irf_delta_model', rm_ticks=False)

## Scene Impulse Response Model (Gaussian)
mu = T / 3.
sigma = 3*bin_size

gauss_scene_irf = coding.gaussian_pulse(t, mu=mu, width=sigma, circ_shifted=True)
plt.clf()
plt.plot(t / T, gauss_scene_irf, linewidth=3)
legend = ['Scene IRF']
finalize_plot(legend=legend, fontsize=16)
save_rgb(dirpath=results_dirpath, filename='scene_irf_gauss_model', rm_ticks=False)

## Scene Impulse Response Model (Gauss Mixture)
mu1 = T / 3.
mu2 = T / 1.7
sigma1 = 1*bin_size
sigma2 = 1*bin_size
alpha1 = 1
alpha2 = 0.6

gauss1 = alpha1*coding.gaussian_pulse(t, mu=mu1, width=sigma1, circ_shifted=True)
gauss2 = alpha2*coding.gaussian_pulse(t, mu=mu2, width=sigma2, circ_shifted=True)

gmm_scene_irf = (gauss1 + gauss2) / (alpha1 + alpha2)
plt.clf()
plt.plot(t / T, gmm_scene_irf, linewidth=3)
legend = ['Scene IRF']
finalize_plot(legend=legend, fontsize=16)
save_rgb(dirpath=results_dirpath, filename='scene_irf_gmm_model', rm_ticks=False)

## Scene Impulse Response Model (Gauss Mixture)
transient_dirpath = io_dirpaths.get_dirpath(dirpath_id='sample_transient_images_dirpath', io_dirpath_json_file_dir='../')
scene_id = 'bathroom2'
scene_filename = '{}_nr-240_nc-320_nt-2000_samples-2048_view-0.npz'.format(scene_id)

real_world_transient_img = np.load(os.path.join(transient_dirpath, scene_filename))['arr_0']

(r,c) = (190,214)
(r,c) = (52,255)

real_world_scene_irf = real_world_transient_img[r,c,:] 
real_world_scene_irf /= real_world_scene_irf.sum() 

n_tbins = real_world_scene_irf.shape[-1]
bin_size = 1
T = bin_size * n_tbins
t = np.arange(0,n_tbins)*bin_size

plt.clf()
plt.plot(t / T, real_world_scene_irf, linewidth=3)
legend = ['Scene IRF']
finalize_plot(legend=legend, fontsize=16)
save_rgb(dirpath=results_dirpath, filename='scene_irf_{}_scene'.format(scene_id), rm_ticks=False)

