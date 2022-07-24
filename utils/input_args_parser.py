'''
	Contains the code that puts together the parsers for the different scripts
'''
## Standard Library Imports
import argparse

## Library Imports
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports


def add_flash_lidar_scene_args( parser ):
	parser.add_argument('-n_rows', type=int, default = 120, help='Number of rows')
	parser.add_argument('-n_cols', type=int, default = 160, help='Number of cols')
	parser.add_argument('-n_tbins', type=int, default = 2000, help='Number of tbins')
	parser.add_argument('-max_transient_path_len', type=float, default = 20, help='Max path length simulated in the transient in meters. This means that the last element in the transient corresponds to a signal that traveled this distance. The max depth will be 0.5 of this value')
	parser.add_argument('--directonly', default=False, action='store_true', help='Use scene with direct only component.')
	parser.add_argument('-scene_id', type=str, required=True, help='ID of scene to simulate')
	parser.add_argument('-view_id', type=int, default = 0, help='ID of view of scene')
	return parser

def add_transient_sim_args( parser ):
	parser.add_argument('-repetition_freq', type=float, default = 5, help='Repetition/Fundamental frequency of light source in MHz (sets the max depth)')
	parser.add_argument('-n_photons', type=int, default = 1000, help='Total number of signal photons incident on the sensor')
	parser.add_argument('-SBR', type=float, default = None, help='Signal to background ratio. n_ambient_photons = n_photons / SBR. At very high SBR, ambient illumination is basically ignored.')
	parser.add_argument('-n_tbins', type=int, default = 1000, help='Number of time bins to discretize a single period of the time domain signal')
	parser.add_argument('-time_res', type=int, help='Time resolution in picoseconds. If this is specified, n_tbins is ignored')
	parser.add_argument('--add_photon_noise', action='store_true', help='Add noise.', default = False)
	parser.add_argument('-n_mc_samples', type=int, default = 1000, help='Number of monte carlo samples to use when computer expected lifetime error')
	return parser

def add_illumination_sim_args( parser ):
	parser.add_argument('-pulse_width_factor', type=float, default = 1.0, help='Controls pulse width (sigma/stddev) for gaussian pulse. A factor of 1 sets sigma=time_res')
	parser.add_argument('-exp_lambda_factor', type=float, default = None, help='Controls the exponential tail of the pulse. If None, use gaussian pulse')
	return parser

def add_tof_sim_args( parser ):
	add_transient_sim_args(parser)
	add_illumination_sim_args(parser)
	parser.add_argument('-true_depth', type=float, help='Ground truth depth')
	return parser

def add_sptof_sim_mae_args( parser ):
	add_transient_sim_args(parser)
	add_illumination_sim_args(parser)
	parser.add_argument('-n_depths', type=int, help='Number of depths at which the mae is calculated')
	parser.add_argument('-n_signal_levels', type=int, help='Number of signal levels to evaluate')
	parser.add_argument('-n_sbr_levels', type=int, help='Number of SBR levels to evaluate')
	parser.add_argument('-n_freqs', type=int, default=16 , help='Number of Frequencies to use for freq domain depth sensing')
	return parser

def add_hist_analysis_args( parser ):
	add_tof_sim_args( parser )
	parser.add_argument('-hres_factor', type=int, default = 25, help='Determines number of time bins in high-res histogram. n_tbins_hres = hres_factor*n_tbins')
	parser.add_argument('-gtres_factor', type=int, default = 4, help='Determines number of time bins in ground truth histogram, from which hres and lres histograms are derives. n_tbins_gt = gtres_factor*n_tbins_hres')
	parser.add_argument('-n_freqs', type=int, default = 10, help='Determines the number of frequencies to keep from the hres transient, and then use for depth estimation')
	parser.add_argument('-n_mae_depths', type=int, default = 1, help='Number of depths to discretize the mae_depth_range and calculate the MAE for.'  )
	parser.add_argument('-mae_depth_range', nargs='*', required = False, default = [], type=float, help='We compute the MAE for the depths in this range.'  )
	return parser

def add_fd_poisson_noise_analysis_args( parser ):
	add_tof_sim_args( parser )
	return parser

def get_tof_sim_arg_parser( ):
	parser = argparse.ArgumentParser(description='Input argmument parser for single-photon tof simulation')
	add_tof_sim_args( parser )
	return parser


def get_sptof_sim_mae_arg_parser( ):
	parser = argparse.ArgumentParser(description='Input argmument parser for script that calculates the MAE of an sptof setup over the full depth range')
	add_sptof_sim_mae_args( parser )
	return parser

def get_hist_analysis_arg_parser( ):
	parser = argparse.ArgumentParser(description='Input argmument parser for single-photon histogram analysis')
	add_hist_analysis_args( parser )
	return parser

def get_fd_poisson_noise_analysis_arg_parser( ):
	parser = argparse.ArgumentParser(description='Input argmument parser for fourier domain poisson noise analysis')
	add_fd_poisson_noise_analysis_args( parser )
	return parser

def validate_tof_sim_args( args ):
	return True

def validate_sptof_sim_mae_args( args ):
	return True

def validate_hist_analysis_args( args ):
	return True

def validate_fd_poisson_noise_analysis_args( args ):
	return True
