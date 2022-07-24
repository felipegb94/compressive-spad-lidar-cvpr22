'''
	Test functionality of toflib.tirf.py
	Example run command: 
		run tests/tirf_modelbased_test.py -n_tbins 1000
'''
#### Standard Library Imports
import argparse
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt

#### Local imports
from toflib import input_args_utils
from toflib import tirf, tof_utils
from research_utils.signalproc_ops import get_random_expgaussian_pulse_params  

def generate_model_tirf(n_tbins, n_samples=1):
	(mu, sigma, exp_lambda) = get_random_expgaussian_pulse_params( n=n_tbins, n_samples=n_samples)
	print("Mu = {}, Sigma = {}, lambda = {}".format(mu, sigma, exp_lambda))
	# Generate gaussian and exponentially modified gaussian
	g_tirf = tirf.GaussianTIRF(n_tbins, mu=mu, sigma=sigma)
	expg_tirf = tirf.ExpModGaussianTIRF(n_tbins, mu=mu, sigma=sigma, exp_lambda=exp_lambda )
	return (g_tirf, expg_tirf)

if __name__=='__main__':
	print("---- Running model-based tirf test ----")
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for tirf_test.')
	parser = input_args_utils.add_model_tirf_args(parser)
	args = parser.parse_args()
	#### Test model-based tirf
	n_tbins = args.n_tbins
	(g_tirf, expg_tirf) = generate_model_tirf(n_tbins, n_samples=2)
	# Plot
	plt.clf()
	plt.plot(g_tirf.tirf.transpose(), linewidth=2)
	plt.plot(expg_tirf.tirf.transpose(), linewidth=2)
