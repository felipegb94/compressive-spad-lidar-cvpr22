'''
	Dataloaders based on the tirf.py and tirf_scene.py objects
'''
## Standard Library Imports

## Library Imports
import numpy as np
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.shared_constants import *
from research_utils.signalproc_ops import get_random_gaussian_pulse_params
from toflib import tirf

class GaussianTIRFData(torch.utils.data.Dataset):
	'''
		Simulate $n_samples gaussians with mean distributed between 0-$n_tbins 
		Everytime simulate_new_tirf is called a new set of gaussians corrupted by poisson noise is generated
	'''

	def __init__(self, n_tbins, n_samples=1, mode='random', device='cpu', min_max_nphotons_exp=None, min_max_sbr_exp=None):
		self.padding = int(0.01*n_tbins)
		sigma=1
		if(mode is 'random'):
			(gt_mus, gt_sigmas) = get_random_gaussian_pulse_params(n=n_tbins-2*self.padding, min_max_sigma=(sigma,sigma), n_samples=n_samples)
			gt_mus += self.padding
		else:
			gt_mus = np.linspace(self.padding, n_tbins-self.padding, n_samples)
			gt_sigmas = np.ones_like(gt_mus)*sigma
		self.n_samples = n_samples
		self.pulse_obj = tirf.GaussianTIRF(n_tbins, mu=gt_mus, sigma=gt_sigmas)
		self.device = device
		self.tirf = torch.tensor(self.pulse_obj.tirf, device=self.device).type(torch.float32)
		self.gt_mus = torch.tensor(gt_mus, device=self.device).type(torch.float32)
		self.gt_mu_indeces = self.gt_mus.round().type(torch.int)
		self.gt_sigmas = torch.tensor(gt_sigmas, device=self.device).type(torch.float32)
		# 
		if(min_max_nphotons_exp is None): self.min_max_nphotons_exp=(0,2)
		else: self.min_max_nphotons_exp=min_max_nphotons_exp
		if(min_max_sbr_exp is None): self.min_max_sbr_exp=(-1,1)
		else: self.min_max_sbr_exp=min_max_sbr_exp

	def __len__(self):
		return self.n_samples

	# def simulate_new_tirf(self, min_max_nphotons_exp=(1,2), min_max_sbr_exp=(-1,1)):
	# def simulate_new_tirf(self, min_max_nphotons_exp=(0.5,1), min_max_sbr_exp=(-1,0)):
	def simulate_new_tirf(self, min_max_nphotons_exp=None, min_max_sbr_exp=None):
		if(min_max_nphotons_exp is None): min_max_nphotons_exp=self.min_max_nphotons_exp
		if(min_max_sbr_exp is None): min_max_sbr_exp=self.min_max_sbr_exp
		nphotons_exp = np.random.uniform(low=min_max_nphotons_exp[0], high=min_max_nphotons_exp[1], size=self.n_samples)
		nphotons = np.round(np.power(10,nphotons_exp)).astype(int)
		sbr_exp = np.random.uniform(low=min_max_sbr_exp[0], high=min_max_sbr_exp[1], size=self.n_samples)
		sbr = np.power(10, sbr_exp)
		# Set sbr
		self.pulse_obj.set_sbr(sbr)
		## Simulate noisy tirf whose noiseless sum is nphotons
		self.np_tirf = self.pulse_obj.simulate_n_photons(nphotons).squeeze()
		## Cast to torch tensor, and normalize to sum = 1
		self.tirf = torch.tensor(self.np_tirf, device=self.device).type(torch.float32)
		self.tirf = self.tirf / (self.tirf.sum(dim=-1, keepdim=True) + EPSILON)

	def __getitem__(self, idx):
		return (self.tirf[idx], self.gt_mus[idx], self.gt_sigmas[idx], self.gt_mu_indeces[idx])
