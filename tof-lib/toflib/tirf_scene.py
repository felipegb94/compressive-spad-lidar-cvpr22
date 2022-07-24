'''
	Base class for temporal impulse response functions
'''
## Standard Library Imports
from abc import ABC, abstractmethod

## Library Imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from toflib.tirf import TemporalIRF
from toflib import tof_utils
from research_utils import np_utils
from research_utils.timer import Timer
from research_utils.shared_constants import *

class ToFScene(ABC):
	'''
		Class that holds the information for a scene with a temporal impulse response associated to it.
		Important elements of scene:
			* Data-based tirf object. This can be generated from a depth image or a transient image
			* ambient_img_filepath. If available this image will be an RGB or monochrome image of the scene
			* mean_sbr: Signal-to-background ratio of the scene. Only useful if we have an ambient image. We use this number to scale 
			the ambient image so that we simulate the correct mean_sbr
	'''
	def __init__(self, tirf_obj, ambient_img=None, mean_sbr=None, verbose=False):
		# if we want print statement for debugging
		self.verbose=verbose
		# Store tirf object
		assert(isinstance(tirf_obj, TemporalIRF)), "Input object needs to be a TemporalIRF class or inherit from that class"
		self.tirf_obj = tirf_obj
		self.tirf_img = self.tirf_obj.tirf
		# set number of signal, ambient, and total photons
		self.mean_signal_photons = self.get_mean_signal_photons()
		self.mean_sbr = mean_sbr
		if(self.mean_sbr is None): self.mean_ambient_photons = 0
		else: self.mean_ambient_photons = self.mean_signal_photons / self.mean_sbr
		self.mean_nphotons = self.mean_signal_photons + self.mean_ambient_photons
		# Load ambient and scale it
		self.set_ambient_img(ambient_img)
		self.update_mean_sbr(mean_sbr)

	def get_mean_signal_photons(self): 
		return self.tirf_img.sum(axis=-1).mean()

	def __update_mean_signal_photons(self, mean_signal_photons):
		'''
			should only be able to call this function from update_mean_nphotons
		'''
		self.mean_signal_photons = mean_signal_photons
		curr_mean_signal_photons = self.get_mean_signal_photons()
		self.tirf_img *= (mean_signal_photons / curr_mean_signal_photons)

	def update_mean_nphotons(self, mean_nphotons=None):
		'''
			We use the following two expressions to make all updates
				mean_nphotons = mean_signal + mean_ambient
				mean_sbr = mean_signal / mean_ambient
		'''
		# If not None update them
		if(not (mean_nphotons is None)): self.mean_nphotons = mean_nphotons
		if(self.mean_sbr is None):
			new_mean_ambient_photons = 0
			new_mean_signal_photons = self.mean_nphotons
		else:
			new_mean_ambient_photons = self.mean_nphotons / (1. + self.mean_sbr)
			new_mean_signal_photons = self.mean_nphotons / (1. + (1./self.mean_sbr)) 
		self.__update_mean_signal_photons(new_mean_signal_photons)
		self.__update_mean_ambient_photons(new_mean_ambient_photons)

	def update_mean_sbr(self, mean_sbr):
		'''
			We use the following two expressions to make all updates
				mean_nphotons = mean_signal + mean_ambient
				mean_sbr = mean_signal / mean_ambient
		'''
		self.mean_sbr = mean_sbr
		self.update_mean_nphotons(None) # do not update the total photon count, just the ratios
	
	def __update_mean_ambient_photons(self, mean_ambient_photons):
		'''
			Scale the ambient image given the mean_sbr
		'''
		self.mean_ambient_photons = mean_ambient_photons
		if((self.ambient_img is None) or (self.mean_sbr is None)): 
			if(self.verbose):
				print("Not scaling ambient because mean_sbr is None OR ambient_img is None")
		else:
			# mean_signal_photons = self.get_mean_signal_photons()
			curr_mean_ambient_photons = self.ambient_img.mean() 
			# target_mean_ambient_photons = mean_signal_photons / self.mean_sbr
			# Scale ambient image to have the correct scale with respect to tirf
			self.ambient_img *= (self.mean_ambient_photons / curr_mean_ambient_photons)
			self.sbr_img = self.tirf_img.sum(axis=-1) / (self.ambient_img + EPSILON)
			self.tirf_obj.set_sbr(self.sbr_img) 

	def set_ambient_img(self, ambient_img):
		self.sbr_img = None
		if(ambient_img is None): 
			print("No ambient img. Setting it to empty")
			self.ambient_img = None
		else:
			self.ambient_img = np.array(ambient_img)
			is_rgb = self.ambient_img.ndim == 3
			is_monochrome = self.ambient_img.ndim == 2
			if(is_rgb):
				smallest_dim = np.argmin(self.ambient_img.shape)
				if(smallest_dim != (self.ambient_img.ndim-1)):
					print("Warning: Smallest axis in ambient image is expected to be the RGB axis which should be the last dim")
				if(self.verbose): print("ambient is RGB storing only the red channel")
				self.ambient_img = self.ambient_img[..., 0]
			elif(is_monochrome): print("ambient is monochrome")
			else: assert(False), "Input ambient image should have 2 or 3 dims (i.e., be monochrome or rgb)"

	# def __dtof_sim_old(self, mean_signal_photons=None, mean_sbr=None):
	# 	'''
	# 		For a given a scene with a mean_signal_photons and mean_sbr, add poisson noise to transient
	# 	'''
	# 	if(not (mean_signal_photons is None)): self.update_mean_signal_photons(mean_signal_photons)
	# 	if(not (mean_sbr is None)): self.update_mean_sbr(mean_sbr)
	# 	# Get noisy transient
	# 	tirf_tmp = self.tirf_obj.simulate_n_signal_photons().squeeze() 
	# 	# There may be pixels with no transient signal but with ambient signal, we need to deal with them separately
	# 	tirf_tmp[self.tirf_obj.nosignal_mask] += self.ambient_img[self.tirf_obj.nosignal_mask][:,np.newaxis] / self.tirf_img.shape[-1]
	# 	tirf_tmp[self.tirf_obj.nosignal_mask] = tof_utils.add_poisson_noise(tirf_tmp[self.tirf_obj.nosignal_mask]).squeeze()
	# 	return tirf_tmp

	# def dtof_sim_old(self, mean_signal_photons=None, mean_sbr=None):
	# 	'''
	# 		Call dtof_sim_old with a list of mean_signal_photons and mean_sbr
	# 	'''
	# 	if((mean_sbr is None) or (mean_signal_photons is None)):
	# 		return self.__dtof_sim_old()[np.newaxis,:]
	# 	mean_signal_photons_v = np_utils.to_nparray(mean_signal_photons).flatten()
	# 	mean_sbr_v = np_utils.to_nparray(mean_sbr).flatten()
	# 	n_simulations = len(mean_signal_photons_v)
	# 	assert(len(mean_signal_photons_v) == len(mean_sbr_v)), "input mean_signal_photons and mean_sbr dims should match"
	# 	transient_img_sim = np.zeros((n_simulations,) + self.tirf_img.shape, dtype=self.tirf_img.dtype)	
	# 	for i in range(n_simulations):
	# 		transient_img_sim[i,:] = self.__dtof_sim_old(mean_signal_photons=mean_signal_photons_v[i],mean_sbr=mean_sbr_v[i])
	# 	return transient_img_sim

	def __dtof_sim(self, mean_nphotons=None, mean_sbr=None):
		'''
			For a given a scene with a mean_nphotons and mean_sbr, add poisson noise to transient
		'''
		if(not (mean_nphotons is None)): self.update_mean_nphotons(mean_nphotons)
		if(not (mean_sbr is None)): self.update_mean_sbr(mean_sbr)
		# Get noisy transient. (time consuming step of this function)
		tirf_tmp = self.tirf_obj.simulate_n_photons().squeeze()
		# There may be pixels with no transient signal but with ambient signal, we need to deal with them separately
		tirf_tmp[self.tirf_obj.nosignal_mask] += self.ambient_img[self.tirf_obj.nosignal_mask][:,np.newaxis] / self.tirf_img.shape[-1]
		tirf_tmp[self.tirf_obj.nosignal_mask] = tof_utils.add_poisson_noise(tirf_tmp[self.tirf_obj.nosignal_mask]).squeeze()
		return tirf_tmp

	def dtof_sim(self, mean_nphotons=None, mean_sbr=None):
		'''
			Call dtof_sim with a list of mean_nphotons and mean_sbr
		'''
		if((mean_sbr is None) and (mean_nphotons is None)):
			return self.__dtof_sim()[np.newaxis,:]
		mean_nphotons_v = np_utils.to_nparray(mean_nphotons).flatten()
		mean_sbr_v = np_utils.to_nparray(mean_sbr).flatten()
		n_simulations = len(mean_nphotons_v)
		assert(len(mean_nphotons_v) == len(mean_sbr_v)), "input mean_nphotons and mean_sbr dims should match"
		transient_img_sim = np.zeros((n_simulations,) + self.tirf_img.shape, dtype=self.tirf_img.dtype)	
		for i in range(n_simulations):
			transient_img_sim[i,:] = self.__dtof_sim(mean_nphotons=mean_nphotons_v[i],mean_sbr=mean_sbr_v[i])
		return transient_img_sim