#### Standard Library Imports
import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils import io_ops, torch_utils
from research_utils.timer import Timer
from research_utils.torch_datasets import MultiFolderPairedNumpyData
from toflib import tirf, tirf_scene

class FlashLidarSceneData(MultiFolderPairedNumpyData):
	'''
		The FlashLidarSceneData is a special type Dataset object.
		The __getitem__ function can be expensive because it loads a transient image which is data volume.
		Therefore, during training it is not recommended to call __getitem__ many times.
		Instead we recommend calling it once, and using the same image a few times before calling it again
		
	'''

	def __init__(self, transient_dirpath, rgb_dirpath, depth_dirpath, min_max_mean_signal_exp=(0.5,2), min_max_mean_sbr_exp=(-1,1)):
		super(FlashLidarSceneData, self).__init__([transient_dirpath, rgb_dirpath, depth_dirpath])
		self.curr_transient_obj = None
		self.curr_scene_obj = None
		# Parameters to simulate random SNR during loading
		self.min_max_mean_signal_exp=min_max_mean_signal_exp
		self.min_max_mean_sbr_exp=min_max_mean_sbr_exp
		# Keep trakc of the mean signal and sbr used 
		self.curr_mean_signal=None
		self.curr_mean_sbr=None
		# Set parameters for data transformations
		self.crop_size = (64,64)

	def unpack_data_sample(self, data_sample):
		transient_img = data_sample[0]['arr_0'].astype(np.float32)
		rgb_img = data_sample[1].astype(np.float32)
		depth_img = data_sample[2].astype(np.float32)
		return (transient_img, rgb_img, depth_img)

	def simulate_flash_lidar_scene(self, transient_img, rgb_img, mean_signal=None, mean_sbr=None, simulate=True):
		self.curr_transient_obj = tirf.TemporalIRF(transient_img)
		self.curr_scene_obj = tirf_scene.ToFScene(self.curr_transient_obj, rgb_img)
		if(simulate):
			# If no sbr or signal are given simulate a random one
			if(mean_signal is None):
				mean_signal_exp = np.random.uniform(low=self.min_max_mean_signal_exp[0], high=self.min_max_mean_signal_exp[1])
				mean_signal = np.power(10., mean_signal_exp)
			if(mean_sbr is None):
				mean_sbr_exp = np.random.uniform(low=self.min_max_mean_sbr_exp[0], high=self.min_max_mean_sbr_exp[1])
				mean_sbr = np.power(10., mean_sbr_exp)
			self.curr_mean_signal = mean_signal
			self.curr_mean_sbr = mean_sbr
			transient_img_sim = self.curr_scene_obj.dtof_sim(mean_signal_photons=mean_signal, mean_sbr=mean_sbr)
		else: 
			self.curr_mean_signal = None
			self.curr_mean_sbr = None
			transient_img_sim = transient_img
		return transient_img_sim.squeeze()

	def __getitem__(self, idx, mean_signal=None, mean_sbr=None, simulate=True):
		# Get data
		(data_sample, curr_base_fname) = super(FlashLidarSceneData, self).__getitem__(idx)
		(transient_img, rgb_img, depth_img) = self.unpack_data_sample(data_sample)
		noisy_transient_img = self.simulate_flash_lidar_scene(transient_img, rgb_img, mean_signal=mean_signal, mean_sbr=mean_sbr, simulate=simulate)
		clean_transient_img = self.curr_scene_obj.tirf_obj.tirf #  tirf without any scaling
		ambient_img = self.curr_scene_obj.ambient_img
		# Move spatial dims to the last 2 dims because pytorch vision ops usually require this
		noisy_transient_img = np.moveaxis(noisy_transient_img, -1, -3)
		clean_transient_img = np.moveaxis(clean_transient_img, -1, -3)
		rgb_img = np.moveaxis(rgb_img, -1, -3)
		return ([noisy_transient_img, clean_transient_img, ambient_img, rgb_img, depth_img], curr_base_fname)

	def transform_img_list(self, img_list):
		'''
			Apply same set of transforms to a list of images. Assumes the last 2 dims are the nrows and ncols
		'''
		# Random crop
		img_list = torch_utils.multi_img_random_crop(img_list, self.crop_size)
		# Random horizontal flipping
		img_list = torch_utils.multi_img_random_hflip(img_list)
		# Random vertical flipping
		img_list = torch_utils.multi_img_random_vflip(img_list)
		return img_list

if __name__=='__main__':
	# Get dirpaths for data
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	transient_images_dirpath = io_dirpaths['transient_images_dirpath']
	rgb_images_dirpath = io_dirpaths['rgb_images_dirpath']
	depth_images_dirpath = io_dirpaths['depth_images_dirpath']    

	fl_dataset = FlashLidarSceneData(transient_images_dirpath, rgb_images_dirpath, depth_images_dirpath)

	## Example of how to load a single scene
	sample_filename = fl_dataset.base_filenames[1]
	print("Loading: {}".format(sample_filename))
	(data_sample, fname) = fl_dataset.get_sample(sample_filename)
	noisy_transient_img = data_sample[0]
	clean_transient_img = data_sample[1] 
	ambient_img = data_sample[2]
	rgb_img = data_sample[3]
	depth_img = data_sample[4] 
	(nr, nc) = depth_img.shape


	## Example of how to iterate through all the scenes
	n_workers = 8
	batch_size = 1
	loader = DataLoader(fl_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

	from research_utils.improc_ops import gamma_tonemap
	from research_utils.timer import Timer
	import matplotlib.pyplot as plt

	with Timer("Time to iterate through dataset with {} workers and batch {}".format(n_workers, batch_size)):
		for step, data in enumerate(loader):
			(data_sample, fname) = data
			print("Loaded: {}, step: {}".format(fname, step))
			# Apply transformations
			## We do it outside the dataloader because we may want to apply transforms more than once to the same data_saple
			data_sample = fl_dataset.transform_img_list(data_sample)
			# Unpack images from dataset
			noisy_transient_img = data_sample[0]
			clean_transient_img = data_sample[1] 
			ambient_img = data_sample[2]
			rgb_img = data_sample[3]
			depth_img = data_sample[4] 
			plt.clf()
			plt.imshow(gamma_tonemap(rgb_img[0,:].cpu().numpy()).transpose((-2,-1,-3)))
			# plt.imshow(gamma_tonemap(ambient_img[0,:].cpu().numpy()))
			# plt.imshow(gamma_tonemap(noisy_transient_img.sum(axis=-3)[0,:].cpu().numpy()))
			plt.pause(0.1)