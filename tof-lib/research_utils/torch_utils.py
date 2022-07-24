## Standard Library Imports

## Library Imports
import numpy as np
import torch
import torch.nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports


def normalize_known_range(x, min_val=0., max_val=1.):
	# First normalize signal between 0 and 1, and multiply 2 and subtract 1 to make it -1 to 1
	return (((x - min_val) / (max_val - min_val))*2) - 1 

def softmax_scoring(scores, gt_indeces, beta=300., eps=1, axis=-1):
	'''
		apply softmax to scores to make into probability distribution
		then use the gt_indeces to take a look at the softmax scores of each sample in the +/- eps neightborhood
		return the sum of these scores
	'''
	assert(eps >= 0),'eps should be non-negative'
	softmax_scores = torch.nn.functional.softmax(scores*beta, dim=axis)
	n_scores = (2*eps)+1
	(min_idx, max_idx) = (0, scores.shape[axis])
	for i in range(n_scores):
		offset = -1*eps + i
		indeces = torch.clamp(gt_indeces + offset, min=min_idx, max=max_idx-1)
		selected_scores = softmax_scores.gather(axis, indeces.long().unsqueeze(axis))
	return selected_scores.sum()  

def multi_img_random_hflip(img_list):
	'''
		Apply same random hflip to a list of images
	'''
	if np.random.rand() > 0.5:
		for i in range(len(img_list)):
			img_list[i] = TF.hflip(img_list[i]) 
	return img_list

def multi_img_random_vflip(img_list):
	'''
		Apply same random vflip to a list of images
	'''
	if np.random.rand() > 0.5:
		for i in range(len(img_list)):
			img_list[i] = TF.vflip(img_list[i]) 
	return img_list

def multi_img_crop(img_list, top, left, height, width):
	'''
		Apply same crop to all images
	'''
	for i in range(len(img_list)):
		img_list[i] = TF.crop(img_list[i], top, left, height, width) 
	return img_list

def multi_img_random_crop(img_list, crop_size=(32,32)):
	'''
		Apply same random crop to all images
	'''
	i, j, h, w = T.RandomCrop.get_params(img_list[0], crop_size)
	img_list = multi_img_crop(img_list, i,j,h,w)
	return img_list