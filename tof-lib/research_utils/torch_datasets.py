'''
	Useful custom pytorch dataloaders
'''
## Standard Library Imports
import os

## Library Imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils.io_ops import get_multi_folder_paired_fnames

class MultiFolderPairedNumpyData(torch.utils.data.Dataset):
	'''
		Dataset with pairs of numpy data files stored in different folders in the following way:
			dataset/	
				folder1/f1.npy, 
						f2.npy, 
				
				folder2/f1.npy, 
						f2.npy, 
				
				folder3/f1.npz 
						f2.npz
	'''
	valid_file_ext = ['npy', 'npz']
	def __init__(self, dirpath_list):
		assert(len(dirpath_list)>0), "empty dirpath list"
		self.dirpath_list = dirpath_list
		self.n_dirpaths = len(dirpath_list)
		(self.base_filenames, self.file_ext_per_dirpath) = get_multi_folder_paired_fnames(dirpath_list, self.valid_file_ext)
		self.n_samples = len(self.base_filenames)
	
	def __len__(self):
		return self.n_samples
	
	def __getitem__(self, idx):
		np_data_sample = []
		curr_base_fname = self.base_filenames[idx]
		for i in range(self.n_dirpaths):
			curr_fpath = os.path.join(self.dirpath_list[i], curr_base_fname + '.' + self.file_ext_per_dirpath[i])
			np_data_sample.append(np.load(curr_fpath))
		return (np_data_sample, curr_base_fname)

	def get_sample(self, sample_filename, **kwargs):
		'''
			kwargs are any extra key-word args that __getitem__ may take as input
		'''
		try:
			idx = self.base_filenames.index(sample_filename)
			return self.__getitem__(idx, **kwargs)
		except ValueError:
			print("{} not in datasets".format(sample_filename))
			return None

if __name__=='__main__':
	dirpath1 = '/home/felipe/Dropbox/research_projects/data/synthetic_data_min/data_no-conductors_no-dielectric_automatic/transient_images_120x160_nt-2000'
	dirpath2 = '/home/felipe/Dropbox/research_projects/data/synthetic_data_min/data_no-conductors_no-dielectric_automatic/rgb_images_120x160_nt-2000'
	dirpath3 = '/home/felipe/Dropbox/research_projects/data/synthetic_data_min/data_no-conductors_no-dielectric_automatic/depth_images_120x160_nt-2000'
	dirpaths = [dirpath1,dirpath2,dirpath3]

	dataset = MultiFolderPairedNumpyData(dirpaths)
	loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

	# if we use get_sample we can get the numpy object directyl
	(np_data, fname) = dataset.get_sample(dataset.base_filenames[0])

	for step, data in enumerate(loader):
		# the loader automatically casts everyting as tensor
		(data_sample, fname) = data
		breakpoint()
		print("Loaded: {}".format(fname))
