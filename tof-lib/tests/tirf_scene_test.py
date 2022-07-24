'''
	Test functionality of toflib.tirf.py
	Example run command:
		run tests/tirf_scene_test.py -data_dirpath ./sample_data -scene_id vgroove
'''
#### Standard Library Imports
import argparse
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt

#### Local imports
from toflib import tirf_scene
from toflib import tof_utils, input_args_utils
from tirf_databased_test import generate_data_tirfs

if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for scene_tirf_test.')
	parser = input_args_utils.add_tirf_scene_args(parser)
	args = parser.parse_args()
	# Parse input args
	scene_id = args.scene_id
	data_dirpath = args.data_dirpath
	sbr = 1 
	# get data-based tirfs
	(transient_tirf, depth_tirf) = generate_data_tirfs(scene_id, data_dirpath)

	rgb_img_fpath = os.path.join(data_dirpath, 'rgb_data/{}.npy'.format(scene_id))
	plt.clf()
	if(os.path.exists(rgb_img_fpath)):
		depth_based_tof_scene = tirf_scene.ToFScene(depth_tirf, ambient_img_filepath=rgb_img_fpath, sbr=sbr)
		transient_based_tof_scene = tirf_scene.ToFScene(transient_tirf, ambient_img_filepath=rgb_img_fpath, sbr=sbr)
		plt.subplot(2,2,1)
		plt.imshow(depth_based_tof_scene.ambient_img)
		plt.title("Depth Based Ambient Img")
		plt.colorbar()
		plt.subplot(2,2,2)
		plt.imshow(transient_based_tof_scene.ambient_img)
		plt.title("Transient Based Ambient")
		plt.colorbar()
	else:
		depth_based_tof_scene = tirf_scene.ToFScene(depth_tirf)
		transient_based_tof_scene = tirf_scene.ToFScene(transient_tirf)
	plt.subplot(2,2,3)
	plt.imshow(depth_based_tof_scene.tirf_img.sum(axis=-1))
	plt.title("Depth Based TIRF Intensity")
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.imshow(transient_based_tof_scene.tirf_img.sum(axis=-1).astype(np.float))
	plt.title("Transient Based TIRF Intensity")
	plt.colorbar()

	print("Transient Based ToF Scene:")
	print("    * Mean Signal = {}".format(transient_based_tof_scene.tirf_img.sum(axis=-1).mean()))
	print("    * Mean Ambient = {}".format(transient_based_tof_scene.ambient_img.mean()))
