#### Standard Library Imports
import os
import glob
import json
import re
import pickle

#### Library imports
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


def load_json( json_filepath ):
	assert( os.path.exists( json_filepath )), "{} does not exist".format( json_filepath )
	with open( json_filepath, "r" ) as json_file: 
		return json.load( json_file )

def write_json( json_filepath, input_dict ):
	assert(isinstance(input_dict, dict)), "write_json only works if the input_dict is of type dict"
	with open(json_filepath, 'w') as output_file: 
		json.dump(input_dict, output_file, indent=4)

def save_object(obj, filepath):
	with open(filepath, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filepath): 
	with open(filepath, 'rb') as input_pickle_file:
		return pickle.load(input_pickle_file)

def simple_grep( filepath, str_to_search, n_lines=-1 ):
	'''
		Search text file and return the first n_lines containing that string
		If the line contains the string multiple times, it is only counted as a single line 
	'''
	assert(os.path.exists(filepath)), "{} does not exist".format(filepath)
	assert(n_lines >= -1), "n_lines needs to be -1 OR a non-negative integer. If it is -1 then we return all lines".format(filepath)
	f = open(filepath, "r")
	lines_with_str = []
	n_lines_found = 0
	for line in f:
		# Return if we found all lines asked to. If n_lines ==-1 then we just continue searching for all lines
		if((n_lines_found >= n_lines) and (n_lines >= 0)): return lines_with_str
		# search if line contains string, and save the line if it does
		if re.search(str_to_search, line):
			n_lines_found += 1
			lines_with_str.append(line.split('\n')[0]) # Remove the new line characted if there is any
	return lines_with_str

def get_dirnames_in_dir(dirpath, str_in_dirname=None):
	'''
		Output all the dirnames inside of dirpath.
		If str_in_dirname is given, only return the dirnames containing that string
	'''
	assert(os.path.exists(dirpath)), "Input dirpath does not exist"
	all_dirnames = next(os.walk(dirpath))[1]
	# If no string pattern is given return all dirnames
	if(str_in_dirname is None): return all_dirnames
	filtered_dirnames = []
	for curr_dirname in all_dirnames:
		if(str_in_dirname in curr_dirname):
			filtered_dirnames.append(curr_dirname)
	return filtered_dirnames

def get_filepaths_in_dir(dirpath, match_str_pattern=None, only_filenames=False, keep_ext=True):
	'''
		Return a list of all filepaths inside a directory that contain the match_str_pattern.
		If we only want the filenames and not the filepath, set only_filenames=True
	'''
	assert(os.path.exists(dirpath)), "Input dirpath does not exist"
	if(match_str_pattern is None): all_matching_filepaths = glob.glob(dirpath)
	else: all_matching_filepaths = glob.glob(os.path.join(dirpath, '*' + match_str_pattern + '*'))
	filepaths = []
	for fpath in all_matching_filepaths:
		if(os.path.isfile(fpath)): 
			# if not file ext, remove it
			if(not keep_ext): fpath = os.path.splitext(fpath)[0]
			# if only_filanemaes, remove the dirpath and only return the filename
			if(only_filenames): filepaths.append(os.path.basename(fpath))
			else: filepaths.append(fpath)
	return filepaths

def get_multi_folder_paired_fnames(dirpath_list, valid_file_ext_list):
	'''
		Go through each folder in dirpath_list, get all filenames with the file extension in valid_file_ext_list.
		Then check that across all folders you can find paired filenames. 
			If so, then return the filenames
			If not, generate error
		Example:
			dirpath_list = ['dir1', 'dir2', 'dir3']
			valid_file_ext_list = ['npy', 'npz']
			We have the following files:
				folder1/f1.npy, 
				folder1/f2.npy, 
				
				folder2/f1.npy, 
				folder2/f2.npy, 
				folder3/xkcd.txt 
				
				folder3/f1.npz 
				folder3/f2.npz 
				folder3/random.png
			this function will return:
			paired_filenames = ['f1', 'f2']
			ext_per_dirpath = ['npy', 'npy', 'npz'] 
		Why does this function exist? Because it is useful to organize small datasets in this way.
	'''
	assert(len(dirpath_list)>0), "empty dirpath list"
	n_dirpaths = len(dirpath_list)
	filenames_per_dirpath = []
	file_ext_per_dirpath = []
	all_filenames = []
	n_filenames_per_dirpath = []
	for i in range(n_dirpaths):
		curr_dirpath = dirpath_list[i]
		filenames_in_dir = []
		for file_ext in valid_file_ext_list:
			curr_filenames = get_filepaths_in_dir(curr_dirpath, match_str_pattern='*.'+file_ext, only_filenames=True, keep_ext=False)
			if(len(curr_filenames) != 0): 
				# We should only enter this condition once per deirpath
				filenames_in_dir += curr_filenames
				file_ext_per_dirpath.append(file_ext)
		filenames_per_dirpath.append(filenames_in_dir)
		all_filenames += filenames_in_dir
		n_filenames_per_dirpath.append(len(filenames_in_dir))
	# Check that all dirpath have the same number of filemaes
	assert(len(set(n_filenames_per_dirpath)) == 1), "Check that all dirpaths have the same number of files"
	n_samples = n_filenames_per_dirpath[0]
	# Check that the filenames within each folder are the same 
	# i.e., folder1/f1.npy, folder2/f1.npy , folder3/f1.npz 
	paired_filenames = list(set(all_filenames))
	assert(len(paired_filenames) == n_samples), "Filenames within each folder should match (folder1/f1.npy, folder2/f1.npy , folder3/f1.npz)"
	# Check that we only have one file extension per directory
	assert(len(file_ext_per_dirpath) == n_dirpaths)
	return (paired_filenames, file_ext_per_dirpath)

def get_string_from_file(filepath):
	f = open(filepath)
	path = f.read().replace('\n','')
	f.close()
	return path