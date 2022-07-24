'''
    Useful functions to get the dirpaths where we should read and write data to and from
'''

## Standard Library Imports
import os

## Library Imports

## Local Imports
from .research_utils import io_ops

def load_io_dirpaths_json(file_dirpath='.'):
    return io_ops.load_json(os.path.join(file_dirpath, 'io_dirpaths.json'))

def get_dirpath( dirpath_id, io_dirpath_json_file_dir='.' ):
    dirpaths_json = load_io_dirpaths_json(file_dirpath=io_dirpath_json_file_dir)
    return os.path.join(io_dirpath_json_file_dir, dirpaths_json[dirpath_id])