'''
	Useful function based on scipy
'''
#### Standard Library Imports

#### Library imports
import numpy as np
import scipy
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from .shared_constants import *


def npz2mat(npz_fpath):
    '''
        Load an uncompressed .npz file and save it as a .mat MATLAB file
    '''
    from scipy import io
    data_dict = np.load(npz_fpath)
    mat_fpath = npz_fpath.replace('.npz', '.mat')
    io.savemat(mat_fpath, data_dict)