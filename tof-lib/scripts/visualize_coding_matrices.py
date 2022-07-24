#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils import plot_utils, np_utils, io_ops
from toflib import coding

def plot_c_matrix(c):
    img = plt.imshow(c.get_pretty_C(), cmap='gray', vmin=-1, vmax=1)
    plot_utils.remove_yticks()
    plot_utils.remove_xticks()
    # plot_utils.set_cbar(img)



out_dirpath = '../../../results/compressive_sp_histograms_slides'

n = 256
plt.close('all')
## Fourier 1 freq
c = coding.TruncatedFourierCoding(n, n_freqs=1, include_zeroth_harmonic=False)
fname = 'fourier_nfreqs-1_n-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)

## Fourier 4 freq
c = coding.TruncatedFourierCoding(n, n_freqs=4, include_zeroth_harmonic=False)
fname = 'fourier_nfreqs-4_n-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)

## 4 bit gray coding
c = coding.GrayCoding(n, n_bits=4)
fname = 'gray_nbits-4_n-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)

## 7 bit gray coding
c = coding.GrayCoding(n, n_bits=7)
fname = 'gray_nbits-7_n-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)

## 8 bit gray coding
c = coding.GrayCoding(n, n_bits=8)
fname = 'gray_nbits-8_n-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)

## identity  coding
c = coding.GatedCoding(n, n)
fname = 'gated_ngates-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)

## gated coding
print(n)
c = coding.GatedCoding(n, n_gates=16)
fname = 'gated_ngates-16_n-{}'.format(int(n))
np.save(os.path.join(out_dirpath, fname), c.C)
plt.figure()
plot_c_matrix(c)
plot_utils.save_currfig_png(out_dirpath, fname)
plt.title(fname, fontsize=16)


# ## haar coding
# c = coding.HaarCoding(n, n_lvls=4)
# fname = 'haar_nlvls-4_n-{}'.format(int(n))
# plt.figure()
# img = plt.imshow(c.get_pretty_C(), cmap='gray', vmin=-1, vmax=1)
# # plot_utils.remove_yticks()
# plot_utils.set_cbar(img)
# plot_utils.save_currfig_png(out_dirpath, fname)
# plt.title(fname, fontsize=16)

# ## haar coding
# c = coding.HaarCoding(n, n_lvls=8)
# fname = 'haar_nlvls-8_n-{}'.format(int(n))
# plt.figure()
# img = plt.imshow(c.get_pretty_C(), cmap='gray', vmin=-1, vmax=1)
# # plot_utils.remove_yticks()
# plot_utils.set_cbar(img)
# plot_utils.save_currfig_png(out_dirpath, fname)
# plt.title(fname, fontsize=16)



