#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

def get_ax_if_none(ax):
	if(ax is None): return plt.gca()
	else: return ax

def save_currfig( dirpath = '.', filename = 'curr_fig', file_ext = 'png', use_imsave=False  ):
	# Create directory to store figure if it does not exist
	os.makedirs(dirpath, exist_ok=True)
	# Pause to make sure plot is fully rendered and not warnings or errors are thown
	plt.pause(0.02)
	# If filename contains file extension then ignore the input file ext
	# Else add the input file etension
	if('.{}'.format(file_ext) in  filename): filepath = os.path.join(dirpath, filename)
	else: filepath = os.path.join(dirpath, filename) + '.{}'.format(file_ext)
	plt.savefig(filepath, 
				dpi=None, 
				# facecolor='w', 
				# edgecolor='w',
				# orientation='portrait', 
				# papertype=None, 
				transparent=True, 
				bbox_inches='tight', 
				# pad_inches=0.1,
				# metadata=None 
				format=file_ext
				)

def save_img(data, out_dirpath, out_filename, file_ext='png'):
	# if(max_val is None): max_val = data.mean() + 3*data.std()
	out_filepath = os.path.join(out_dirpath, 'image_'+out_filename+'.'+file_ext)
	plt.imsave(out_filepath, data)

def save_currfig_png( dirpath = '.', filename = 'curr_fig'  ): 
	save_currfig( dirpath = dirpath, filename = filename, file_ext = 'png' )

def save_ax(ax = None, dirpath = '.', filename = 'curr_fig', file_ext = 'png'):
	ax = get_ax_if_none(ax)
	plt.sca(ax)
	save_currfig(dirpath=dirpath, filename=filename, file_ext=file_ext)

def save_ax_png(ax=None, dirpath = '.', filename = 'curr_fig'  ): 
	save_ax(ax=ax, dirpath = dirpath, filename = filename, file_ext = 'png' )

def save_rgb( dirpath = '.', filename = 'curr_rgb', file_ext='svg', rm_ticks=True):
	if(rm_ticks): remove_ticks()
	save_currfig(dirpath = dirpath, filename = filename, file_ext=file_ext)

def plot_and_save_rgb(data, out_dirpath='./', out_filename='out_img', min_val=None, max_val=None, add_colorbar=False, rm_ticks=True, cbar_orientation='vertical', file_ext='png', save_fig=False, add_title=False, use_imsave=False):
	assert(data.ndim == 2 or data.ndim == 3), "Input data should have 2 dimensions"
	if(data.ndim == 3): assert(data.shape[-1] == 3 or data.shape[-1] == 1), "last image dimension needs to be 3 for RGB and 1 for mono image"
	if(min_val is None): min_val = np.min(data)
	if(max_val is None): max_val = np.max(data)
	(fig, ax) = plt.subplots()
	img = ax.imshow(data, vmin=min_val, vmax=max_val)
	if(rm_ticks): remove_ticks()
	if(add_colorbar): set_cbar(cbar_orientation) 
	if(add_title): plt.title(out_filename)
	if(save_fig): 	
		save_rgb(out_dirpath, out_filename, file_ext=file_ext, rm_ticks=False)
		if(use_imsave): save_img(data, out_dirpath, out_filename, min_val=min_val, max_val=max_val, file_ext=file_ext)

def set_cbar(img, cbar_orientation='vertical', fontsize=14):
	fig = plt.gcf()
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	if(cbar_orientation == 'vertical'): 
		# cax = divider.append_axes('right', size='4%', pad=0.05)
		cax = divider.append_axes('right', size='10%', pad=0.05)
	else: cax = divider.append_axes('bottom', size='7%', pad=0.05)
	cb = fig.colorbar(img, cax=cax, orientation=cbar_orientation)
	cb.ax.tick_params(labelsize=fontsize)
	plt.sca(ax) # Set axis back to what it was
	# fig.colorbar(img, orientation=cbar_orientation, ax=ax)

def draw_histogram(x, height, draw_line=True):
	curr_ax = plt.gca()
	delta_x = x[1] - x[0]
	curr_ax.bar(x, height, align='center', alpha=0.5, width=delta_x)
	if(draw_line):
		curr_ax.plot(x, height, linewidth=2)

def remove_ticks(ax = None):
	ax = get_ax_if_none(ax)
	ax.tick_params(
		axis='both',          # changes apply to the x-axis and y-axis
		which='both',      # both major and minor ticks are affected
		bottom=False, top=False, left=False, right=False,      # ticks along the bottom edge are off
		labelbottom=False, labeltop=False, labelleft=False, labelright=False
		) # labels along the bottom edge are off

def remove_xticks(ax = None):
	ax = get_ax_if_none(ax)
	ax.tick_params(
		axis='x',          # changes apply to the x-axis and y-axis
		which='both',      # both major and minor ticks are affected
		bottom=False, top=False,      # ticks along the bottom edge are off
		labelbottom=False, labeltop=False
		) # labels along the bottom edge are off

def remove_yticks(ax = None):
	ax = get_ax_if_none(ax)
	ax.tick_params(
		axis='y',          # changes apply to the x-axis and y-axis
		which='both',      # both major and minor ticks are affected
		left=False, right=False,      # ticks along the bottom edge are off
		labelleft=False, labelright=False
		) # labels along the bottom edge are off

def set_ticks(ax = None, fontsize=12):
	ax = get_ax_if_none(ax)
	ax.tick_params(
		axis='both',          # changes apply to the x-axis and y-axis
		which='both',      # both major and minor ticks are affected
		labelsize=fontsize
		) # labels along the bottom edge are off

def set_xtick_labels(x_max_val, x_labels, ax=None ):
	ax = get_ax_if_none(ax)
	ax.set_xticks(np.linspace(0, x_max_val, len(x_labels)))
	ax.set_xticklabels(x_labels)

def set_ytick_labels(y_max_val, y_labels, ax=None ):
	ax = get_ax_if_none(ax)
	ax.set_yticks(np.linspace(0, y_max_val, len(y_labels)))
	ax.set_yticklabels(y_labels)

def set_plot_border_visibility(top_visibility=True, bottom_visibility=True, right_visibility=True, left_visibility=True):
	ax = plt.gca()
	ax.spines['top'].set_visible(top_visibility)
	ax.spines['bottom'].set_visible(bottom_visibility)
	ax.spines['right'].set_visible(right_visibility)
	ax.spines['left'].set_visible(left_visibility)

def remove_box():
	plt.box(False)

def set_axis_linewidth(ax=None, width=1):
	ax = get_ax_if_none(ax)
	# Set with of axis lines
	for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(width)
	# Set with of ticks
	ax.xaxis.set_tick_params(width=0.75*width)
	ax.yaxis.set_tick_params(width=0.75*width)

def set_xy_box(linewidth=None):
	gca = plt.gca()
	gca.spines["right"].set_visible(False)
	gca.spines["top"].set_visible(False)

def set_x_box():
	gca = plt.gca()
	gca.spines["right"].set_visible(False)
	gca.spines["top"].set_visible(False)
	gca.spines["left"].set_visible(False)
	remove_yticks()

def set_x_arrow(ax=None):
	ax = get_ax_if_none(ax)
	ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)

def set_y_arrow(ax=None):
	ax = get_ax_if_none(ax)
	ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

def set_xy_arrow(ax=None):
	set_x_arrow(ax)
	set_y_arrow(ax)

def set_legend(legend_strings=None, ax=None, fontsize=12, loc='best', n_cols=1):
	ax = get_ax_if_none(ax)	
	if(legend_strings is None):
		ax.legend(ncol=n_cols, loc=loc, fontsize=fontsize)
	else:
		ax.legend(legend_strings, ncol=n_cols, loc=loc, fontsize=fontsize)

def update_fig_size(fig=None, height=4, width=6):
	if(fig is None):
		fig = plt.gcf()
		fig.set_size_inches(width, height, forward=True)
	else:
		fig.set_size_inches(width, height, forward=True)
	return fig

def get_color_cycle(): 
	return plt.rcParams['axes.prop_cycle'].by_key()['color']

def reset_color_cycle():
	curr_version = mpl.__version__.split('.')
	if(int(curr_version[0]) <= 1):
		if((int(curr_version[0]) == 1) and (int(curr_version[1]) > 5)): plt.gca().set_prop_cycle(None)
		else: plt.gca().set_color_cycle(None)
	else: plt.gca().set_prop_cycle(None)

def calc_errbars(true_vals, meas_vals, axis=0):
	'''
		Useful function to calculate errors bars for the matplotlib plt.errbars function
		neg_mae corresponds to the mae of all values that were LOWER than the true_val
		pos_mae corresponds to the mae of all values that were HIGHER than the true_val
	'''
	true_vals = true_vals.squeeze()
	meas_vals = meas_vals.squeeze()
	assert((axis==0) or (axis==-1)), 'Error: Input axis needs to be the first or last axis of tensor'
	assert((meas_vals.ndim==1) or (meas_vals.ndim==2)), 'Error: meas_vals needs to be a 1D or 2D tensor'
	assert((meas_vals.ndim-1) == true_vals.ndim), 'Error: true_vals needs to have 1 less dim than meas_vals, i.e., if meas_vals is 2D true_vals is 1D'
	# place measurements in the last dimension
	if(axis == 0): meas_vals = meas_vals.transpose()
	# Reshape meas_vals and true_vals into 2D and 1D tensors respectively.
	if(meas_vals.ndim==1):
		meas_vals = np.expand_dims(meas_vals, axis=0)	
		true_vals = np.expand_dims(true_vals, axis=0)
	# Calculate errors
	errors = meas_vals - np.expand_dims(true_vals, axis=-1) 
	# Figure out how many elements there are and pre-allocate arrays for the positive and negative errors
	# Note that this steps cannot be easily vectorized, because for each element there may be a different number of positive/negative errors.
	n_elems = meas_vals.shape[0] # we know that the elems will be in the first dimension, because the measurements are always in the last dimension
	# Calculate positive errors mean absolute error
	pos_mae = np.zeros((n_elems,))
	neg_mae = np.zeros((n_elems,))
	for i in range(n_elems):
		curr_errors = errors[i]
		pos_mae[i] = np.mean(np.abs(curr_errors[curr_errors>=0])) 
		neg_mae[i] = np.mean(np.abs(curr_errors[curr_errors<=0])) 
	return np.stack((neg_mae, pos_mae), axis=0)

def calc_mean_errbars(y, axis=0):
	y_mean = np.mean(y, axis=axis)
	y_negpos_mae = calc_errbars(y_mean, y, axis=axis)
	return y_negpos_mae

def get_good_min_max_range(img):
	mean_val = np.mean(img)
	stddev_val = np.std(img)
	vmin = mean_val - 2*stddev_val
	vmax = mean_val + 2.5*stddev_val
	return (vmin, vmax)
	
def enable_latex_fonts():
	plt.rcParams.update({
		"text.usetex": True,
	})

def disable_latex_fonts():
	plt.rcParams.update({
		"text.usetex": False,
	})

