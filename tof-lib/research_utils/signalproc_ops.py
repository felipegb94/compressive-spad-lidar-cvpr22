## Standard Library Imports

## Library Imports
import numpy as np
from scipy import signal
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from .np_utils import vectorize_tensor, unvectorize_tensor, to_nparray, get_extended_domain, extend_tensor_circularly
from .shared_constants import *

# Smoothing windows that are available to band-limit a signal
SMOOTHING_WINDOWS = ['flat', 'impulse', 'hanning', 'hamming', 'bartlett', 'blackman']  

def circular_conv( v1, v2, axis=-1 ):
	"""Circular convolution: Calculate the circular convolution for vectors v1 and v2. v1 and v2 are the same size
	
	Args:
		v1 (numpy.ndarray): ...xN vector	
		v2 (numpy.ndarray): ...xN vector	
	Returns:
		v1convv2 (numpy.ndarray): convolution result. N x 1 vector.
	"""
	v1convv2 = np.fft.irfft( np.fft.rfft( v1, axis=axis ) * np.fft.rfft( v2, axis=axis ), axis=axis, n=v1.shape[axis] )
	return v1convv2

def circular_corr( v1, v2, axis=-1 ):
	"""Circular correlation: Calculate the circular correlation for vectors v1 and v2. v1 and v2 are the same size
	
	Args:
		v1 (numpy.ndarray): Nx1 vector	
		v2 (numpy.ndarray): Nx1 vector	
	Returns:
		v1corrv2 (numpy.ndarray): correlation result. N x 1 vector.
	"""
	v1corrv2 = np.fft.ifft( np.fft.fft( v1, axis=axis ).conj() * np.fft.fft( v2, axis=axis ), axis=axis ).real
	return v1corrv2

def circular_matched_filter(s, template, axis=-1):
	assert(s.shape[axis] == template.shape[axis]), "input signal and template dims need to match at axis"
	corrf = circular_corr(template, s, axis=axis)
	return np.argmax(corrf, axis=axis)

def get_smoothing_window(N=100,window_len=11,window='flat'):
	"""
		smooth the data using a window with requested size.
	"""
	## Validate Inputs
	if(N < window_len):
		raise ValueError("Input vector needs to be bigger than window size.")
	if(not window in SMOOTHING_WINDOWS):
		raise ValueError( "Chosen smoothing window needs to be one of: {}".format( SMOOTHING_WINDOWS ) )
	## Generate smoothing window
	w = np.zeros((N,))
	if window == 'flat': #moving average
		w[0:int(window_len)]=np.ones(int(window_len),'d')
	elif window == 'impulse':
		w[0] = 1 
	else:
		w[0:int(window_len)]=eval('np.'+window+'(int(window_len))')
	shift = np.argmax(w)
	w = np.roll(w, shift=-1*shift )
	# Return normalized smoothhing window
	return (w / (w.sum()))

def smooth(x, window_len=11, window='flat'):
	"""smooth the data using a window with requested size.
	 
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	 
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.
	 
	output:
		the smoothed signal
		 
	example:
	 
	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	 
	see also: 
	 
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
	 
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""
	#### Validate Inputs
	if( x.ndim != 1 ):
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if( window_len < 3 ):
		return x
	if( window_len > len(x)):
		print("Not smoothing. signal is smaller than window lengths")
		return x
	# Get smoothing window 
	w = get_smoothing_window( N = len( x ), window = window, window_len = window_len )
	y = np.real( circular_conv( x, w ) ) / ( w.sum() )
	# y = np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(w)))/(w.sum())
	#### The line below performs the same operation as the line above but slower
	# np.convolve(w/(w.sum()),s,mode='valid')
	return y

def smooth_tensor(X, window_duty=0.1, window='hanning'):
	assert(window_duty < 1.0), "window_duty needs to be less than one"
	assert(window_duty > 0.0), "window_duty needs to be greater than 0"
	X_shape = X.shape
	n = X.shape[-1]
	n_arrays = int(X.size / n)
	X = X.reshape((n_arrays,n))

	window = get_smoothing_window(N=n, window_len=window_duty*n, window=window)
	window = window.reshape((1,n))

	Y = np.real( circular_conv(X, window) ) / (window.sum())

	return Y.reshape(X_shape)

def smooth_codes( modfs, demodfs, window_duty=0.15 ):
	(N,K) = modfs.shape
	smoothed_modfs = np.zeros( (N,K) )
	smoothed_demodfs = np.zeros( (N,K) )
	#### Smooth functions. No smoothing is applied by default
	for i in range(0,K):
		smoothed_modfs[:,i] = Smooth( modfs[:,i], window_len = N*window_duty, window='hanning' ) 
		smoothed_demodfs[:,i] = Smooth( demodfs[:,i], window_len = N*window_duty, window='hanning' )
	return (smoothed_modfs, smoothed_demodfs)

def circulant(f, direction = 1):
	"""Circulant
	 
	Args:
		f (numpy.ndarray): Vector to generate circulant matrix from
		direction (int, optional): Direction used to shift the vector when generating the matrix.
	 
	Returns:
		np.ndarray: Circulant matrix.
	"""
	#### Verify input
	# assert(UtilsTesting.IsVector(f)),'Input Error - Circulant: f should be a vector.'
	# assert((direction == 1) or (direction == -1)), 'Input Error - Circulant: The direction needs \
	# to be either forward (dir=1) or backward (dir=-1).'
	#### Get parameters
	N = f.size # We know f is a vector so just use its size.
	C = np.zeros((N,N))
	isRow = (f.shape[0] == 1) # Doesn't matter for ndarrays
	#### Generate circulant matrix
	if(isRow):
		for i in range(0,N):
			C[[i],:] = np.roll(f,i*direction)
	else:
		for i in range(0,N):
			C[:,[i]] = np.roll(f,i*direction).reshape((N,1))
 
	return C

def sinc_interp(lres_signal, hres_n, axis=-1):
	'''
		I found out the scipy's resample does sinc interpolation so I have replaced this code with that
	'''
	hres_signal = signal.resample(lres_signal, hres_n, axis=axis)
	return hres_signal

def sinc_interp_old(lres_signal, hres_n):
	'''
		I tested the output of this code with the sinc interp function from scipy (scipy.signal.resample)
		and the outputs matched. So this works find.
		But, it is 3-5x slower than scipy so I replaced it with the scipy implementation
		But I am leaving this here for future reference
	'''
	# Reshape transient to simplify vectorized operations
	(lres_signal, lres_signal_original_shape) = vectorize_tensor(lres_signal)
	n_elems = lres_signal.shape[0]
	lres_n = lres_signal.shape[-1]
	assert((hres_n % lres_n) == 0), "Current sinc_interp is only implemented for integer multiples of lres_n"
	upscaling_factor = hres_n / lres_n
	f_lres_signal = np.fft.rfft(lres_signal, axis=-1)
	lres_nf = f_lres_signal.shape[-1]
	hres_nf = (hres_n // 2) + 1
	f_hres_signal = np.zeros((n_elems, hres_nf), dtype=f_lres_signal.dtype)
	f_hres_signal[..., 0:lres_nf] = f_lres_signal
	# NOTE: For some reason we have to multiply by the upscaling factor if we want the output signal to have the same amplitude
	hres_signal = np.fft.irfft(f_hres_signal)*upscaling_factor
	# Reshape final vectors
	hres_signal_original_shape = np.array(lres_signal_original_shape)
	hres_signal_original_shape[-1] = hres_n
	hres_signal = hres_signal.reshape(hres_signal_original_shape)
	lres_signal = lres_signal.reshape(lres_signal_original_shape)
	return hres_signal

def normalize_signal(v, axis=-1): return v / (v.sum(axis=axis, keepdims=True) + EPSILON)
def standardize_signal(v, axis=-1): return (v - v.min(axis=axis, keepdims=True)) / (v.max(axis=axis, keepdims=True) - v.min(axis=axis, keepdims=True) + EPSILON)

def gaussian_pulse(time_domain, mu, width, circ_shifted=True):
	'''
		Generate K gaussian pulses with mean=mu and sigma=width.
		If circ_shifted is set to true we create a gaussian that wraps around at the boundaries.
	'''
	mu_arr = to_nparray(mu)
	width_arr = to_nparray(width)
	assert((width_arr.size==1) or (width_arr.size==mu_arr.size)), "Input mu and width should have the same dimensions OR width should only be 1 element"
	if(circ_shifted):
		ext_time_domain = get_extended_domain(time_domain)
		ext_pulse = np.exp(-1*np.square((ext_time_domain[np.newaxis,:] - mu_arr[:, np.newaxis]) / width_arr[:, np.newaxis]))
		n_bins = time_domain.shape[-1]
		pulse = ext_pulse[...,0:n_bins] + ext_pulse[...,n_bins:2*n_bins] + ext_pulse[...,2*n_bins:3*n_bins]
	else:
		pulse = np.exp(-1*np.square((time_domain[np.newaxis,:] - mu_arr[:, np.newaxis]) / width_arr[:, np.newaxis]))
	return normalize_signal(pulse.squeeze(), axis=-1)

def expgaussian_pulse_erfc(time_domain, mu, sigma, exp_lambda):
	if(exp_lambda is None): return gaussian_pulse(time_domain, mu, sigma)
	mu_arr = to_nparray(mu)
	sigma_sq = np.square(sigma)
	mu_minus_t = mu_arr[:, np.newaxis] - time_domain[np.newaxis,:]  
	lambda_sigma_sq = exp_lambda*sigma_sq
	erfc_input = (mu_minus_t + lambda_sigma_sq) / sigma
	pulse = exp_lambda*np.exp(0.5*exp_lambda*(lambda_sigma_sq + 2*mu_minus_t))*scipy.special.erfc(erfc_input)
	return normalize_signal(pulse.squeeze(), axis=-1)

def expgaussian_pulse_conv(time_domain, mu, sigma, exp_lambda):
	gauss_pulse = gaussian_pulse(time_domain, mu, sigma)
	if(exp_lambda is None): return gauss_pulse
	exp_lambda = to_nparray(exp_lambda)
	exp_decay = np.exp(-1*exp_lambda[:, np.newaxis]*time_domain[np.newaxis,:])
	expgauss_pulse = circular_conv(exp_decay, gauss_pulse, axis=-1)
	return normalize_signal(expgauss_pulse.squeeze(), axis=-1)

def verify_time_domain(time_domain=None, n=1000):
	if(not (time_domain is None)):
		time_domain = to_nparray(time_domain)
		n = time_domain.shape[-1]
	else:
		time_domain = np.arange(0, n)
	assert(n > 1), "Number of time bins in time domain needs to be larger than 1 (n = {})".format(n)
	dt = time_domain[1] - time_domain[0]
	tau = time_domain[-1] + dt
	return (time_domain, n, tau, dt)

def get_random_gaussian_pulse_params(time_domain=None, n=1000, min_max_sigma=None, n_samples=1):
	(time_domain, n, tau, dt) = verify_time_domain(time_domain, n)
	mu = tau*np.random.rand(n_samples)
	if(min_max_sigma is None): min_max_sigma = (1, 10)
	if(min_max_sigma[1] == min_max_sigma[0]): sigma = np.ones_like(mu)*min_max_sigma[0]
	else: sigma = dt*np.random.randint(low=min_max_sigma[0], high=min_max_sigma[1], size=(n_samples,))
	return (mu, sigma)

def get_random_expgaussian_pulse_params(time_domain=None, n=1000, min_max_sigma=None, min_max_lambda=None, n_samples=1):
	(time_domain, n, tau, dt) = verify_time_domain(time_domain, n)
	(mu, sigma) = get_random_gaussian_pulse_params(time_domain=time_domain, n=n, min_max_sigma=min_max_sigma, n_samples=n_samples)
	if(min_max_lambda is None): min_max_lambda = (1, 50)
	if(min_max_lambda[1] == min_max_lambda[0]): exp_lambda = np.ones_like(mu)*min_max_lambda[0]
	else: exp_lambda = dt*np.random.randint(low=min_max_lambda[0], high=min_max_lambda[1], size=(n_samples,))
	exp_lambda = 1. / (dt*np.random.randint(low=min_max_lambda[0], high=min_max_lambda[1], size=(n_samples,)))
	return (mu, sigma, exp_lambda)

def get_fourier_mat(n, freq_idx=None):
	'''
		n is the number of samples in the primary domain
		freq_idx are the frequencies you want to get

		Return an nxk matrix where each column is a cmpx sinusoid with
	'''
	# If no frequency indeces are given simply return the full dft matrix
	if(freq_idx is None):
		return scipy.linalg.dft(n).transpose()
	# For each frequency idx add them to their corresponding cmpx sinusoid to the matrix
	n_freqs = len(freq_idx)
	domain = np.arange(0, n)*(TWOPI / n)
	fourier_mat = np.zeros((n, n_freqs), dtype=np.complex64)
	for i in range(n_freqs):
		fourier_mat[:, i] = np.cos(freq_idx[i]*domain) + 1j*np.sin(freq_idx[i]*domain)
	return fourier_mat

def broadcast_toeplitz( C_tensor, R_tensor=None):
	'''
		Create a toeplitz matrix using the last dimension of the input tensor
	'''
	if R_tensor is None:
		R_tensor = C_tensor.conjugate()
	else:
		R_tensor = np.asarray(R_tensor)
	# Form a 1D array of values to be used in the matrix, containing a reversed
	# copy of r[1:], followed by c.
	vals_tensor = np.concatenate((R_tensor[...,-1:0:-1], C_tensor), axis=-1)
	a, b = np.ogrid[0:C_tensor.shape[-1], R_tensor.shape[-1] - 1:-1:-1]
	indx = a + b
	# `indx` is a 2D array of indices into the 1D array `vals`, arranged so
	# that `vals[indx]` is the Toeplitz matrix.
	return vals_tensor[..., indx]

def max_gaussian_center_of_mass_mle(transient, tbins=None, sigma_tbins = 1):
	'''
		In this function we find the maximum of the transient and then calculate the center of mass in the neighborhood of the maximum.
		NOTE: At low SNR, low depths will have lower depth error on average than far away depths. 
		This is because, at low SNR (low SBR/low photon counts), it becomes very likely that there are multiple maximums, some maximums are 
		due to the signal and others due to ambient photons. And since numpy's argmax function always takes the 1st maximum it finds, then at low depths
		the maximum due to the signal are preferred, but at large depths the maximums due to ambient (that come before) are chosen.
	'''
	# Reshape transient to simplify vectorized operations
	(transient, transient_original_shape) = vectorize_tensor(transient)
	n_elems = transient.shape[0]
	n_tbins = transient.shape[-1]
	# Remove ambient (assume that median is a good estimate of ambient component)
	ambient_estimate = np.median(transient, axis=-1, keepdims=True)
	transient_noamb = transient - ambient_estimate
	# Make sure there are not negative values
	transient_noamb[transient_noamb < 0] = 0
	# Find start and end tbin of gaussian pulse
	argmax_tbin = np.argmax(transient, axis=-1)
	# start_tbin = np.clip(argmax_tbin - int(np.ceil(2*sigma_tbins)), 0, n_tbins)
	# end_tbin = np.clip(argmax_tbin + int(np.ceil(2*sigma_tbins)) + 1, 0, n_tbins)
	start_tbin = argmax_tbin - int(np.ceil(2*sigma_tbins))
	end_tbin = argmax_tbin + int(np.ceil(2*sigma_tbins)) + 1
	# Create a dummy tbin array if tbins are not given
	if(tbins is None): tbins = np.arange(0, n_tbins) 
	assert(transient.shape[-1] == len(tbins)), 'transient and tbins should have the same number of elements'
	# For each 1D transient calculate the center of mass max likelihood estimate
	center_of_mass_mle = np.zeros((n_elems,))
	extended_tbins = get_extended_domain(tbins, axis=-1)
	extended_transient_noamb = extend_tensor_circularly(transient_noamb, axis=-1)
	# for i in range(n_elems):
	# 	tbin_vec = tbins[start_tbin[i]:end_tbin[i]]
	# 	transient_vec = transient_noamb[i, start_tbin[i]:end_tbin[i]]
	# 	center_of_mass_mle[i] = np.dot(transient_vec, tbin_vec) / (np.sum(transient_vec) + EPSILON)
	for i in range(n_elems):
		start_idx = start_tbin[i]+n_tbins
		end_idx = end_tbin[i]+n_tbins
		tbin_vec = extended_tbins[start_idx:end_idx]
		transient_vec = extended_transient_noamb[i, start_idx:end_idx]
		center_of_mass_mle[i] = np.dot(transient_vec, tbin_vec) / (np.sum(transient_vec) + EPSILON)
	# Reshape to original shapes, useful when dealing with images
	transient = unvectorize_tensor(transient, transient_original_shape)
	center_of_mass_mle = center_of_mass_mle.reshape(transient_original_shape[0:-1])
	return center_of_mass_mle

def haar_matrix(n, n_levels):
	assert(n_levels >= 0), 'n_levels should be larger than '
	n_codes = np.power(2, n_levels)
	assert((n % n_codes) == 0), "only implemented multiples of 2^n_levels"
	H = np.zeros((n, n_codes))
	for i in range(n_levels+1):
		curr_total_codes = np.power(2, i)
		n_codes_at_curr_lvl = int(np.ceil(curr_total_codes / 2))
		half_duty_len = int(n / curr_total_codes)
		curr_start_code_idx = int(curr_total_codes - n_codes_at_curr_lvl)
		for j in range(n_codes_at_curr_lvl):
			start_idx = (j*half_duty_len*2)
			mid_point_idx = start_idx + half_duty_len
			end_idx = start_idx + 2*half_duty_len
			H[start_idx:mid_point_idx, curr_start_code_idx+j] = 1.0
			H[mid_point_idx:end_idx, curr_start_code_idx+j] = -1.0
	return H

def generate_gray_code(n_bits):
	assert(n_bits >= 1), "invalid n_bits"
	n_binary_codes = np.power(2, n_bits)
	n_binary_codes_over2 = int(n_binary_codes / 2)
	codes = np.zeros((n_binary_codes, n_bits))
	codes[1, -1] = 1 
	for i in range(n_bits-1):
		curr_n_bits = i + 2 
		start_code_idx = 0
		end_code_idx1 = np.power(2, i+1)
		end_code_idx2 = 2*end_code_idx1
		# Reflect 
		reflected_codes = np.flipud(codes[start_code_idx:end_code_idx1, :])
		codes[end_code_idx1:end_code_idx2, :] = reflected_codes
		# Prefix with ones
		codes[end_code_idx1:end_code_idx2, -curr_n_bits] = 1
	return codes

def get_orthogonal_binary_code(c):
	'''
		only works for square codes with 50% duty cycle
	'''
	assert(c.ndim == 1), "input c should be 1D vector"
	n = c.shape[0]
	shift_f0_90deg = n // 4
	# Find the repetition frequency of the code
	f_c = np.fft.rfft(c, axis=0)
	fk = np.abs(f_c).argmax()
	# Shift the code
	shift = int(np.round(shift_f0_90deg / fk))
	c_orth = np.roll(c, shift=shift, axis=0)
	# print("Orthogonal Measure: {}".format(np.dot(c_orth,c)))
	return c_orth

def get_dominant_freqs(Cmat, axis=0):
	f_Cmat = np.fft.rfft(Cmat, axis=axis)
	return np.argmax(np.abs(f_Cmat), axis=axis)

def get_low_confidence_freqs(h_irf, valid_freq_thresh=0.2):
	'''
		Look at frequency response of h_irf vector, and find frequencies with low amplitude
	'''
	nt = h_irf.shape[-1]
	abs_max_freq_idx = nt // 2
	all_freq_idx = np.arange(0, abs_max_freq_idx+1)
	# Calculate FFT of IRF and get frequencies with magnitude above threshold
	f_h_irf = np.fft.rfft(h_irf)
	amp_f_h_irf = np.abs(f_h_irf)
	# Frequencies should have a magnitude higher than the following computed w.r.t the 1st harmonic
	threshold = f_h_irf[1]*valid_freq_thresh
	low_confidence_freqs = amp_f_h_irf < threshold
	low_confidence_freq_idx = all_freq_idx[low_confidence_freqs]
	return (low_confidence_freq_idx, low_confidence_freqs)
