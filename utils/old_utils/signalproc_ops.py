## Standard Library Imports

## Library Imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports

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
	v1convv2 = np.fft.irfft( np.fft.rfft( v1, axis=axis ) * np.fft.rfft( v2, axis=axis ), axis=axis, n=v1.shape[-1] )
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


def get_smoothing_window(N=100,window_len=11,window='flat'):
	"""smooth the data using a window with requested size.
 
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
	# Get smoothing window 
	w = get_smoothing_window( N = len( x ), window = window, window_len = window_len )
	y = np.real( CircularConv( x, w ) ) / ( w.sum() )
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

	Y = np.real( CircularConv(X, window) ) / (window.sum())

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

def sinc_interp(lres_signal, hres_n):
	# Reshape transient to simplify vectorized operations
	(lres_signal, lres_signal_original_shape) = vectorize_tensor(lres_signal)
	n_elems = lres_signal.shape[0]
	lres_n = lres_signal.shape[-1]
	assert((hres_n % lres_n) == 0), "Current sinc_interp is only implemented for integer multiples of lres_n"
	f_lres_signal = np.fft.rfft(lres_signal, axis=-1)
	lres_nf = f_lres_signal.shape[-1]
	hres_nf = (hres_n // 2) + 1
	f_hres_signal = np.zeros((n_elems, hres_nf), dtype=f_lres_signal.dtype)
	f_hres_signal[..., 0:lres_nf] = f_lres_signal
	hres_signal = np.fft.irfft(f_hres_signal)
	# Reshape final vectors
	hres_signal_original_shape = np.array(lres_signal_original_shape)
	hres_signal_original_shape[-1] = hres_n
	hres_signal = hres_signal.reshape(hres_signal_original_shape)
	lres_signal = lres_signal.reshape(lres_signal_original_shape)
	return hres_signal