## Standard Library Imports
import time

## Library Imports

## Local Imports

class Timer(object):
	def __init__(self, name=None):
		self.name = name
	def __enter__(self):
		self.tstart = time.time()
	def __exit__(self, type, value, traceback):
		if self.name:
			print('[{}] - Elapsed: {} seconds.'.format(self.name, time.time() - self.tstart))
		else:
			print('Elapsed: {}'.format(time.time() - self.tstart))


