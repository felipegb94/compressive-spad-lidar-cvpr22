'''
	Sample Run command: 
		run tests/torch_code_opt_test.py -n_tbins 128 -n_opt_codes 4

	Example Optimize Codes for High Flux (1000-10000 photons), Very Low SBR (0.005-0.05): 

run tests/torch_code_opt_test.py -n_tbins 128 -epochs 400 -n_opt_codes 8 -min_max_nphotons_exp 3 4 -min_max_sbr_exp -2.3 -1.3 -init_id truncfourier

	This script is a simple 1D coding matrix optimization for depth sensing
	
	We generate a dataset composed of gaussian pulses with random mean and random sigma, and also have an unknown vertical offset 
	Each gaussian pulse is a 1D vector of size N, and it is corrupted by poisson noise 
	
	Consider the setup where we can only take K linear measurements of the gaussian pulse, and the goal is to estimate its mean.

	In this script the goal is to optimize the linear coding matrix C (NxK matrix), that will give us the best estimate of the mean of the gaussian pulse
	Each column in the matrix corresponds to a single linear measurement.
	To estimate the mean from the arbitrary K linear measurements we use Zero-Norm Cross-Corrlation decoding.  
'''
#### Standard Library Imports
import argparse
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils.timer import Timer
from research_utils import np_utils, plot_utils, io_ops, torch_utils
from research_utils.signalproc_ops import smooth
from research_utils.shared_constants import *
from toflib import coding
from toflib import tof_utils, input_args_utils
from toflib import torch_tirf_dataloaders, torch_coding

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_softmax = torch.nn.Softmax(dim=0).to(device)


if __name__=='__main__':
	# Get input arguments (e.g., config_id)
	parser = argparse.ArgumentParser(description='Parser for torch_coding_test.py')
	parser = input_args_utils.add_tbins_arg(parser)
	parser = input_args_utils.add_torch_code_opt_args(parser)
	args = parser.parse_args()
	# Parse input args
	n_tbins = args.n_tbins
	n_codes = args.n_opt_codes
	mini_batch_size = n_tbins
	mini_batch_size = 512
	add_noise_after_coding = False
	fourier_opt = False
	plot_freq = 100

	# Fix random seeds
	np.random.seed(seed=1)
	torch.manual_seed(seed=1)
	
	# Initialize dataloader
	tirf_dataset = torch_tirf_dataloaders.GaussianTIRFData(n_tbins=n_tbins, n_samples=4*n_tbins, mode='random', device=device 
												, min_max_nphotons_exp=args.min_max_nphotons_exp
												, min_max_sbr_exp=args.min_max_sbr_exp )
	loader = DataLoader(tirf_dataset, batch_size=mini_batch_size, shuffle=True)
	# no loader needed for validation
	tirf_val_dataset = torch_tirf_dataloaders.GaussianTIRFData(n_tbins=n_tbins, n_samples=20*n_tbins, mode='random', device=device 
												, min_max_nphotons_exp=args.min_max_nphotons_exp
												, min_max_sbr_exp=args.min_max_sbr_exp )
	tirf_val_dataset.simulate_new_tirf()

	# Put together output filenames. The opt codes in these files will be replaced if a set of codes with better loss are found
	out_dirpath = 'opt_codes_data'
	out_fname_params_str = 'loss-{}_ncodes-{}_ntbins-{}_nphotons-{}-{}_sbr-{}-{}'.format(args.loss_id, n_codes, n_tbins, tirf_dataset.min_max_nphotons_exp[0], tirf_dataset.min_max_nphotons_exp[1], tirf_dataset.min_max_sbr_exp[0], tirf_dataset.min_max_sbr_exp[1] ) 
	out_optC_fname = 'opt_C_{}'.format(out_fname_params_str)
	out_optC_results_fname = 'opt_results_{}'.format(out_fname_params_str)
	out_optC_fpath = os.path.join(out_dirpath, out_optC_fname + '.npy')
	out_optC_results_fpath = os.path.join(out_dirpath, out_optC_results_fname + '.json')

	# Generate fourier coding matrix (for debugging)
	fourier_coding = coding.TruncatedFourierCoding(n_tbins, n_freqs=(n_codes//2), include_zeroth_harmonic=False)
	gray_coding = coding.GrayCoding(n_tbins, n_bits=n_codes)
	identity_coding = coding.IdentityCoding(n_tbins)

	# initialize torch coding object
	# coding_layer = torch_coding.CodingLayer(C=fourier_C, n_maxres=n_tbins, n_codes=n_codes).to(device)
	assert(n_codes > 1), "input n_codes needs to be > 1"
	if(fourier_opt):
		print("Optimizing Fourier Codes")
		coding_layer = torch_coding.FourierCodingLayer(n_maxres=n_tbins, n_freqs=n_codes//2).to(device)
	else:
		print("Optimizing Unconstrained Codes")
		# coding_layer = torch_coding.HybridFourierCodingLayer(n_maxres=n_tbins, n_codes=n_codes).to(device)
		coding_layer = torch_coding.CodingLayer(n_maxres=n_tbins, n_codes=n_codes, init_id=args.init_id).to(device)
	beta=100 # higher beta peakier result
	## torch optimizer 
	epochs = args.epochs
	learning_rate = 1e-2
	loss_fn = torch.nn.L1Loss(reduction='mean')
	optimizer = torch.optim.Adam(coding_layer.parameters(), lr=learning_rate)
	# optimizer = torch.optim.SGD(coding_layer.parameters(), lr=learning_rate, momentum=0.9)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[750,1500,2250, 3000, 4000], gamma=0.5)

	losses = np.zeros((epochs,))
	softmax_mae_losses = np.zeros((epochs,))
	softmax_medae_losses = np.zeros((epochs,))
	softmax_onetol_losses = np.zeros((epochs,))
	argmax_mae_losses = np.zeros((epochs,))
	argmax_medae_losses = np.zeros((epochs,))
	argmax_onetol_losses = np.zeros((epochs,))
	fourier_mae_losses = np.zeros((epochs,))
	fourier_medae_losses = np.zeros((epochs,))
	fourier_onetol_losses = np.zeros((epochs,))
	freq_history = np.zeros((epochs, n_codes//2)).astype(np.float32)
	## Initialize results dict with the parameters of the optimization
	results_dict = {}
	results_dict['max_epochs'] = epochs 
	results_dict['learning_rate'] = learning_rate 
	results_dict['optimizer'] = str(type(optimizer)) # TODO: Keep track of all optimizer parameters, not just its type
	results_dict['beta'] = beta 
	results_dict['mini_batch_size'] = mini_batch_size 
	results_dict['n_codes'] = n_codes 
	results_dict['n_tbins'] = n_tbins 
	results_dict['loss_id'] = args.loss_id 
	results_dict['min_max_nphotons_exp'] = tirf_dataset.min_max_nphotons_exp 
	results_dict['min_max_sbr_exp'] = tirf_dataset.min_max_sbr_exp 
	plt.clf()
	for i in range(epochs):
		total_epoch_loss = 0
		n_iters_per_epoch = 0
		if((i > 1) or fourier_opt): 
			tirf_dataset.simulate_new_tirf()
		for step, data in enumerate(loader):
			(pulses, gt_mus, gt_sigmas, gt_mu_indeces) = data
			# encode
			c_vals = coding_layer.forward(pulses)
			if(args.loss_id == 'l1'):
				##### l1 loss 
				decoded_mus = coding_layer.softmax_peak_decoding(c_vals, rec_algo_id='ncc', beta=beta)
				# curr_loss = loss_fn(decoded_mus, gt_mus)
				curr_loss = loss_fn(torch_utils.normalize_known_range(decoded_mus, min_val=0, max_val=n_tbins), torch_utils.normalize_known_range(gt_mus, min_val=0, max_val=n_tbins))
			elif('tol' in args.loss_id):
				##### epsilon tolerance error loss
				ncc_scores = coding_layer.reconstruction(c_vals, rec_algo_id='ncc')
				if(args.loss_id == '1tol'):
					softmax_score = torch_utils.softmax_scoring(ncc_scores, gt_mu_indeces, beta=beta, eps=1)
				elif(args.loss_id == '2tol'):
					softmax_score = torch_utils.softmax_scoring(ncc_scores, gt_mu_indeces, beta=beta, eps=2)
				elif(args.loss_id == '0tol'):
					softmax_score = torch_utils.softmax_scoring(ncc_scores, gt_mu_indeces, beta=beta, eps=0)
				else: assert(False), "invalid loss function {}".format(args.loss_id)
				curr_loss = gt_mus.numel() - softmax_score
			elif(args.loss_id == 'hybrid'):
				decoded_mus = coding_layer.softmax_peak_decoding(c_vals, rec_algo_id='ncc', beta=beta)
				l1_loss = loss_fn(torch_utils.normalize_known_range(decoded_mus, min_val=0, max_val=n_tbins), torch_utils.normalize_known_range(gt_mus, min_val=0, max_val=n_tbins))
				ncc_scores = coding_layer.reconstruction(c_vals, rec_algo_id='ncc')
				softmax_score = torch_utils.softmax_scoring(ncc_scores, gt_mu_indeces, beta=beta, eps=2)
				curr_loss = l1_loss + 0.1*(gt_mus.numel() - softmax_score)
			else: assert(False), "invalid loss function {}".format(args.loss_id)
			total_epoch_loss += curr_loss
			# Backpropagation
			optimizer.zero_grad()
			curr_loss.backward()
			optimizer.step()
			if(fourier_opt):
				with torch.no_grad():
					coding_layer.C_unconstrained.clamp_(1, n_tbins // 4)
			else:
				with torch.no_grad():
					coding_layer.C_unconstrained.clamp_(-1,1)
		scheduler.step()
		losses[i] = total_epoch_loss / (step+1)
		# print("Softmax Score: {}".format(softmax_score))
		## Testing over full train set
		(test_pulses, test_gt_mus, test_gt_sigmas) = (tirf_val_dataset.tirf, tirf_val_dataset.gt_mus, tirf_val_dataset.gt_sigmas)
		c_vals = coding_layer.forward(test_pulses)
		# softmax_rec = coding_layer.softncc_reconstruction(c_vals, beta=beta)
		softmax_decoded_mus = coding_layer.softmax_peak_decoding(c_vals, rec_algo_id='ncc', beta=beta)
		softmax_abs_errors = np.abs(test_gt_mus.cpu().numpy() - softmax_decoded_mus.detach().data.cpu().numpy())
		softmax_metrics = np_utils.calc_error_metrics(softmax_abs_errors)
		argmax_decoded_mus = coding_layer.max_peak_decoding(c_vals, rec_algo_id='ncc')
		argmax_abs_errors = np.abs(test_gt_mus.cpu().numpy() - argmax_decoded_mus.detach().data.cpu().numpy())
		argmax_metrics = np_utils.calc_error_metrics(argmax_abs_errors)
		# Test on fourier codes too
		fourier_c_vals = fourier_coding.encode(test_pulses.data.cpu().numpy())
		decoded_fourier_mus = fourier_coding.max_peak_decoding(fourier_c_vals, rec_algo_id='ncc')
		fourier_abs_errors = np.abs(test_gt_mus.cpu().numpy() - decoded_fourier_mus)
		fourier_metrics = np_utils.calc_error_metrics(fourier_abs_errors)
		# Test on Gray codes too
		gray_c_vals = gray_coding.encode(test_pulses.data.cpu().numpy())
		decoded_gray_mus = gray_coding.max_peak_decoding(gray_c_vals, rec_algo_id='ncc')
		gray_abs_errors = np.abs(test_gt_mus.cpu().numpy() - decoded_gray_mus)
		gray_metrics = np_utils.calc_error_metrics(gray_abs_errors)
		# Test on Identity codes too
		identity_c_vals = identity_coding.encode(test_pulses.data.cpu().numpy())
		decoded_identity_mus = identity_coding.max_peak_decoding(identity_c_vals, rec_algo_id='linear')
		identity_abs_errors = np.abs(test_gt_mus.cpu().numpy() - decoded_identity_mus)
		identity_metrics = np_utils.calc_error_metrics(identity_abs_errors)
		## Calc metrics
		softmax_mae_losses[i] = softmax_metrics['mae']
		softmax_medae_losses[i] = softmax_metrics['medae']
		softmax_onetol_losses[i] = softmax_metrics['1_tol_errs']
		argmax_mae_losses[i] = argmax_metrics['mae']
		argmax_medae_losses[i] = argmax_metrics['medae']
		argmax_onetol_losses[i] = argmax_metrics['1_tol_errs']
		fourier_mae_losses[i] = fourier_metrics['mae']
		fourier_medae_losses[i] = fourier_metrics['medae']
		fourier_onetol_losses[i] = fourier_metrics['1_tol_errs']
		if(i % 25 == 0):
			print("Epoch: {} of {}, mean epoch loss = {}".format(i,epochs, losses[i]))
			print("    Latest Metrics (softmax): ")
			np_utils.print_error_metrics(softmax_metrics, prefix='        ')		
			print("    Latest Metrics (argmax): ")
			np_utils.print_error_metrics(argmax_metrics, prefix='        ')		
			print("    Fourier metrics: ")		
			np_utils.print_error_metrics(fourier_metrics, prefix='        ')		
			print("    Gray metrics: ")		
			np_utils.print_error_metrics(gray_metrics, prefix='        ')		
			print("    Identity metrics: ")		
			np_utils.print_error_metrics(identity_metrics, prefix='        ')		
		if(fourier_opt):
			print("    Freqs: {}".format(coding_layer.C_unconstrained))		
			freq_history[i,:] = coding_layer.C_unconstrained.detach().data.cpu().numpy()
		if(i % plot_freq == 0):
			plt.clf()
			plt.subplot(3,1,1)
			plt.title('Epochs: {}'.format(i))
			plt.plot(coding_layer.C.detach().data.cpu().numpy())
			# plt.plot(coding_layer.hybrid_C.detach().data.cpu().numpy())
			plt.subplot(3,1,2)
			plt.plot(pulses.data.cpu().numpy()[0:4].transpose())
			lookups = coding_layer.softncc_reconstruction(c_vals, beta=beta)
			plt.subplot(3,2,5)
			plt.plot(smooth(softmax_mae_losses[11:i],window_len=11), label='softmax_mae')
			plt.plot(smooth(argmax_mae_losses[11:i],window_len=11), label='argmax_mae')
			plt.plot(smooth(fourier_mae_losses[11:i],window_len=11), label='fourier_mae')
			plt.xlim(11, epochs)
			plt.legend()
			plt.subplot(3,2,6)
			plt.plot(smooth(softmax_onetol_losses[11:i], window_len=11), label='softmax_onetol')
			plt.plot(smooth(argmax_onetol_losses[11:i], window_len=11), label='argmax_onetol')
			plt.plot(smooth(fourier_onetol_losses[11:i], window_len=11), label='fourier_onetol')
			plt.xlim(11, epochs)
			plt.legend()
			plt.pause(0.1)
		# After the 10th epoch start checking if codes are better than the ones saved
		if(i > 10):
			# Calc current status
			val_loss = 0.
			update_best_opt_C = False
			if(args.loss_id == 'l1'):
				## If we use L1, converge using mae loss
				val_loss = argmax_mae_losses[i-10:i].mean()
			elif('tol' in args.loss_id):
				## If we use EPSILON-Tolerance, converge using EPS-tolerance loss
				val_loss = argmax_onetol_losses[i-10:i].mean()
			elif(args.loss_id == 'hybrid'):
				## If we use L1, converge using mae loss
				val_loss = argmax_mae_losses[i-10:i].mean()
			else: assert(False), 'Invalid loss...'
			results_dict['val_loss'] = val_loss 
			# If a prev results dict load it and compare the validation loss
			if(os.path.exists(out_optC_results_fpath)):
				prev_results_dict = io_ops.load_json(out_optC_results_fpath)
				prev_val_loss = prev_results_dict['val_loss']
				if(args.loss_id == 'l1'):
					update_best_opt_C = prev_val_loss > results_dict['val_loss']  
				elif('tol' in args.loss_id):
					update_best_opt_C = prev_val_loss < results_dict['val_loss']  
			else:
				prev_val_loss = val_loss
				# If a previous results dict did not exist, create it 
				update_best_opt_C = True
			if(update_best_opt_C):
				print("Saving new best C. Old Loss = {:.2f}, New Loss = {:.2f}".format(prev_val_loss, val_loss))		
				results_dict['softmax_mae_losses'] = softmax_mae_losses.tolist() 
				results_dict['softmax_medae_losses'] = softmax_medae_losses.tolist() 
				results_dict['softmax_onetol_losses'] = softmax_onetol_losses.tolist() 
				results_dict['argmax_mae_losses'] = argmax_mae_losses.tolist() 
				results_dict['argmax_medae_losses'] = argmax_medae_losses.tolist() 
				results_dict['argmax_onetol_losses'] = argmax_onetol_losses.tolist() 
				io_ops.write_json(out_optC_results_fpath, results_dict)
				np.save(out_optC_fpath,  coding_layer.C.cpu().detach().numpy())




