'''
	Test functionality of toflib.tirf.py
	Example run command:
		run tests/tirf_scene_test.py -data_dirpath ./sample_data -scene_id vgroove
'''
#### Standard Library Imports

#### Library imports
from random import gauss
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt

#### Local imports
from toflib import coding
from toflib import tirf

if __name__=='__main__':

    ## Basic params
    n_tbins = 1024
    n_codes = 63

    ## tirf params
    (mu, sigma) = (0, 10)
    gauss_pulse = tirf.GaussianTIRF(n_tbins, mu=mu, sigma=sigma)
    exp_lambda = 1./1024
    expmodgauss_pulse = tirf.ExpModGaussianTIRF(n_tbins, mu=mu, sigma=sigma, exp_lambda=exp_lambda)
    h_irf = gauss_pulse.tirf
    # h_irf = expmodgauss_pulse.tirf
    h_irf[h_irf < 1e-8] = 0
    account_irf = True
    # plt.plot(h_irf)
    plt.plot(np.abs(np.fft.rfft(h_irf)))

    ## Gray based fourier coding
    pser_cobj = coding.PSeriesGrayBasedFourierCoding(n_tbins, n_codes=n_codes, h_irf=h_irf, account_irf=account_irf)
    # pser_cobj = coding.PSeriesGrayBasedFourierCoding(n_tbins, n_codes=n_codes, h_irf=h_irf, account_irf=account_irf, include_zeroth_harmonic=True)


    hybrid_cobj = coding.HybridGrayBasedFourierCoding(n_tbins, n_codes=n_codes, h_irf=h_irf, account_irf=account_irf)
    # hybrid_cobj = coding.HybridGrayBasedFourierCoding(n_tbins, n_codes=n_codes, h_irf=h_irf, account_irf=account_irf, include_zeroth_harmonic=True)



    # ## Print and plot
    # print(cobj.freq_idx)
    # plt.imshow(cobj.get_pretty_C())

    