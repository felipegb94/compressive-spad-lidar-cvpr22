# Scripts for real data processing

**NOTE:**  This `README.md` is for informational purposes only and is related to the pre-processing scripts that were used to generate the data used in the paper. A lot of these scripts were not included here and are not needed to reproduce the paper results. You can find the pre-processing scripts under this repository: https://github.com/felipegb94/WISC-SinglePhoton3DData 


Folder content:

* `t3mode-scripts`: Old scripts downloaded from https://bitbucket.org/compoptics/t3mode-scripts/src/master/
* `test_data`: A few `.out` SPAD hydraharp data files from `20190205_face_scanning` scene
* `read_hydraharp_outfile_t3_debug.py`: File to read the data file and generate the timestamp histogram.
* `hist2timestamps.py`: Function that loads a histogram and generates the timestamps that created the histogram. The inversion will only be perfect if the input histogram is the same resolution as the timestamps.

Operating/Shifting modes:

* `free`: Free running mode == Photon-driven mode
* `det`: Deterministic shifting == Uniform shifting
* `ext`: External triggering (by laser). Synchronous mode. No attenuation
* `ext_opt_filtering`: External triggering (by laser). Synchronous mode. Optimal Filterning
* `pulse_waveform_calib`: Single point scan for calibrating waveform. Some waveforms still had undesired reflections.

## Processed Data

The `raw_hist_imgs` folder contains a few histogram images with minimal processing. An example of how to load them is found in `hist2timestamps.py`

## Pre-processing Histogram Images Notes

The `scan_params.json` file contains some `"hist_preprocessing_params"`.

The histograms are initially pre-processed with the script `read_fullscan_hydraharp_t3.py`.

**Run Scripts in the Following Order:**

1. `read_fullscan_hydraharp_t3.py`: Loads full raw histogram, and then saves a cropped histogram using the `scan_params.json`
2. `preprocess_irf.py`: (Optional) Only need to run it once for a set of `scan_params.json`. It will update or create the IRF for the given `scan_params.json`. This script requires that `read_fullscan_hydraharp_t3.py` to be ran once for the IRF scene
3. `bimodal2unimodal_hist_img.py`: Reads output hist img by `read_fullscan_hydraharp_t3.py` and turns all bi-modal peaks to uni-modal
4. `process_hist_img.py`: Processes a the hist image and generates depth images using different coding approaches

### Cropping histogram

One pre-processing step we do is crop the histogram at the beginning and at the end **while keeping the time resolution the same**.

Cropping the histogram has 2 goals:

1. Gate undesired lens reflections and stray light at the beginning and at the of the histogram.
2. Vary Signal-to-background levels. If we know the signal is at the middle of the histogram, and we crop the sides of the histogram we effectively reduce the total number of background photons in the histogram.

Start and End times for cropping:

1. (37000, 47000)
2. (37000, 44504) --> We use 44504 instead of 44500 because the histogram length needs to be a multiple of tbin size which is 8
3. (37000, 42000)
1. (37000, 57000)
1. (37000, 45192)

### Bimodal2Unimodal Conversion

Uni-modal histograms go through an additional pre-processing step which is performed by the `bimodal2unimodal_hist_img.py`.

The raw data we acquired has a bi-modal IRF (due to lens interreflections), which is uncommon in well calibrated systems. Therefore, we remove the second peak with this script. 

## Dependencies

* Python >=3.6 ('3.6.13 | packaged by conda-forge | (default, Feb 19 2021, 05:36:01) \n[GCC 9.3.0]') 
* Numpy ('1.19.5')
* Matplotlib ('3.2.2')

## Useful conversions

A 50ns period (20MHz) has a 7.5m unambiguous range  
A 100ns period (10MHz) has a 15m unambiguous range  