# tof-lib

Library for Indirect and Direct Time-of-Flight Imaging. Includes algorithms for estimating transient images, depth estimation, and noise models. 

## Setup

Clone repository with submodules included:

```
git clone --recursive git@github.com:felipegb94/tof-lib.git
```

To setup the python environment you can use the `environment.yml` Anaconda env. For the scripts that use pytorch you will need to setup Pytorch in your system.

## Running Tests

We assume all tests are being run from the top-level project folder, i.e., `python tests/tirf_t.py`.
