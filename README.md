# compressive-spad-lidar-cvpr22
Code and Data for CVPR 2022 paper: Compressive Single-Photon 3D Cameras

## Coding Schemes Evaluated In CVPR 2022 Paper

The implementation of the coding schemes used in the paper are implemented as individual classes and can be found under `tof-lib/toflib/coding.py`. 

The following classes have a one-to-one correspondence to the coding schemes described in the *main paper*:

1. `GatedCoding`: This class corresponds to **Coarse Histograms** coding scheme
2. `TruncatedFourierCoding`: This class corresponds to **Truncated Fourier** coding scheme.
3. `PSeriesFourierCoding`: This class corresponds to **Gray-based Fourier** coding scheme. This coding scheme samples frequencies from the Fourier matrix by doubling the frequency that is sampled. Once it cannot double the frequency anymore, it reverts back to `TruncatedFourierCoding` and samples the remaining frequencies from lowest to highest
4. `GrayCoding`: This class corresponds to **Continuous Gray** coding scheme. This coding scheme is exactly the same as Gray coding when `K == log2(N)`. For all other `K` values the Gray codes are linearly interpolated. Note that this scheme is only valid for `K <= log2(N)`. For a coding scheme that uses approximately binary codes and supports `K > log2(N)`, see `PSeriesGray` below. 

Furthermore, the following classes have a ont-to-one correspondence with the coding schemes described in the supplementary document:

1. `PSeriesGrayCoding`: This class corresponds to **Fourier-based Gray** coding scheme. This coding scheme is similar to `PSeriesFourierCoding` but uses binarized codes. When `K <= log2(N)`, this coding scheme is the same as `Gray`
