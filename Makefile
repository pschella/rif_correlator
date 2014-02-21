all: rif_correlator

rif_correlator: rif_correlator.cu
	nvcc -O2 -L/opt/cuda/lib64 -lcufft -o rif_correlator rif_correlator.cu

