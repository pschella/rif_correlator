all: rif_correlator rif_correlator_sender

rif_correlator: rif_correlator.cu
	nvcc -O2 -L/opt/cuda/lib64 -lcufft -o rif_correlator rif_correlator.cu

rif_sender: rif_sender.c
	gcc -o rif_sender rif_sender.c

