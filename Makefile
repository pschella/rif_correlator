all: rif_correlator rif_client rif_server

rif_correlator: rif_correlator.cu
	nvcc -O2 -L/opt/cuda/lib64 -lcufft -o rif_correlator rif_correlator.cu

rif_client: rif_client.c
	gcc -o rif_client rif_client.c

rif_server: rif_server.c
	gcc -o rif_server rif_server.c

