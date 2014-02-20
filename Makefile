all: rif_correlator rif_correlator_sender rif_client rif_server

rif_correlator: rif_correlator.cu
	nvcc -O2 -L/opt/cuda/lib64 -lcufft -o rif_correlator rif_correlator.cu

rif_correlator_sender: rif_correlator_sender.c
	gcc -o rif_correlator_sender rif_correlator_sender.c

rif_client: rif_client.c
	gcc -o rif_client rif_client.c

rif_server: rif_server.c
	gcc -o rif_server rif_server.c

