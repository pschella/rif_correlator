/*
   Copyright (C) 2014 Pim Schellart <P.Schellart@astro.ru.nl>

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <netdb.h>

int imin(int a, int b)
{
	return (a < b ? a : b);
}

#define PACKET_SIZE 1024

/* TCP port */
#define PORT_NUMBER 32000

/* Number of samples per spectrum */
#define NX 1024

/* Number of frequencies per spectrum */
#define NF (NX/2+1)

/* Total number of samples in a time bin */
#define NTOT 20000000

/* Number of samples to average per channel per time bin
 * remaining samples are skipped. */
#define N ((NTOT / NX) * NX)

/* Number of FFTs to perform in one batch */
#define BATCH (N / NX)

/* Number of samples after FFT */
#define NS (NF*BATCH)

/* Compute the number of threads.
   With values below for at maximum 32 * 2048 = 2**16 frequency channels.
   When more are needed first increase threadsPerBlock to a higher power of two. */
const int threadsPerBlock = 32;
const int blocksPerGrid = imin(2048, (NF + threadsPerBlock-1) / threadsPerBlock);

__global__ void correlate(float *c, float *s, cufftComplex *a, cufftComplex *b)
{
	float temp_c = 0;
	float temp_s = 0;
	cufftComplex ccorr;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	/* We launch more threads then needed, so some do nothing */
	if (tid < NF) {
		/* Each thread processes a single frequency */
		while (tid < NS) {
			/* Normalize FFT */
			a[tid].x /= NX;
			a[tid].y /= NX;
			b[tid].x /= NX;
			b[tid].y /= NX;

			/* Correlate */
			ccorr = cuCmulf(a[tid], cuConjf(b[tid]));

			/* Sum channel over time */
			temp_c += cuCrealf(ccorr);
			temp_s += cuCimagf(ccorr);

			/* Go to next time step, NF frequencies away */
			tid += NF;
		}

		c[threadIdx.x + blockIdx.x * blockDim.x] = temp_c;
		s[threadIdx.x + blockIdx.x * blockDim.x] = temp_s;
	}
}

int main(int argc, char* argv[])
{
		int i, j;
		FILE *fo;
		char buffer[PACKET_SIZE];
		float *c, *s, *dev_c, *dev_s;
		float *a, *b, *dev_a, *dev_b;
		cufftComplex *cdev_a, *cdev_b;
		cudaError_t err;
		cufftHandle plan;

		float *ap, *bp;
		int ntotal, nnew;
		int sockfd, portno, n;
		struct sockaddr_in serv_addr;
		struct hostent *server;
		portno = 5020;
		sockfd = socket(AF_INET, SOCK_STREAM, 0);
		if (sockfd < 0)
				printf("ERROR opening socket\n");
		server = gethostbyname("131.174.192.69");

		if (server == NULL) {
				fprintf(stderr,"ERROR, no such host\n");
				exit(0);
		}
		bzero((char *) &serv_addr, sizeof(serv_addr));
		serv_addr.sin_family = AF_INET;
		bcopy((char *)server->h_addr,
						(char *)&serv_addr.sin_addr.s_addr,
						server->h_length);
		serv_addr.sin_port = htons(portno);
		if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) {
				printf("ERROR connecting\n");
				exit(1);
		}


		if (argc != 2)
		{
				printf("usage: rif_correlator <file>\n");
				exit(1);
		}

		printf("threadsPerBlock: %d\n", threadsPerBlock);
		printf("blocksPerGrid: %d\n", blocksPerGrid);

		/* Allocate memory on host */
		a = (float*) malloc(NTOT*sizeof(float)); /* Only the first N are used for FFT */
		b = (float*) malloc(NTOT*sizeof(float)); /* Only the first N are used for FFT */
		c = (float*) malloc(NF*sizeof(float));
		s = (float*) malloc(NF*sizeof(float));

		/* Allocate memory on device */
		err = cudaMalloc(&dev_c, NF*sizeof(float));
		if (err != cudaSuccess) {
				fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
				goto exit;
		}

		err = cudaMalloc(&dev_s, NF*sizeof(float));
		if (err != cudaSuccess) {
				fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
				goto exit;
		}

		err = cudaMalloc(&dev_a, N*sizeof(float));
		if (err != cudaSuccess) {
				fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
				goto exit;
		}

		err = cudaMalloc(&dev_b, N*sizeof(float));
		if (err != cudaSuccess) {
				fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
				goto exit;
		}

		err = cudaMalloc(&cdev_a, NS*sizeof(cufftComplex));
		if (err != cudaSuccess) {
				fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
				goto exit;
		}

		err = cudaMalloc(&cdev_b, NS*sizeof(cufftComplex));
		if (err != cudaSuccess) {
				fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
				goto exit;
		}

		/* Create FFT plan */
		if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: Plan creation failed\n");
				goto exit;
		}

		/* Open output file */
		fo = fopen(argv[1], "w");
		if (fo == NULL) {
				fprintf(stderr, "Error: could not open output file!\n");
				goto exit;
		}

		i = 0;
		for (;;) {

				ap = a;
				bp = b;
				ntotal = 0;
				nnew = 0;
				while (ntotal < 2*NTOT) {
						n = recv(sockfd,buffer,PACKET_SIZE*sizeof(char),0);
						nnew = n / sizeof(char);
						/*if (nnew != PACKET_SIZE) fprintf(stderr, "expected %d got %d\n", PACKET_SIZE, nnew);*/
						for (j=0; j<nnew / 2; j++) {
								*ap = (float) buffer[2*j];
								*bp = (float) buffer[2*j+1];
								ap++;
								bp++;
						}
						ntotal += nnew;
				}

				err = cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
						printf("Error %s\n", cudaGetErrorString(err));
						goto exit;
				}

				err = cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
						printf("Error %s\n", cudaGetErrorString(err));
						goto exit;
				}

				/* Perform FFT on device */
				if (cufftExecR2C(plan, dev_a, cdev_a) != CUFFT_SUCCESS){
						fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
						goto exit;
				}

				if (cudaDeviceSynchronize() != cudaSuccess){
						fprintf(stderr, "Cuda error: Failed to synchronize\n");
						goto exit;
				}

				/* Perform FFT on device */
				if (cufftExecR2C(plan, dev_b, cdev_b) != CUFFT_SUCCESS){
						fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
						goto exit;
				}

				if (cudaDeviceSynchronize() != cudaSuccess){
						fprintf(stderr, "Cuda error: Failed to synchronize\n");
						goto exit;
				}

				correlate<<<blocksPerGrid,threadsPerBlock>>>(dev_c, dev_s, cdev_a, cdev_b);

				if (cudaDeviceSynchronize() != cudaSuccess){
						fprintf(stderr, "Cuda error: Failed to synchronize\n");
						goto exit;
				}

				err = cudaMemcpy(c, dev_c, NF*sizeof(float), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
						printf("Error %s\n", cudaGetErrorString(err));
						goto exit;
				}

				err = cudaMemcpy(s, dev_s, NF*sizeof(float), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
						printf("Error %s\n", cudaGetErrorString(err));
						goto exit;
				}

				/* From sum to average on the CPU */
				for (j=0; j<NF; j++) {
						c[j] /= BATCH;
						s[j] /= BATCH;
						fprintf(fo, "%.3f\t%.3f\n", c[j], s[j]);
				}

				fflush(fo);

				i++;
		}

exit:
		/* Cleanup */
		free(a);
		free(b);
		free(s);
		free(c);
		cufftDestroy(plan);
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(cdev_a);
		cudaFree(cdev_b);
		cudaFree(dev_s);
		cudaFree(dev_c);

		/* Close file */
		fclose(fo);
		exit(1);
}

