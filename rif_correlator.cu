#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

int imin(int a, int b)
{
	return (a < b ? a : b);
}

/* UDP port */
#define UDP_PORT_NUMBER 32000

/* Number of samples per spectrum */
#define NX 1024

/* Number of samples to average per channel per time bin */
#define N 19999744
/*20e6*/

/* Number of FFTs to perform in one batch */
#define BATCH (N / NX)

/* Number of samples after FFT */
#define NF (NX/2+1)*BATCH

/* Dimensions for thread blocks */
const int threadsPerBlock = 1024;
const int minBlocksPerGrid = 512;
const int blocksPerGrid = imin(minBlocksPerGrid, (NF+threadsPerBlock-1) / threadsPerBlock);

__global__ void correlate(float *c, float *s, cufftComplex *a, cufftComplex *b)
{
	__shared__ float cache_c[threadsPerBlock];
	__shared__ float cache_s[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp_c = 0;
	float temp_s = 0;

	cufftComplex corr;

	while (tid < NF) {
		/* Normalize FFT */
		a[tid].x /= NX;
		a[tid].y /= NX;
		b[tid].x /= NX;
		b[tid].y /= NX;

		corr = cuCmulf(a[tid], cuConjf(b[tid]));
		
		temp_c += cuCrealf(corr);
		temp_s += cuCimagf(corr);

		tid += blockDim.x * gridDim.x;
	}

	cache_c[cacheIndex] = temp_c;
	cache_s[cacheIndex] = temp_s;

	__syncthreads();

	/* Average values in cache */
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache_c[cacheIndex] += cache_c[cacheIndex + i];
			cache_s[cacheIndex] += cache_s[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	/* Store the result */
	if (cacheIndex == 0) {
		c[blockIdx.x] = cache_c[0];
		s[blockIdx.x] = cache_s[0];
	}
}

int main(int argc, char* argv[])
{
	int i, j;
	FILE *fo;
	char *buffer;
	float c, s;
	float *a, *b, *partial_c, *partial_s, *dev_a, *dev_b, *dev_partial_c, *dev_partial_s;
	cufftComplex *cdev_a, *cdev_b;
	cudaError_t err;
	cufftHandle plan;

  int sockfd, n;
  struct sockaddr_in servaddr, cliaddr;
  char mesg[1000];

  if (argc != 2)
  {
    printf("usage: rif_correlator <file>\n");
    exit(1);
  }

	printf("threadsPerBlock: %d\n", threadsPerBlock);
	printf("minBlocksPerGrid: %d\n", minBlocksPerGrid);
	printf("blocksPerGrid: %d\n", blocksPerGrid);

  /* Setup UDP port */
  sockfd=socket(AF_INET, SOCK_DGRAM, 0);

  bzero(&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr=htonl(INADDR_ANY);
  servaddr.sin_port=htons(UDP_PORT_NUMBER);
  bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr));

	/* Allocate memory on host */
	buffer = (char*) malloc(2*N*sizeof(char));
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	partial_c = (float*) malloc(blocksPerGrid*sizeof(float));
	partial_s = (float*) malloc(blocksPerGrid*sizeof(float));

	/* Allocate memory on device */
	err = cudaMalloc(&dev_partial_c, blocksPerGrid*sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&dev_partial_s, blocksPerGrid*sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&dev_a, N*sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&dev_b, N*sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}
	
	err = cudaMalloc(&cdev_a, (NX/2+1)*BATCH*sizeof(cufftComplex));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&cdev_b, (NX/2+1)*BATCH*sizeof(cufftComplex));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	/* Create FFT plan */
	if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 1;	
	}

	/* Open output file */
	fo = fopen(argv[1], "w");
	if (fo == NULL) {
		fprintf(stderr, "Error: could not open output file!\n");
		return 1;
	}

	i = 0;
	for (;;) {
    n = recvfrom(sockfd, buffer, 2*N*sizeof(char), 0, (struct sockaddr *)&cliaddr, sizeof(cliaddr));

		printf("%d\n", i);

		/* Copy data to device */
		for (j=0; j<N; j++) {
			a[j] = (float) buffer[2*j];
			b[j] = (float) buffer[2*j+1];
		}

		err = cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Error %s\n", cudaGetErrorString(err));
		}
	
		err = cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Error %s\n", cudaGetErrorString(err));
		}

		/* Perform FFT on device */
		if (cufftExecR2C(plan, dev_a, cdev_a) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
			return 1;	
		}

		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
			return 1;	
		}

		/* Perform FFT on device */
		if (cufftExecR2C(plan, dev_b, cdev_b) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
			return 1;	
		}

		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
			return 1;	
		}

		correlate<<<blocksPerGrid,threadsPerBlock>>>(dev_partial_c, dev_partial_s, cdev_a, cdev_b);
	
		err = cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Error %s\n", cudaGetErrorString(err));
		}

		err = cudaMemcpy(partial_s, dev_partial_s, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Error %s\n", cudaGetErrorString(err));
		}

		/* Finish partial sums on the CPU */
		c = 0;
		s = 0;
		for (j=0; j<blocksPerGrid; j++) {
			c += partial_c[j];
			s += partial_s[j];
		}
		c /= BATCH;
		s /= BATCH;

		fprintf(fo, "%.3f\t%.3f\n", c, s);

		i++;
	}

	/* Cleanup */
	free(a);
	free(b);
	free(buffer);
	cufftDestroy(plan);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(cdev_a);
	cudaFree(cdev_b);

	/* Close file */
	fclose(fo);
	return 0;
}

