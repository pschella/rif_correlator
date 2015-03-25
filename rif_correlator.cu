#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>

int imin(int a, int b)
{
	return (a < b ? a : b);
}

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

const int threadsPerBlock = 32;
const int blocksPerGrid = imin(32, (NF + threadsPerBlock-1) / threadsPerBlock);

__global__ void correlate(float *c, float *s, cufftComplex *a, cufftComplex *b)
{
	float temp_c = 0;
	float temp_s = 0;
	cufftComplex ccorr;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	/* We launch more threads then needed, so some do nothing */
	if (tid < NF) {
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
	FILE *fp, *fo;
	char *buffer;
	float *c, *s, *dev_c, *dev_s;
	float *a, *b, *dev_a, *dev_b;
	cufftComplex *cdev_a, *cdev_b;
	cudaError_t err;
	cufftHandle plan;

	/* Allocate memory on host */
	buffer = (char*) malloc(2*NTOT*sizeof(char));
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	c = (float*) malloc(NF*sizeof(float));
	s = (float*) malloc(NF*sizeof(float));

	/* Allocate memory on device */
	err = cudaMalloc(&dev_c, NF*sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&dev_s, NF*sizeof(float));
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
	
	err = cudaMalloc(&cdev_a, NF*BATCH*sizeof(cufftComplex));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc(&cdev_b, NF*BATCH*sizeof(cufftComplex));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
		return 1;
	}

	/* Create FFT plan */
	if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 1;	
	}

	/* Open input file */
	fp = fopen(argv[1], "rb");
	if (fp == NULL) {
		fprintf(stderr, "Error: could not open input file!\n");
		return 1;
	}

	/* Open output file */
	fo = fopen(argv[2], "w");
	if (fo == NULL) {
		fprintf(stderr, "Error: could not open output file!\n");
		return 1;
	}

	i = 0;
	while (fread(buffer, sizeof(char), 2*NTOT, fp) == 2*NTOT*sizeof(char)) {
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

		correlate<<<blocksPerGrid,threadsPerBlock>>>(dev_c, dev_s, cdev_a, cdev_b);
	
		err = cudaMemcpy(c, dev_c, NF*sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Error %s\n", cudaGetErrorString(err));
		}

		err = cudaMemcpy(s, dev_s, NF*sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Error %s\n", cudaGetErrorString(err));
		}

		/* From sum to average on the CPU */
		for (j=0; j<NF; j++) {
			c[j] /= BATCH;
			s[j] /= BATCH;
			fprintf(fo, "%d %.3f\t%.3f\n", j, c[j], s[j]);
		}

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
	fclose(fp);
	fclose(fo);
	return 0;
}

