#include<stdio.h>

#define SIZE 1024


// Kernel function to be executed by the GPU
__global__ void vectorAddition(int* a, int* b, int* c, int n) {

	int threadId = threadIdx.x;

	if(threadId < n)
		c[threadId] = a[threadId] + b[threadId];
}

int main() {

	//declaring 3 integer arrays
	int *a, *b, *c;

	// Making the variables available to the global scope (unified memory) so that it can be accessed by both the CPU and GPU
	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i+1;
		c[i] = 0;
	}

	// calling the kernel function to be executed by the GPU

	vectorAddition <<<1, SIZE>>> (a, b, c, SIZE);

	cudaDeviceSynchronize();

	// end of call

	for (int i = 0; i < 10; i++) {
		printf("%d\n", c[i]);
	}

	// free the allocated unified memory
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}

