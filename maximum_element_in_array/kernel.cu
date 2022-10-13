#include <stdio.h>
#include <time.h>

#define SIZE 16

__global__ void findMaximum(int *h_max, int *d_max) {
	int threadId = threadIdx.x;

	int stride = SIZE / 2;

	while (stride >= 1 && threadId<stride) {
		int num = -1;
		if (d_max[threadId] > d_max[threadId + stride])
			num = d_max[threadId];
		else
			num = d_max[threadId + stride];
		d_max[threadId] = num;

		stride = stride / 2;
	}

}


// The below commented code shows a race condition which does not allow the same address space to be written by all the threads

//__global__ void findMax(int* h_max, int* d_max) {
//	int threadId = threadIdx.x;
//
//	if (threadId < SIZE) {

		// race condition
		// if (d_max[0] < h_max[threadId])
		//	d_max[0] = h_max[threadId];
	//}

//}

int main()
{
	// declaring integer arrays
	int *h_max, *d_max;
	
	// allocating memory in global spcae
	cudaMallocManaged(&h_max, SIZE * sizeof(int));
	cudaMallocManaged(&d_max, SIZE * sizeof(int));

	//generating random array
	srand(time(0));

	printf("The elements of the array are :\n");
	for (int i = 0; i < SIZE; i++) {
		int num = (rand() % (30 - 1 + 1)) + 1;
		d_max[i] = num;
		printf("%d ", d_max[i]);
	}
	printf("\n");

	// call kernel function to be executed by the GPU
	findMaximum <<<1, SIZE >>> (h_max, d_max);

	cudaDeviceSynchronize();

	// end of call

	printf("Maximum value in the array = %d", d_max[0]);

	cudaFree(h_max);
	cudaFree(d_max);

	return 0;
}
