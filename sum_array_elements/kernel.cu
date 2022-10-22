#include<iostream>
#include<time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define N 10000 // no of elements of the array

// Function to calculate sum of elements using parallel reduction
__global__ void calculateSum(int *arr, int n, int *test) {
	int currentRow = blockIdx.y * blockDim.y + threadIdx.y;
	int currentColumn = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Current Row = %d, Current Column = %d\n", currentRow, currentColumn);
	//printf("Current Grid_x = %d, Grid_y = %d\n", gridDim.x, gridDim.y);
	//printf("Current Block_x = %d, Block_y = %d\n", blockDim.x, blockDim.y);
	int currentThread = currentRow * (gridDim.x*blockDim.x) + currentColumn;
	int stride = 1;
	int temp = 2;
	
	if (currentThread >= n)
		return;
	//printf("current_thread = %d", currentThread);
	test[currentThread] = 1;
	//printf("Current Thread = %d", currentThread);

	__syncthreads();

	for (stride = 1; stride < n; stride *= 2) {
		int indx = currentThread * temp;
		if (indx < n) {
			atomicAdd(&(arr[indx]), arr[indx + stride]);
			//arr[indx] = arr[indx] + arr[indx + stride];
		}
		temp = temp * 2;
		//printf("Current thread : %d calculated arr[%d] = %d\n", currentThread, currentThread, arr[currentThread]);
		__syncthreads();
	}

	//while (currentThread % temp == 0 && stride < n) {
		//atomicAdd(&(arr[currentThread]), arr[currentThread + stride]);
		//arr[currentThread] = arr[currentThread] + arr[currentThread + stride];
		//stride = stride * 2;
		//temp = temp * 2;
		//printf("Current thread : %d calculated arr[%d] = %d\n", currentThread, currentThread, arr[currentThread]);
		//__syncthreads();
	//}

	//__syncthreads();
	return;
}

// Function to calculate sum of elements sequentially
int calculateSumSequentially(int *arr, int n) {
	int sum = 0;
	for (int i = 0; i< n; i++) {
		sum = sum + arr[i];
	}
	return sum;
}

// Function to generate random array of integers
int* generateRandomArray(int *arr) {
	for (int i = 0; i < N; i++) {
		int num = (rand() % (10 - 1 + 1)) + 1;
		arr[i] = num;
	}
	return arr;
}

int main() {

	srand(time(0));

	int* arr;
	int* test;
	cudaMallocManaged(&arr, N * sizeof(int));
	cudaMallocManaged(&test, N * sizeof(int));

	for (int i = 0; i < N; i++)
		test[i] = 0;

	arr = generateRandomArray(arr);

	// Print the generated array
	//cout << "The generated array is :\n";
	//for (int i = 0; i < N; i++) {
	//	cout << arr[i] << " ";
	//}
	//cout << "\n";

	auto start = high_resolution_clock::now();
	int sequential_sum = calculateSumSequentially(arr, N);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by sequential part =  " << duration.count() << " microseconds\n";

	// call the kernel function to calculate sum parallely
	int tb_size = 256;
	int grid_size = int(ceil(n / tb_size));
	dim3 blocksPerGrid(100, 100);
	dim3 threadsPerBlock(10, 10);
	start = high_resolution_clock::now();
	calculateSum << <grid_size, tb_size >> > (arr, N, test);
	calculateSum << <grid_size, tb_size >> > (arr, N, test);
	stop = high_resolution_clock::now();
	cudaDeviceSynchronize();
	
	duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by parallel part =  " << duration.count() << " microseconds\n";

	for (int i = 0; i < N; i++) {
		if (test[i] == 0)
			cout << i << "\n";
	}

	int parallel_sum = arr[0];
	cout << "The sum calculated sequentially = " << sequential_sum << "\n";
	cout << "The sum calculated parallely = " << parallel_sum << "\n";
}
