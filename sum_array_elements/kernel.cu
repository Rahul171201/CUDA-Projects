#include<iostream>
#include<time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define N 20 // no of elements of the array

// Function to calculate sum of elements using parallel reduction
__global__ void calculateSum(int *arr, int n) {
	int currentRow = blockIdx.y * blockDim.y + threadIdx.y; // row id
	int currentColumn = blockIdx.x * blockDim.x + threadIdx.x; // column id
	int currentThread = currentRow * (gridDim.x*blockDim.x) + currentColumn; // thread id
	int stride = 1; // stride
	int temp = 2; // computation factor
	
	if (currentThread >= n) // check if thread is in given range
		return;

	__syncthreads(); // barrier synchronization

	for (stride = 1; stride < n; stride *= 2) {
		int indx = currentThread * temp;
		if (indx < n) {
			atomicAdd(&(arr[indx]), arr[indx + stride]); // critical section
		}
		temp = temp * 2;
		__syncthreads(); // barrier synchronization
	}
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

	int* arr; // array to store the N integers

	// unified memory access
	cudaMallocManaged(&arr, N * sizeof(int));

	// generate random array of integers
	arr = generateRandomArray(arr);

	// Print the generated array
	cout << "The generated array is :\n";
	for (int i = 0; i < N; i++) {
		cout << arr[i] << " ";
	}
	cout << "\n";

	// Sequential Sum of all Array Elements
	auto start = high_resolution_clock::now();
	int sequential_sum = calculateSumSequentially(arr, N);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by sequential part =  " << duration.count() << " microseconds\n";

	// Call the kernel function to calculate sum parallely
	dim3 blocksPerGrid(100, 100);
	dim3 threadsPerBlock(10, 10);
	start = high_resolution_clock::now();
	calculateSum << <blocksPerGrid, threadsPerBlock >> > (arr, N);
	stop = high_resolution_clock::now();
	cudaDeviceSynchronize();
	
	duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by parallel part =  " << duration.count() << " microseconds\n";

	int parallel_sum = arr[0];
	cout << "The sum calculated sequentially = " << sequential_sum << "\n";
	cout << "The sum calculated parallely = " << parallel_sum << "\n";
}
