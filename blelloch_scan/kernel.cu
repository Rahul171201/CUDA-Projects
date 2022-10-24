// Author -> Rahul Roy
// Algorithm -> Blelloch Scan Algorithm
// Step Complexity = O(logn)
// Work Complexity = O(n)

#include<iostream>
#include<time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define N 8 // no of elements of the array

// Function for down sweep
__global__ void down_sweep(int *arr, int n, int stride, int temp) {
	int currentThread = blockIdx.x * blockDim.x + threadIdx.x; // thread id
	if (currentThread >= n) // check if thread is in given range
		return;

	int q; // variable for swapping elements
	__syncthreads(); // creating synchronization barrier
	for (int s = stride; s > 0; s = s / 2) {
		if ((currentThread % temp == ((temp / 2) - 1)) && (currentThread + s < n)) {
			printf("Swap %d with %d", arr[currentThread], arr[currentThread + s]);
			atomicAdd(&q, arr[currentThread+s]);// critical section
			atomicAdd(&(arr[currentThread + s]), arr[currentThread]);// critical section
			atomicAdd(&(arr[currentThread]), q-arr[currentThread]); // critical section
		}
		temp = temp/2;
		__syncthreads(); // creating synchronization barrier
	}
	return;
}

// Function for parallel reduction
__global__ void reduction(int* arr, int n,int* temp,int* max_stride) {
	int currentThread = blockIdx.x * blockDim.x + threadIdx.x; // thread id
	int stride = 1; // stride
	temp[0] = 2; // factor of computation

	if (currentThread >= n) // check if thread is in given range
		return;

	__syncthreads(); // creating synchronization barrier
	max_stride[0] = 0;
	for (stride = 1; stride < n; stride *= 2) {
		if ((currentThread % temp[0] == ((temp[0] / 2) - 1)) && currentThread + stride < n) {
			atomicAdd(&(arr[currentThread + stride]), arr[currentThread]); // critical section
		}
		if (max_stride[0] < stride) {
			atomicMax(&max_stride[0], stride); // critical section
		}
		temp[0] = temp[0] * 2;
		//creating synchronization barrier
		__syncthreads();
	}
	temp[0] = temp[0] / 2;
	return;
}

// Function to calculate prefix sum of elements sequentially
int* calculatePrefixSumSequentially(int* arr, int* pre, int n) {
	for (int i = 0; i < n; i++) {
		pre[i] = 0;
	}
	pre[0] = arr[0];
	for (int i = 1; i < n; i++) {
		pre[i] = pre[i - 1] + arr[i];
	}
	return pre;
}

// Function to generate random array of integers
int* generateRandomArray(int* arr) {
	for (int i = 0; i < N; i++) {
		arr[i] = i;
	}
	return arr;
}

// Main function
int main() {

	srand(time(0));

	// declaring arrays 
	int* arr, * pre;
	int* temp, * max_stride;

	// assigning memory
	pre = (int*)malloc(N * sizeof(int));

	// unified memory access
	cudaMallocManaged(&arr, N * sizeof(int));
	cudaMallocManaged(&temp, 1 * sizeof(int));
	cudaMallocManaged(&max_stride, 1 * sizeof(int));

	// generate a random array
	arr = generateRandomArray(arr);

	// Print the generated array
	cout << "The generated array is :\n";
	for (int i = 0; i < N; i++) {
		cout << arr[i] << " ";
	}
	cout << "\n";

	// Calculate Prefix Sum Sequentially
	auto start = high_resolution_clock::now();
	pre = calculatePrefixSumSequentially(arr, pre, N);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by sequential part =  " << duration.count() << " microseconds\n";

	// call the kernel function to initiate parallel reduction
	int noOfBlocks = 100;
	int noOfThreads = 1000;
	start = high_resolution_clock::now();
	reduction << <noOfBlocks, noOfThreads >> > (arr, N, temp, max_stride);
	cudaDeviceSynchronize();
	cout << "The array after reduction :\n";
	for (int i = 0; i < N; i++)
		cout << arr[i] << " ";
	cout << "\n";
	arr[N - 1] = 0;

	// call the kernel function to initiate down sweep parallely
	down_sweep<<<noOfBlocks, noOfThreads>>>(arr, N, max_stride[0], temp[0]);
	cudaDeviceSynchronize();
	stop = high_resolution_clock::now();
	
	duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by parallel part =  " << duration.count() << " microseconds\n";

	cout << "The array after blelloch scan :\n";
	for (int i = 0; i < N; i++)
		cout << arr[i] << " ";
	cout << "\n";
	cout << "The last element of prefix sum calculated sequentially = " << pre[N - 2] << "\n";
	cout << "The last element of prefix sum calculated parallely = " << arr[N - 1] << "\n";
	return 0;
}
