// Author -> Rahul Roy
// Algorithm -> Hillis Scan Algorithm
// Step Complexity = O(nlogn)
// Work Complexity = O(n)

#include<iostream>
#include<time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define N 16 // no of elements of the array

//Kernel Function to calculate prefix sum usinh Hillis Scan Algorithm
__global__ void Hillis_Scan(int* arr, int n) {
	int currentThread = blockIdx.x*blockDim.x+threadIdx.x; // thread id
	int stride = 1; // stride

	if (currentThread >= n) // check if thread is in given range
		return;

	__syncthreads(); // barrier synchronization

	for (stride = 1; stride < n; stride *= 2) {
		if (currentThread + stride < n) {
			atomicAdd(&(arr[currentThread + stride]), arr[currentThread]);
		}
		__syncthreads(); // barrier synchronization
	}
	return;
}

// Function to calculate prefix sum sequentially
int* calculatePrefixSumSequentially(int* arr,int* pre, int n) {
	for (int i = 0; i < n; i++) {
		pre[i] = 0;
	}
	pre[0] = arr[0];
	for (int i = 1; i < n; i++) {
		pre[i] = pre[i-1] + arr[i];
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


int main() {

	srand(time(0));

	int* arr, *pre;
	pre = (int*)malloc(N * sizeof(int));
	cudaMallocManaged(&arr, N * sizeof(int));

	arr = generateRandomArray(arr);

	// Print the generated array
	cout << "The generated array is :\n";
	for (int i = 0; i < N; i++) {
		cout << arr[i] << " ";
	}
	cout << "\n";

	auto start = high_resolution_clock::now();
	pre = calculatePrefixSumSequentially(arr,pre,N);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by sequential part =  " << duration.count() << " microseconds\n";

	// call the kernel function to calculate sum parallely
	int noOfThreads = 1000;
	start = high_resolution_clock::now();
	Hillis_Scan << <1, noOfThreads >> > (arr, N);

	//calculateSum << <grid_size, tb_size >> > (arr, N, test);
	stop = high_resolution_clock::now();
	cudaDeviceSynchronize();

	duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by parallel part =  " << duration.count() << " microseconds\n";

	cout << "The prefix sum calculated sequentially :\n";
	for (int i = 0; i < N; i++)
		cout << pre[i] << " ";
	cout << "\n";
	cout << "The prefix sum calculated parallely :\n";
	for (int i = 0; i < N; i++)
		cout << arr[i] << " ";
	cout << "\n";
	return 0;
}
