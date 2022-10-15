#include<iostream>

using namespace std;

#define N 16

__global__ void matrixMultiplication(int *matrix, int* vector, int* result, int n) {
	int currentThread = threadIdx.x;

	for (int i = 0; i < n; i++) {
		result[currentThread] = result[currentThread] + matrix[currentThread*n+i] * vector[i];
	}
}

int main()
{
	int *matrix, *vector, *result;

	// allocate memory
	cudaMallocManaged(&matrix, N * N * sizeof(int));
	cudaMallocManaged(&vector, N * sizeof(int));
	cudaMallocManaged(&result, N * sizeof(int));

	srand(time(0));

	// generate random dense matrix
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i*N + j] = (rand() % (20 - 1 + 1)) + 1;
		}
	}

	// print the dense matrix
	cout << "The dense matrix is\n";
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout<<matrix[i*N+j]<<" ";
		}
		cout<<"\n";
	}

	cout << "\n";

	// generate random vector
	for (int i = 0; i < N; i++) {
		vector[i] = (rand() % (20 - 1 + 1)) + 1;
	}

	// print the random vector
	cout << "The vector is : \n";
	for (int i = 0; i < N; i++) {
		cout<< vector[i] <<" ";
	}
	cout << "\n";

	//call kernel function on device

	int blocksPerGrid = 1;
	int threadsPerBlock = N;

	matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(matrix, vector, result, N);
	cudaDeviceSynchronize();
	// end of call

	cout << "\n";
	cout << "The elements of resultant vector are :\n";
	for (int i = 0; i < N; i++) {
		cout << result[i] << " ";
	}
}
