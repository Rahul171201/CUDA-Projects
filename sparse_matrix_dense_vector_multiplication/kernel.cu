// Author -> Rahul Roy

// GPU Used -> NVIDIA GeForce GTX 1650 Ti
// Number of SM's -> 16
// Number of cores per SM -> 1024


#include<iostream>
#include<fstream>
#include<time.h>
#include<iomanip>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>

using namespace std;
using namespace std::chrono;

#define NUM_THREADS 4
#define N 8192

// to find number of cores per SM
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2:
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3:
        cores = mp * 192;
        break;
    case 5:
        cores = mp * 128;
        break;
    case 6:
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7:
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8:
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

// kernel function to multiply sparse matrix with dense vector
__global__ void matrixMultiplication(double* matrix, double* vector, double* result, int n) {
    int currentThread = threadIdx.x;

    for (int i = 0; i < n; i++) {
        result[currentThread] = result[currentThread] + matrix[currentThread * n + i] * vector[i];
    }
}

// Function to multiply sparse matrix with dense vector
void multiplyMatrixVectorSequentially(double* matrix, double* vector, double* c1, int n) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c1[j] = c1[j] + matrix[i * n + j] * vector[j];
        }
    }

    cout << "After sequential multiplication of sparse matrix with dense vector, we get C1 :\n";
    for (int i = 0; i < n; i++) {
        cout << c1[i] << " ";
    }
    cout << "\n";

    return;
}


int main()
{
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    int no_of_cores_per_SM = getSPcores(dev);
    cout << "The number of SM's are : "<<dev.multiProcessorCount << "\n";
    cout << "The number of cores per SM are : " << no_of_cores_per_SM << "\n";

    ifstream fp("inputfile.mtx"); // read inputfile.mtx into F1

    if (!fp.good()) {
        cout << "File is not good to open";
        exit(0);
    }
    else if (!fp.is_open()) {
        cout << "Not able to open file";
        exit(0);
    }

    int noOfRows, noOfColumns, NNZ;
    fp >> noOfRows >> noOfColumns >> NNZ;

    cout << noOfRows << " " << noOfColumns << " " << NNZ << "\n";

    int n = noOfColumns;
    int* A_row, * A_column;
    double* A_value, * matrix, * vector, * c1, * c2, *large_matrix, *large_vector, *large_c1, *large_c2;

    


    A_row = (int*)malloc(n * sizeof(int));
    A_column = (int*)malloc(n * sizeof(int));
    A_value = (double*)malloc(n * sizeof(double));

    cudaMallocManaged(&matrix, n * n * sizeof(double));
    cudaMallocManaged(&vector, n * sizeof(double));
    cudaMallocManaged(&c1, n * sizeof(double));
    cudaMallocManaged(&c2, n * sizeof(double));

    cudaMallocManaged(&large_matrix, N * N * sizeof(double));
    cudaMallocManaged(&large_vector, N * sizeof(double));
    cudaMallocManaged(&large_c1, N * sizeof(double));
    cudaMallocManaged(&large_c2, N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            large_matrix[i] = (double)(rand() % (1000 - 1 + 1)) + 1;
        }
    }
    for (int i = 0; i < N; i++) {
        large_vector[i] = (double)(rand() % (1000 - 1 + 1)) + 1;
    }

    for (int i = 0; i < NNZ; i++) {
        int q1, q2;
        double val;
        fp >> q1 >> q2 >> val;
        A_row[i] = q1 - 1;
        A_column[i] = q2 - 1;
        A_value[i] = val;
    }

    fp.close();

    cout << "#Rows = " << noOfRows << "\n";
    cout << "#Cols = " << noOfColumns << "\n";
    cout << "#Non-Zeroes = " << NNZ << "\n";
    cout << "#threads = " << NUM_THREADS << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = 0;
        }
    }

    for (int i = 0; i < n; i++) {
        int x = A_row[i];
        int y = A_column[i];
        double val = A_value[i];
        matrix[x * n + y] = val;
    }


    // print the sparse matrix A
    cout << "Matrix A :\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i * n + j] << " ";
        }
        cout << "\n";
    }

    //generate random vector
    for (int i = 0; i < n; i++) {
        vector[i] = (double)(rand() % (20 - 1 + 1)) + 1;
    }

    // print the dense vector B
    cout << "The vector B is :\n";
    for (int i = 0; i < n; i++) {
        cout << vector[i] << " ";
    }
    cout << "\n";

    // sequential multiplication
    multiplyMatrixVectorSequentially(matrix, vector, c1, n);

    //call kernel function on device
    int blocksPerGrid = 1;
    int threadsPerBlock = n;
    matrixMultiplication << <blocksPerGrid, threadsPerBlock >> > (matrix, vector, c2, n);
    cudaDeviceSynchronize();
    // end of call

    // sequential multiplication
    auto start = high_resolution_clock::now();
    multiplyMatrixVectorSequentially(large_matrix, large_vector, large_c1, N);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by the sequential code is : " <<(duration.count()/1000000.00);
    cout << " seconds " << endl;
    cout << "\n";

    //call kernel function on device

    blocksPerGrid = 1;
    threadsPerBlock = N;

    double best_time = 10000000.000000;
    int best_no_of_blocks = -1;
    int best_no_of_threads = -1;

    int counter = 0;
    while (pow(2, counter) <= N) {
        start = high_resolution_clock::now();

        blocksPerGrid = pow(2, counter);
        threadsPerBlock = N/blocksPerGrid;
        matrixMultiplication << <blocksPerGrid, threadsPerBlock >> > (large_matrix, large_vector, large_c2, N);
        cudaDeviceSynchronize();

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken by "<<blocksPerGrid <<" blocks with number of threads per block = "<<threadsPerBlock<<" is " <<duration.count();
        cout << " microseconds " << endl;
        cout << "\n";

        if (counter > 2) {
            if (best_time > duration.count()) {
                best_time = duration.count();
                best_no_of_blocks = blocksPerGrid;
                best_no_of_threads = threadsPerBlock;
            }
        }
        counter++;
    }

    cout << "The best time of execution is " << best_time << ", and is acheived through " << best_no_of_blocks << " number of blocks and " << best_no_of_threads << " number of threads per block\n";

    start = high_resolution_clock::now();
   
    matrixMultiplication << <blocksPerGrid, threadsPerBlock >> > (large_matrix, large_vector, large_c2, N);
    cudaDeviceSynchronize();

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by the parallel code is : " << (duration.count()/1000000.00);
    cout << " seconds " << endl;
    cout << "\n";
    
    // end of call

    cout << "\n";
    cout << "The elements of resultant vector (using CUDA parallel algorithm) C2 are :\n";
    for (int i = 0; i < n; i++) {
        cout << c2[i] << " ";
    }
    cout << "\n";
    cout << "\n";

    // check if c1 and c2 match
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (c1[i] != c2[i])
            flag = false;
    }
    flag == true ? cout << "C1 and C2 match" : cout << "C1 and C2 do not match";

    // free memory
    free(A_row);
    free(A_column);
    free(A_value);
    cudaFree(matrix);
    cudaFree(vector);
    cudaFree(c1);
    cudaFree(c2);

}
