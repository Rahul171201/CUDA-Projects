#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define N 1000 // image of size NxN

// Convert colored image to greyscale image in parallel
__global__ void convertToGreyScale(int* colored_image, double* greyscale_image, int n, double factor_r, double factor_g, double factor_b)
{
    int currentBlock_row = blockIdx.y;
    int currentBlock_col = blockIdx.x;
    int currentThread_row = currentBlock_col * blockDim.y + threadIdx.y;
    int currentThread_col = currentBlock_row * blockDim.x + threadIdx.x;
    int currentThread = currentThread_row * n + currentThread_col;

    // printf("current thread = %d", currentThread);;
    // printf("thread details : block_x=%d block_y=%d threadRow=%d threadCol=%d\n", currentBlock_row, currentBlock_col, currentThread_row, currentThread_col);
    if (currentThread < n * n)
    {
        double sum;
        sum = double(colored_image[currentThread * 3]) * factor_r + double(colored_image[currentThread * 3 + 1]) * factor_g + double(colored_image[currentThread * 3 + 2]) * factor_b;
        greyscale_image[currentThread] = sum;
    }
}

// Sequential Random matrix generation
int* generateRandomColoredImage(int* colored_image, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int r = (rand() % (255 - 1 + 1)) + 1;
            int g = (rand() % (255 - 1 + 1)) + 1;
            int b = (rand() % (255 - 1 + 1)) + 1;
            colored_image[(i * n + j) * 3] = r;
            colored_image[(i * n + j) * 3 + 1] = g;
            colored_image[(i * n + j) * 3 + 2] = b;
        }
    }
    return colored_image;
}

// Sequential conversion of colored image to grey scale
double* sequentialConversionToGreyScale(int* colored_image, int n, double factor_r, double factor_g, double factor_b)
{
    double* ans = (double*)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum;
            sum = double(colored_image[(i * n + j) * 3]) * factor_r + double(colored_image[(i * n + j) * 3 + 1]) * factor_g + double(colored_image[(i * n + j) * 3 + 2]) * factor_b;
            ans[i * n + j] = sum;
        }
    }
    return ans;
}

// Main Function
int main()
{
    srand(time(0));

    int* colored_image;
    double* greyscale_image;
    double factor_r = 0.1, factor_g = 0.7, factor_b = 0.3;
    cudaMallocManaged(&colored_image, N * N * 3 * sizeof(int));
    cudaMallocManaged(&greyscale_image, N * N * sizeof(double));

    colored_image = generateRandomColoredImage(colored_image, N);

    // print matrix
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         cout << "[ ";
    //         for (int k = 0; k < 3; k++)
    //         {
    //             cout << colored_image[(i * N + j) * 3 + k] << ", ";
    //         }
    //         cout << "] ";
    //     }
    //     cout << "\n";
    // }

    // call sequential function to convert to grey scale
    auto start = high_resolution_clock::now();
    double* sequential_greyscale_image = sequentialConversionToGreyScale(colored_image, N, factor_r, factor_g, factor_b);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by sequential grey scale function: " << duration.count() << " microseconds\n";

    // calling kernel  function to convert colored image to grey scale
    start = high_resolution_clock::now();
    dim3 blocksPerGrid(100, 100);
    dim3 threadsPerBlock(10, 10);

    convertToGreyScale << <blocksPerGrid, threadsPerBlock >> > (colored_image, greyscale_image, N, factor_r, factor_g, factor_b);
    cudaError_t err = cudaDeviceSynchronize();
    //cout << err << " is the error\n";
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by parallel grey scale function: " << duration.count() << " microseconds\n";

    // print grey scale image made sequentially
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         cout << sequential_greyscale_image[i * N + j] << " ";
    //     }
    //     cout << "\n";
    // }

    //print grey scale image made parallely
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << greyscale_image[i * N + j] << " ";
        cout << "\n";
    }

    free(sequential_greyscale_image);
    cudaFree(colored_image);
    cudaFree(greyscale_image);
}