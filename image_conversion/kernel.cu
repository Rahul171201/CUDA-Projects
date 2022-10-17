#include<iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N 3 // image of size NxN

__global__ void init_stuff(curandState* state) {
    int currentThread = threadIdx.x;
    curand_init(1337, currentThread, 0, &state[currentThread]);
}

// Parallel Random matrix generation
__global__ void cudaMatrixGeneration(int* colored_image, int n, curandState *state) {
    int currentThread = threadIdx.x;
    printf("thread id :%d\n", currentThread);
    if (currentThread < n) {
       
        colored_image[currentThread] = curand_uniform(&state[currentThread]);
    }
}

// Sequential Random matrix generation
int* generateRandomColoredImageSequentially(int* colored_image, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
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

int main() {

   

    int* colored_image = (int*)malloc(N * N * 3 * sizeof(int));

    //colored_image =generateRandomColoredImageSequentially(colored_image, N);

    cudaMallocManaged(&colored_image, N * N * 3 * sizeof(int));

    int no_of_blocks = 1;
    int no_of_threads = N * N * 3;
    
    curandState* d_state;
    cudaMallocManaged(&d_state, no_of_blocks*no_of_threads);
    init_stuff << <no_of_blocks, no_of_threads >> > (d_state);
    cudaDeviceSynchronize();
    cudaMatrixGeneration << <no_of_blocks, no_of_threads >> > (colored_image, N, d_state);
    cudaDeviceSynchronize();
    cudaFree(d_state);
    //print matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << "[ ";
            for (int k = 0; k < 3; k++) {
                cout << colored_image[(i * N + j) * 3 + k] << ", ";
            }
            cout << "] ";
        }
        cout << "\n";
    }

    // print the image matrix by rgb values
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         cout<<"r = "<<colored_image[(i*n+j)*3]<<" g = "<<colored_image[(i*n+j)*3+1]<<" b = "<<colored_image[(i*n+j)*3+2]<<"\n";
    //     }
    // }
}