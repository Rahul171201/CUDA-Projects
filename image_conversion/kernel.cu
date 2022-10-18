#include<iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N 3 // image of size NxN

// Convert colored image to greyscale image
__global__ void convertToGreyScale(int* colored_image, double* greyscale_image, int n, double factor_r, double factor_g, double factor_b) {

    int currentThread = threadIdx.x;
    printf("thread number = %d \n", currentThread);
    if (currentThread < n * n) {
        double sum;
        sum = double(colored_image[currentThread * 3]) * factor_r + double(colored_image[currentThread * 3 + 1]) * factor_g + double(colored_image[currentThread * 3 + 2]) * factor_b;
        greyscale_image[currentThread] = sum;
    }
}

// Sequential Random matrix generation
int* generateRandomColoredImage(int* colored_image, int n) {
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
    srand(time(0));

    int* colored_image;
    double* greyscale_image;
    cudaMallocManaged(&colored_image, N * N * 3 * sizeof(int));
    cudaMallocManaged(&greyscale_image, N * N * sizeof(double));

    colored_image = generateRandomColoredImage(colored_image, N);

    //print matrix
//   for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            cout << "[ ";
//            for (int k = 0; k < 3; k++) {
//                cout << colored_image[(i * N + j) * 3 + k] << ", ";
//            }
//            cout << "] ";
//        }
//        cout << "\n";
//    }

    // converting image to grey scale using device kernel function
    int no_of_blocks = 1;
    int no_of_threads = N * N * 3;
    double factor_r = 0.1, factor_g = 0.7, factor_b = 0.3;
    convertToGreyScale << <no_of_blocks, no_of_threads >> > (colored_image, greyscale_image, N, factor_r, factor_g, factor_b);
    cudaDeviceSynchronize();

    //print matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << greyscale_image[i * N + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(colored_image);
    cudaFree(greyscale_image);
    // print the image matrix by rgb values
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         cout<<"r = "<<colored_image[(i*n+j)*3]<<" g = "<<colored_image[(i*n+j)*3+1]<<" b = "<<colored_image[(i*n+j)*3+2]<<"\n";
    //     }
    // }
}