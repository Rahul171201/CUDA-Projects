#include<iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N 3 // image of size NxN

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

    int* colored_image;
    cudaMallocManaged(&colored_image, N * N * 3 * sizeof(int));

    colored_image =generateRandomColoredImage(colored_image, N);

    

    int no_of_blocks = 1;
    int no_of_threads = N * N * 3;
   
    //print matrix
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         cout << "[ ";
    //         for (int k = 0; k < 3; k++) {
    //             cout << colored_image[(i * N + j) * 3 + k] << ", ";
    //         }
    //         cout << "] ";
    //     }
    //     cout << "\n";
    // }

    cudaFree(colored_image);
    // print the image matrix by rgb values
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         cout<<"r = "<<colored_image[(i*n+j)*3]<<" g = "<<colored_image[(i*n+j)*3+1]<<" b = "<<colored_image[(i*n+j)*3+2]<<"\n";
    //     }
    // }
}