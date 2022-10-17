#include<iostream>

using namespace std;

#define N 300 // image of size NxN

// Parallel Random matrix generation
__global__ int* cudaMatrixGeneration(int * colored_image, int n){

}

// Sequential Random matrix generation
int* generateRandomColoredImageSequentially(int* colored_image, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            int r = (rand() % (255 - 1 + 1)) + 1;
            int g = (rand() % (255 - 1 + 1)) + 1;
            int b = (rand() % (255 - 1 + 1)) + 1;
            colored_image[(i*n + j)*3] = r;
            colored_image[(i*n + j)*3+1] = g;
            colored_image[(i*n + j)*3+2] = b;
        }
    }
    return colored_image;
}

int main(){
    int *colored_image = (int*)malloc(N*N*3*sizeof(int));

    colored_image =generateRandomColoredImageSequentially(colored_image, N);

    // print the image matrix by rgb values
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         cout<<"r = "<<colored_image[(i*n+j)*3]<<" g = "<<colored_image[(i*n+j)*3+1]<<" b = "<<colored_image[(i*n+j)*3+2]<<"\n";
    //     }
    // }
}