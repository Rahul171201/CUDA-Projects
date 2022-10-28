#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(){
    srand(time(0));

    FILE *fp;
    fp = fopen("input.txt", "w");

    if(fp == NULL) {
        printf("file can't be opened\n");
        exit(1);
    }

    int n = 1024;
    int no_of_bins = 8;
    fprintf(fp, "%d\n%d\n", n,no_of_bins);
    for(int i=0;i<n;i++){
        fprintf(fp, "%d\n", (rand() % (255 - 0 + 1)) + 0);
    }

    fclose(fp);
}