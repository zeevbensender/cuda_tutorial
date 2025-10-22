#include <stdio.h>
__global__ void cuda_hello(){
    printf("\nHello World from GPU!\n");
}

int main() {
    printf("\nBEGIN RUNNING\n");
    cuda_hello<<<1,1>>>(); 
    return 0;
}