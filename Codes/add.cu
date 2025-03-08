#include <iostream>
#include <math.h>

__global__
void add(int n, float*x, float*y){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    // strided access: each thread will start at its own index 
    // then jump by the stride to get to the next element 
    // suppose if gridDim.x=1 and blockDim.x = 2 and n = 10
    // thread 0 will access 0, 2, 4, 6, 8
    // thread 1 will access 1, 3, 5, 7, 9
    for (int i= index; i<n; i+= stride){
        y[i] = x[i] + y[i];
    }
}

int main(void){
    int N = 1 << 20; 

    float *x, *y;

    // we'll create a unified memory space
    // the x and y pointers will be accessible from both the CPU and GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i=0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    // <<<blocksPerGrid, threadsPerBlock>>>
    add<<<numBlocks, blockSize>>>(N, x, y);

    // make the CPU wait until the GPU is done
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i=0; i<N; i++){
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max Error: "<< maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}