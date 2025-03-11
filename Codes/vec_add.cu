#include <iostream> 
#include <cuda_runtime.h>

__global__ 
void vecAdd(float *a, float*b, float*c, int n){
    int block_id = blockIdx.x;
    int block_width = blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_id * block_width + thread_id;
    if (index < n){
        c[index] = a[index] + b[index];
    }

}


int main(){
    const int n = 1024;
    const int size = n*sizeof(float);
    
    // host mem allocation
    float* A_h= new float[n];
    float* B_h= new float[n];
    float* C_h= new float[n];

    for (int i=0; i<n; ++i){
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    // device mem allocation
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock-1)/threadsPerBlock;

    vecAdd<<<blocksPerGrid,threadsPerBlock>>>(A_d,B_d,C_d,n);

    cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);
    
    std::cout << "Checking Error in computation..."<<std::endl;
        for (int i=0; i<n; i++){
        float val = 3.0f;
        std::cout << "Error at index: " << i << ": " << C_h[i] - val << std::endl;
    }
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    
    return 0;
}