/*
Operation: 1D Average Pooling on an input Tensor

H: matrix input size 
*/
#include <cuda_runtime.h>
#include <vector> 
#include <iostream>

using namespace std;

__global__ 
void OneDAvgPool(const float *input, int kernel_size, int stride, int padding, float *output, size_t H){
    // starting position that the current thread will access
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if ((size_t)index < (H+2*padding-kernel_size)/stride+1){
        float sum = 0.0f;
        // my thinking is that each thread will run this for loop 
        // each thread is responsible for a specific window size
        for (int i=0;i<kernel_size;i++){
            int idx = (stride*index)+i-padding;
            if (idx >= 0 && idx < H){
                sum+=input[idx];
            } else {
                sum += 0.0f;
            }
        }
        output[index] = sum/kernel_size;        
    }
}

int main(){
    // host side data: H, input, kernel size, stride , padding, output,  
    size_t H = 100;
    vector<float> M_h;
    M_h.reserve(H); // reserve just sets the capacity the size of the vector is still 0 until you push back smth
    for (int i=0; i<H; i++){
        M_h.push_back(static_cast<float>(i));
    }
    std::cout << "Input M_h:\n";
    for (size_t i = 0; i < M_h.size(); i++) {
        std::cout << "M_h[" << i << "] = " << M_h[i] << "\n";
    }
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    vector<float> Out_h((H+2*padding-kernel_size)/stride+1);

    // device side data: input, output      
    float *M_d;
    float *Out_d;

    cudaMalloc((void **)&M_d, sizeof(float)*H);
    cudaMalloc((void **)&Out_d, sizeof(float)*((H+2*padding-kernel_size)/stride+1));

    cudaMemcpy(M_d,M_h.data(),sizeof(float)*H,cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(64,1,1);
    dim3 blocksPerGrid( (H+threadsPerBlock.x-1) / threadsPerBlock.x);
    
    OneDAvgPool<<<blocksPerGrid, threadsPerBlock>>>(M_d, kernel_size, stride, padding, Out_d, H);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed with error: " << cudaGetErrorString(err) << std::endl; 
    }
    cudaDeviceSynchronize();
    
    cudaMemcpy(Out_h.data(), Out_d, sizeof(float) * ((H + 2 * padding - kernel_size) / stride + 1), cudaMemcpyDeviceToHost);
    std::cout << "Output:\n";
    for (int i = 0; i < (H + 2 * padding - kernel_size) / stride + 1; i++) {
        std::cout << "Out_h[" << i << "] = " << Out_h[i] << "\n";
    }
    cudaFree(M_d);
    cudaFree(Out_d);
    return 0;
}