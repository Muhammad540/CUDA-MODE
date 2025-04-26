/*
Objective: compute softmax function for an array of 32 bit fp number
input x: size(n)
softmax(x): size(n)
    sigma(x)_i = exp(x_i) / sum|j=1 to n|(exp(x_j))

use max trick to avoid overflow for fp representation because of exp(x_i) 
subtract each x_i with max(x) so that the maximum value is exp(0) = 1 
softmax is invariant under addition of the same constant to every element
*/ 
#include <cuda_runtime.h>
#include <iostream> 
#include <cfloat>
#include <cmath>
#include <vector>
#include <algorithm> 
#include <numeric>
#include <chrono>


__global__
void partialMaxKernel(const float* input, float *d_partial_maxs, int N){
    extern __shared__ float shared_data[];
    
    int threadid    = threadIdx.x;
    int blockid     =  blockIdx.x;
    int blockwidth  =  blockDim.x;

    int global_index = blockwidth*blockid + threadid;

    if (global_index < N){
        shared_data[threadid] = input[global_index];
    } else {
        shared_data[threadid] = -FLT_MAX;
    }
    
    __syncthreads();

    // Intra-Block Max Reduction 
    for (unsigned int stride = blockwidth/2; stride > 0; stride>>=1){
        if (threadid < stride){
            shared_data[threadid] = fmaxf(shared_data[threadid], shared_data[threadid+stride]);
        }
        __syncthreads();
    }

    if (threadid==0){
        d_partial_maxs[blockid]=shared_data[0];
    }
}

__global__
void finalMaxKernel(const float* d_partial_maxs, float *d_global_max, int numPartialVals){
    extern __shared__ float shared_data[];
    
    int threadid    = threadIdx.x;
    int blockid     =  blockIdx.x;
    int blockwidth  =  blockDim.x;

    int global_index = blockwidth*blockid + threadid;

    if (global_index < numPartialVals){
        shared_data[threadid] = d_partial_maxs[global_index];
    } else {
        shared_data[threadid] = -FLT_MAX;
    }
    
    __syncthreads();

    // Intra-Block Max Reduction 
    for (unsigned int stride = blockwidth/2; stride > 0; stride>>=1){
        if (threadid < stride){
            shared_data[threadid] = fmaxf(shared_data[threadid], shared_data[threadid+stride]);
        }
        __syncthreads();
    }

    if (threadid==0){
        *d_global_max=shared_data[0];
    }
}

__global__ 
void expAndPartialSumKernel(const float *d_input, float *d_global_max, float *d_partial_sums, int N){
    /*
    extern __shared__ declares a dynamically sized shared memory array whose size is specified at kernel launch, 
    while __shared__ declares a statically sized variable or array whose size is fixed at compile time.
    */
    extern __shared__ float shared_data[];
    __shared__ float block_global_max; 

    int threadid    = threadIdx.x;
    int blockid     =  blockIdx.x;
    int blockwidth  =  blockDim.x;

    int global_index = blockwidth*blockid + threadid;

    if (threadid == 0){
        block_global_max = *d_global_max;
    }
    __syncthreads(); // wait all threads before block's global max is updated
    
    if (global_index < N){
        float val = d_input[global_index];
        shared_data[threadid] = expf(val - block_global_max);
    } else {
        shared_data[threadid] = 0.0f;
    }
    __syncthreads(); // ensure shared data is filled with exponentiated values or 0.0f
    
    for (unsigned int stride = blockwidth/2; stride > 0; stride >>=1){ // bit manipulation (shift right) 8 -> 4 -> 2  
        if (threadid < stride){
            shared_data[threadid] = shared_data[threadid] + shared_data[threadid+stride]; 
        }
        __syncthreads();
    }
    if (threadid == 0){
        d_partial_sums[blockid] = shared_data[0];
    }
}

__global__
void partialSumKernel(const float* input, float *d_partial_sums, int N){
    extern __shared__ float shared_data[];
    
    int threadid    = threadIdx.x;
    int blockid     =  blockIdx.x;
    int blockwidth  =  blockDim.x;

    int global_index = blockwidth*blockid + threadid;

    if (global_index < N){
        shared_data[threadid] = input[global_index];
    } else {
        shared_data[threadid] = 0.0f;
    }
    
    __syncthreads();

    // Intra-Block Sum Reduction 
    for (unsigned int stride = blockwidth/2; stride > 0; stride>>=1){
        if (threadid < stride){
            shared_data[threadid] = shared_data[threadid]+shared_data[threadid+stride];
        }
        __syncthreads();
    }

    if (threadid==0){
        d_partial_sums[blockid]=shared_data[0];
    }
}

__global__ 
void finalSumKernel(const float *d_partial_sums, float *d_global_sum, int numPartialVals){
    /*
    extern __shared__ declares a dynamically sized shared memory array whose size is specified at kernel launch, 
    while __shared__ declares a statically sized variable or array whose size is fixed at compile time.
    */
    extern __shared__ float shared_data[];

    int threadid    = threadIdx.x;
    int blockid     =  blockIdx.x;
    int blockwidth  =  blockDim.x;

    int global_index = blockwidth*blockid + threadid;

    if (global_index < numPartialVals){
        shared_data[threadid] = d_partial_sums[global_index];
    } else {
        shared_data[threadid] = 0.0f;
    }
    __syncthreads(); // ensure shared data is filled with exponentiated values or 0.0f
    
    for (unsigned int stride = blockwidth/2; stride > 0; stride >>=1){ // bit manipulation (shift right) 8 -> 4 -> 2  
        if (threadid < stride){
            shared_data[threadid] = shared_data[threadid] + shared_data[threadid+stride]; 
        }
        __syncthreads();
    }
    if (threadid == 0){
        *d_global_sum = shared_data[0];
    }
}

__global__
void divideKernel(const float *d_input, const float *d_global_max, const float *d_global_sum, float *d_output, int N){
    int threadid    = threadIdx.x;
    int blockid     =  blockIdx.x;
    int blockwidth  =  blockDim.x;

    int global_index = blockwidth*blockid + threadid;

    if (global_index < N){
        float global_max = *d_global_max;
        float global_sum = *d_global_sum;
    
        float val = d_input[global_index];
        float exp_val = expf(val - global_max);
        d_output[global_index] = exp_val/global_sum;
    }

}

int main(void){
    // host side stub
    int N = 10;
    float *input_h, *output_h, *input_d, *output_d, *d_partial_maxs, *d_partial_sums;
    float *d_global_max;
    float *d_global_sum;
    
    input_h  = new float[N];
    output_h = new float[N];
    
    for (int i=0; i<N; i++){
        input_h[i] = i;
    }

    // ——— CPU Softmax & Timing ———  
    auto cpu_start = std::chrono::high_resolution_clock::now();  
    std::vector<float> cpu_out(N);  
    // find max on CPU  
    float cpu_max = input_h[0];  
    for (int i = 1; i < N; ++i)  
        cpu_max = std::max(cpu_max, input_h[i]);  

    // subtract, exp, sum  
    std::vector<float> tmp(N);  
    float cpu_sum = 0.0f;  
    for (int i = 0; i < N; ++i) {  
        tmp[i] = std::exp(input_h[i] - cpu_max);  
        cpu_sum += tmp[i];  
    }  
    for (int i = 0; i < N; ++i) {  
        cpu_out[i] = tmp[i] / cpu_sum;  
    }  
    auto cpu_end = std::chrono::high_resolution_clock::now();  
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();  

    // ——— GPU Timing ———  
    cudaEvent_t g_start, g_stop;  
    cudaEventCreate(&g_start);  
    cudaEventCreate(&g_stop);  
    cudaEventRecord(g_start);  

    cudaMalloc((void **)&input_d,  N*sizeof(float));
    cudaMalloc((void **)&output_d, N*sizeof(float));
    
    cudaMalloc((void **)&d_global_max, sizeof(float));
    cudaMalloc((void **)&d_global_sum, sizeof(float));
    
    cudaMemcpy(input_d, input_h, N*sizeof(float), cudaMemcpyHostToDevice);

    // kernel setup
    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
    
    cudaMalloc((void **)&d_partial_sums,  blocksPerGrid*sizeof(float));
    cudaMalloc((void **)&d_partial_maxs,  blocksPerGrid*sizeof(float));
    
    // --------------------------------------- MAX Reduction 
    // get the partial max (gridsize, blocksize, sharedMem Size) 
    partialMaxKernel<<<blocksPerGrid, threadsPerBlock,threadsPerBlock*sizeof(float) >>>(input_d, d_partial_maxs,N);
    
    float *d_current_input = d_partial_maxs;
    int N_intermediate = blocksPerGrid;
    float *d_temp_output = nullptr;

    while (N_intermediate > threadsPerBlock){
        int blocks_for_this_pass = (N_intermediate+threadsPerBlock-1)/threadsPerBlock;
        
        cudaMalloc((void **)&d_temp_output,  blocks_for_this_pass*sizeof(float));

        partialMaxKernel<<<blocks_for_this_pass, threadsPerBlock,threadsPerBlock*sizeof(float) >>>(
            d_current_input, d_temp_output,N_intermediate);
        cudaDeviceSynchronize();
        
        if (d_current_input != d_partial_maxs){
            cudaFree(d_current_input);
        }

        d_current_input = d_temp_output;
        N_intermediate = blocks_for_this_pass;
        d_temp_output = nullptr;
    }
    //--------- FINAL Max
    finalMaxKernel<<<1,threadsPerBlock,threadsPerBlock*sizeof(float)>>>(d_current_input, d_global_max, N_intermediate);
    cudaDeviceSynchronize();

    if (d_current_input != d_partial_maxs){
            cudaFree(d_current_input);
    }

    //--------------------------------------EXP and SUM Reduction
    expAndPartialSumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(input_d, d_global_max, d_partial_sums,N);

    d_current_input = d_partial_sums;
    N_intermediate = blocksPerGrid;
    d_temp_output = nullptr;

    while (N_intermediate > threadsPerBlock){
        int blocks_for_this_pass = (N_intermediate+threadsPerBlock-1)/threadsPerBlock;
        
        cudaMalloc((void **)&d_temp_output,  blocks_for_this_pass*sizeof(float));

        partialSumKernel<<<blocks_for_this_pass, threadsPerBlock,threadsPerBlock*sizeof(float) >>>(d_current_input, d_temp_output, N_intermediate);
        cudaDeviceSynchronize();
        
        if (d_current_input != d_partial_sums){
            cudaFree(d_current_input);
        }

        d_current_input = d_temp_output;
        N_intermediate = blocks_for_this_pass;
        d_temp_output = nullptr;
    }
    // get the final max 
    finalSumKernel<<<1,threadsPerBlock,threadsPerBlock*sizeof(float)>>>(d_current_input, d_global_sum, N_intermediate);
    cudaDeviceSynchronize();

    if (d_current_input != d_partial_sums){
            cudaFree(d_current_input);
    }

    // get the softmax
    divideKernel<<<blocksPerGrid, threadsPerBlock>>>(input_d,d_global_max, d_global_sum, output_d,N);
    cudaDeviceSynchronize();

    cudaMemcpy(output_h, output_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(g_stop);  
    cudaEventSynchronize(g_stop);  
    float gpu_ms = 0;  
    cudaEventElapsedTime(&gpu_ms, g_start, g_stop);  

    // ——— Verification ———  
    const float EPS = 1e-4f;  
    bool ok = true;  
    for (int i = 0; i < N; ++i) {  
        if (std::fabs(cpu_out[i] - output_h[i]) > EPS) {  
            ok = false;  
            std::cout << "Mismatch at i="<<i  
                      <<": cpu="<<cpu_out[i]  
                      <<" gpu="<<output_h[i]<<std::endl;  
            break;  
        }  
    }  

    std::cout << (ok ? "RESULT: PASSED\n" : "RESULT: FAILED\n");  
    std::cout << "CPU time: " << cpu_ms << " ms\n";  
    std::cout << "GPU time: " << gpu_ms << " ms\n";  
    if (gpu_ms > 0)  
        std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n";  

    // destroy events  
    cudaEventDestroy(g_start);  
    cudaEventDestroy(g_stop); 

    // free mem
    cudaFree(input_d);
    cudaFree(d_partial_maxs);
    cudaFree(d_global_max);
    cudaFree(d_partial_sums);
    cudaFree(d_global_sum);
    cudaFree(output_d);
    delete [] input_h;
    delete [] output_h;
    return 0;   
}