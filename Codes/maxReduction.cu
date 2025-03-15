#include <iostream> 
#include <cuda_runtime.h>

#define MAX_VAL -INFINITY  // Changed to negative infinity for finding maximum

// !!CUDA DOES NOT SUPPORT ATOMIC MAX FOR FLOATS !!
// !!SO I FOUND THIS SOLUTION ON STACK OVERFLOW: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda !!
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ 
void VecMax(const float* input, float* output, int N){
    int griddim = gridDim.x;
    int blockwidth = blockDim.x;
    int blockidx = blockIdx.x;
    int threadidx = threadIdx.x;
    int stride = griddim * blockwidth;
    int index = (blockidx*blockwidth)+threadidx;

    // typically shared memory in this case called cache equals the sizeof threads per block
    extern __shared__ float cache[];
    // maximum temp value for each thread
    float temp_max_val = MAX_VAL;
    
    // find maximum in the assigned segment of the array
    while (index < N){
        if (input[index] > temp_max_val){
            temp_max_val = input[index];
        }
        // index hops according to the stride 
        index += stride;
    }
    // each thread according to its 'index' stores its local maximum from within its assigned portion of the array into the cache 
    cache[threadidx] = temp_max_val;
    __syncthreads();
    // at this point all the threads have saved their local max in the cache

    // The following reduction is per BLOCK basis at the end of this reduction 
    // each block will have its maximum and we will need to conduct one more reduction 
    // to find the maximum amongs all the blocks result 
    int comparison_step = blockwidth / 2;
    while (comparison_step != 0){
        // first condition is to make sure that we only compare values at thread indices that do no exceed the comparion step (it is like dividing the array into half)
        // second condition is used to update the cache index to store the larger of the two value 
        // so suppose [7,1,19,0] -> imagine [7,1 |break| 19,0]
        // we compare thread with indices less than 3 with threads indices greater than 3 
        // thread index 0 compares the value 7 with 19 and stores 19 in place of 7 so [19, 1 |break| 19, 0]
        // thread index 1 compares the value 1 with 0 and keeps its value
        // in the next iteration we use the [19,1] and use [19 |break| 1]
        // at the end cache[0] will have the maximum value for this block
        if (threadidx < comparison_step && cache[threadidx+comparison_step] > cache[threadidx]){
            cache[threadidx] = cache[threadidx+comparison_step];
        }
        // after each thread has compared its value with the value at comparison steps away. We wait for all threads to finish comparing
        __syncthreads();
        comparison_step /= 2;
    }
    // the maximum value will be at thread idx 0
    if (threadidx==0){
        atomicMax(output, cache[0]);
    }
}

int main(void){
    int N = 20;
    float *input_h, *output_h, *input_d, *output_d;

    input_h = new float[N]; 
    // for practice just fill the input with random values 
    for (int i=0; i< N; i++){
        input_h[i] =  static_cast<float>(rand() % 100);
    }
    for (int i = 0; i < N; i++) {
        std::cout << "input_h[" << i << "] = " << input_h[i] << std::endl;
    }

    cudaMalloc((void**)&input_d, N*sizeof(float));
    cudaMemcpy(input_d, input_h, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&output_d, sizeof(float));
    float initial_max = MAX_VAL;
    // cudaMemcpy expects address to var
    // initialize the output with the initial max val
    cudaMemcpy(output_d, &initial_max, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    auto sharedMemSize = threadsPerBlock*sizeof(float); // allocated shared memory for each block
    VecMax<<<blocksPerGrid,threadsPerBlock,sharedMemSize>>>(input_d,output_d,N);

    output_h = new float;
    cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Maximum value: "<< *output_h << std::endl;

    cudaFree(input_d);
    cudaFree(output_d);
    delete [] input_h;
    delete output_h;
    return 0;
}