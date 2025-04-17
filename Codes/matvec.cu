// Matrix vector multiplication 
// Matrix A of dim M x K
// Vector B of dim K x 1 
// Output Vector C 
// Vector C = AB of dim M x 1 
// Matrix A is stored in row major order 

#include <cuda_runtime.h> 
#include <vector>
#include <cstdlib> // For rand()
#include <iostream> // Optional: for printing results

__global__
void matVec(const float* A, const float* B, float* C, size_t m, size_t k){
    int block_id_y = blockIdx.y;
    int block_width_y = blockDim.y;
    int thread_id_y = threadIdx.y;
    // get the global row index for this block 
    int row = (block_id_y*block_width_y) + thread_id_y;

    if (row < m){
        float local_dot_prod = 0.0f;
        for (int i=0; i < k; i++){
            local_dot_prod += A[row*k+i] * B[i];
        }
        C[row] = local_dot_prod;
    }
}

int main(){
    // Shape of Matrix  
    const int M{2332};
    const int K{2234};
    // size of Matrix and vector 
    size_t size_A = (size_t)M*K*sizeof(float);
    size_t size_B = (size_t)K*sizeof(float);
    size_t size_C = (size_t)M*sizeof(float);
    // create 1d equivalent 
    std::vector<float> A_h(M*K);
    std::vector<float> B_h(K);
    std::vector<float> C_h(M);

    for (int i=0; i< M*K; ++i){
        A_h[i] = (float)(rand() % 100) / 100.0f;
    }
    
    for (int i=0; i< K; ++i){
        B_h[i] = (float)(rand() % 100) / 100.0f;
    }
    // create devices pointers 
    float* A_d; 
    float* B_d;
    float* C_d; 
    // allocate memory on the device 
    cudaMalloc((void**)&A_d, size_A);
    cudaMalloc((void**)&B_d, size_B);
    cudaMalloc((void**)&C_d, size_C);
    // copy data from host to device 
    cudaMemcpy(A_d, A_h.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), size_B, cudaMemcpyHostToDevice);
    // define the kernel execution configuration  
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid( (M+threadsPerBlock.x-1) / threadsPerBlock.x);
    matVec<<<blocksPerGrid,threadsPerBlock>>>(A_d,B_d,C_d,M,K);

    cudaDeviceSynchronize();
    // copy the memory back to the host  
    cudaMemcpy(C_h.data(), C_d,size_C,cudaMemcpyDeviceToHost);
    // free up the data on the device 
    cudaFree(A_d);
    cudaFree(B_d);
    return 0;
}
