// Matrix A of dim M x N
// Matrix B of dim N x K 
// Matrix C = A x B of dim M x K
// assumption: all matrices are stored in row major format (elements in a row will be placed in consecutive memory locations) 

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>  
#include <ctime>    

template<typename T>
__global__
void matMul(const T *A, const T *B, T *C, int M, int width, int K){
    // row index of current thread in Matrix C (& corresponding row A)
    int row = (blockIdx.y*blockDim.y) + threadIdx.y;
    // col index of current thread in Matrix C (& corresponding col B)
    int col = (blockIdx.x*blockDim.x) + threadIdx.x;

    if ( row < M && col < K){
        T temp_sum = 0.0f;
        // now we perform vec-vec product: row of A * col of B
        for (int q=0; q < width; q++){ // q is used to iterate through specific row/col of A/B
            temp_sum += A[row*width+q] * B[q*K+col];
        }
        // for C matrix K is the width
        C[row*K+col] = temp_sum;
    }
}

int main(void){
    const int M{100};
    const int N{100};
    const int K{100};
    int size_A = M*N * sizeof(int);
    int size_B = N*K * sizeof(int);
    int size_C = M*K * sizeof(int);

    srand(static_cast<unsigned int>(time(0)));

    // using flat 1D vector to represent 2D matrices
    std::vector<int> A_h(M * N); 
    std::vector<int> B_h(N * K);        
    std::vector<int> C_h(M * K, 0);

    for (int i = 0; i < M * N; ++i) {
        A_h[i] = rand() % 100; 
    }
    for (int i = 0; i < N * K; ++i) {
        B_h[i] = rand() % 100; 
    }

    int *A_d; 
    int *B_d; 
    int *C_d; 

    // allocate mem on gpu
    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_C);

    // copy the data cpu -> gpu 
    cudaMemcpy(A_d, A_h.data(),size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(),size_B,cudaMemcpyHostToDevice);

    // dim3 (x,y,z)
    dim3 threadsPerBlock(32,32,1);
    // x -> width of block (K)
    // y -> height of block (M)
    dim3 blocksPerGrid((K+threadsPerBlock.x-1)/threadsPerBlock.x,
                       (M+threadsPerBlock.y-1)/threadsPerBlock.y,
                       1);
    matMul<int><<<blocksPerGrid,threadsPerBlock>>>(A_d,B_d,C_d,M,N,K);

    cudaDeviceSynchronize();
    cudaMemcpy(C_h.data(),C_d,size_C,cudaMemcpyDeviceToHost);
    
    for (int i=0; i<M; i++){
        for (int j=0; j<K; j++){
            std::cout << C_h[i*K+j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}