#include <stdio.h>

__global__ 
void saxpy(int n, float a, float *x, float *y) {
    // global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // make sure thread does not access memory out of bounds 
    if (index < n){
        y[index] = a * x[index] + y[index];
    }
}

int main(void) {
    // left shift 1 by 20 ~ 2^20
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;

    // allocate mem on host 
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    // allocate mem on device, note: pass by ref
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        // float 32 bit precision
        // double 64 bit precision
        // we dont want implicit type conversion and that's why we use .f cuz that is just an overhead
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float),cudaMemcpyHostToDevice);

    // kernel launch 
    // ceiling division trick to make sure we cover all elements 
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    // get the result back device -> host
    cudaMemcpy(y, d_y, N*sizeof(float),cudaMemcpyDeviceToHost);

    float maxerror = 0.0f;
    for (int i=0; i<N; i++){
        maxerror = max(maxerror, abs(y[i]-4.0f));
    }
    printf("Max error: %f\n", maxerror);

    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
}