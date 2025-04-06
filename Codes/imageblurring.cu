#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime> 
#include <vector> 

#define BLUR_SIZE 1

struct Pixel{
    unsigned char r,g,b;
}; 

std::vector<std::vector<Pixel>> generateRandomImage(int width, int height){
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<int> dist(0,255);

    // go by rows then by columns 
    std::vector<std::vector<Pixel>> image(height, std::vector<Pixel>(width));

    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            // fill in row by row 
            image[y][x] = {
                static_cast<unsigned char>(dist(rng)),
                static_cast<unsigned char>(dist(rng)),
                static_cast<unsigned char>(dist(rng)),
            };
        }
    }

    return image;
}

__global__
void blurKernel(Pixel *I_d, Pixel *out, int w, int h){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // first check to make sure that the thread is not allocated a region outside the image bounds 
    if (row<h && col<w){
        // lets compute the patch 
        int pixelcount = 0;
        float sumR = 0.0f;
        float sumG = 0.0f;
        float sumB = 0.0f;

        // iterate over a patch of (2*BLUR_SIZE+1)*(2*BLUR_SIZE+1) around the pixel of interest
        for (int pRow=-BLUR_SIZE; pRow<BLUR_SIZE+1; pRow++){
            for (int pCol=-BLUR_SIZE; pCol<BLUR_SIZE+1; pCol++){
                int patchRow = row + pRow;
                int patchCol = col + pCol;
                // second check to make sure that pixels of the patch are within image bounds 
                if (patchRow>=0 && patchRow<h && patchCol>=0 && patchCol<w){
                        int idx = patchRow*w + patchCol;
                        sumR += I_d[idx].r;
                        sumG += I_d[idx].g;
                        sumB += I_d[idx].b;
                        pixelcount++;
                }
            }
        }

        out[row*w + col].r = (unsigned char)(sumR/pixelcount);
        out[row*w + col].g = (unsigned char)(sumG/pixelcount);
        out[row*w + col].b = (unsigned char)(sumB/pixelcount);
    }
}

int main(void){
    const int width = 512;
    const int height = 512;

    int image_size = width*height*sizeof(Pixel);
    auto colouredimage = generateRandomImage(width, height);
    std::cout << "Dummy image created" << std::endl;

    // original to blur on host and device side
    Pixel *flattened_image_h = new Pixel[width*height];
    Pixel *flattened_image_d; 

    // blur it then copy over the blurred image to the host 
    Pixel *blurred_image_h = new Pixel[width*height];
    Pixel *blurred_image_d;

    for (int y=0; y<height; y++){
        for (int x=0;x<width; x++){
            flattened_image_h[y*width+x] = colouredimage[y][x];
        }
    }

    // allocate memory for the output blurred image and the original image on the device side 
    cudaMalloc((void **)&flattened_image_d, image_size);
    cudaMalloc((void **)&blurred_image_d, image_size);
    cudaMemcpy(flattened_image_d, flattened_image_h, image_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16,1);
    dim3 blocksPerGrid((height+threadsPerBlock.y-1)/threadsPerBlock.y,
                       (width+threadsPerBlock.x-1)/threadsPerBlock.x,
                       1);
    blurKernel<<<blocksPerGrid,threadsPerBlock>>>(flattened_image_d, blurred_image_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();
    
    // get the blurred back from the device to host 
    cudaMemcpy(blurred_image_h, blurred_image_d, image_size, cudaMemcpyDeviceToHost);

    std::cout << "Blurring operation complete!" << std::endl;

    cudaFree(flattened_image_d);
    cudaFree(blurred_image_d);
    delete [] flattened_image_h;
    delete [] blurred_image_h;
    return 0;
}