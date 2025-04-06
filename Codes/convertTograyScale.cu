// kernel to convert a coloured image to grayscale
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime>
#include <vector> 

struct Image{
    unsigned char r,g,b;
};

std::vector<std::vector<Image>> generateRandomImage(int width, int height){
    // this method creates a dummy image 
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<int> dist(0,255);

    std::vector<std::vector<Image>> image(height, std::vector<Image>(width));

    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            image[y][x] = {
                static_cast<unsigned char>(dist(rng)),
                static_cast<unsigned char>(dist(rng)),
                static_cast<unsigned char>(dist(rng))
            };
        }
    }
    
    return image;
}

__global__
void convertTograyScale(Image *P_d, Image *P_out, int width, int height){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col < width && row < height){
        int grayoffset = row*width + col;
        unsigned char r = P_d[grayoffset].r;
        unsigned char g = P_d[grayoffset].g;
        unsigned char b = P_d[grayoffset].b;

        unsigned char grayValue = 0.21f*r + 0.71f*g + 0.07f*b;

        P_out[grayoffset].r = grayValue;
        P_out[grayoffset].g = grayValue;
        P_out[grayoffset].b = grayValue;
    }
}

int  main(void){
    const int width = 512;
    const int height = 512;
    int image_size = width*height*sizeof(Image);
    auto colouredimage = generateRandomImage(width, height);
    std::cout << "image created" << std::endl;

    Image *flattened_image_h = new Image[width * height];
    Image *flattened_image_d;

    Image *grayed_image_h = new Image[width * height];
    Image *grayed_image_d;
    // since we cannot pass a vector, flatten out the image into an array 
    for (int y=0;y<height;y++){ // go over the rows 
        for (int x=0;x<width;x++){ // go over the cols
            flattened_image_h[y*width+x]=colouredimage[y][x];
        }
    }

    cudaMalloc((void**)&flattened_image_d, image_size);
    cudaMalloc((void**)&grayed_image_d, image_size);
    cudaMemcpy(flattened_image_d, flattened_image_h, image_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16,1);
    dim3 blocksPerGrid((height+threadsPerBlock.y-1)/threadsPerBlock.y,
                       (width+threadsPerBlock.x-1)/threadsPerBlock.x,
                       1);
    convertTograyScale<<<blocksPerGrid,threadsPerBlock>>>(flattened_image_d,grayed_image_d,width,height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(grayed_image_h,grayed_image_d,image_size,cudaMemcpyDeviceToHost);

    std::cout << "Verifying grayscale conversion..." << std::endl;
    bool conversion_correct = true;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            unsigned char r = flattened_image_h[idx].r;
            unsigned char g = flattened_image_h[idx].g;
            unsigned char b = flattened_image_h[idx].b;
            
            unsigned char grayValue = 0.21f*r + 0.71f*g + 0.07f*b;
            
            if (grayed_image_h[idx].r != grayValue || 
                grayed_image_h[idx].g != grayValue || 
                grayed_image_h[idx].b != grayValue) {
                std::cout << "Mismatch at pixel (" << x << "," << y << ")" << std::endl;
                std::cout << "Original: R=" << (int)r << ", G=" << (int)g << ", B=" << (int)b << std::endl;
                std::cout << "Expected grayscale value: " << (int)grayValue << std::endl;
                std::cout << "Actual: R=" << (int)grayed_image_h[idx].r << ", G=" << (int)grayed_image_h[idx].g 
                          << ", B=" << (int)grayed_image_h[idx].b << std::endl;
                conversion_correct = false;
                break;
            }
        }
        if (!conversion_correct) break;
    }

    if (conversion_correct) {
        std::cout << "Grayscale conversion verified successfully!" << std::endl;
    }

    cudaFree(flattened_image_d);
    cudaFree(grayed_image_d);
    delete[] flattened_image_h;
    delete[] grayed_image_h; 
    return 0;
}