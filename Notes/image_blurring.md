# Image Blurring Implementation Notes

In image blurring, the value of each pixel in the output image is calculated by averaging the values of a patch of neighboring pixels from the input image. A proper implementation might use convolution with specific filters (like Gaussian), I'm keeping it simple with a basic box blur.

My implementation uses a straightforward approach:
1. For each pixel in the original image, I define a square region centered on that pixel
2. calculate the average of all pixel values within this region 
3. The resulting average becomes the new value for that pixel in the blurred image

For example, to compute the blurred value at pixel 'y' in this 3x3 region:

    x a a a x
    x a y a x
    x a a a x   
    x x x x x 

calculate the average of all 'a' pixels plus the 'y' pixel itself. This means I'm including 9 pixels in my average (the center pixel plus its 8 neighbors).

Unlike more sophisticated blurring techniques like Gaussian blur, my implementation doesn't weight pixels based on their distance from the center - all pixels in the region contribute equally to the average. This is sometimes called a "box blur" because the kernel has a box shape with uniform weights.

One important detail: for pixels near the edges of the image, the blur region might extend beyond the image boundaries. In these cases, I only include pixels that fall within the image - this ensures we don't try to access pixels that don't exist.

Just like in the 'convertToGrayScale' kernel, each CUDA thread is responsible for calculating the value for exactly one output pixel. This one-to-one mapping between threads and output pixels makes for a clean parallelization strategy.

The BLUR_SIZE parameter defines how large the blur region is. With BLUR_SIZE=1, we get a 3×3 region (the center pixel plus 1 pixel in each direction). 