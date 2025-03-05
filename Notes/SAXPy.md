# SAXPY

**What is CUDA C/C++?**

An API/interface to run functions on a GPU, using multiple threads for parallel execution.

**Who is the Host and who is the Device?**

*   **Host:** The CPU and its associated memory (RAM).
*   **Device:** The GPU and its dedicated memory.
*   The host can manage both its own memory and the device's memory.
*   CUDA kernels can also allocate memory on devices that support this feature.

**What are Kernels?**

*   Functions designed to be executed on the GPU.
*   Many GPU threads execute a kernel concurrently allowing for parallelism.

**What is a Typical Sequence of Operations?**

1.  **Declare and Allocate Memory:** Allocate memory on both the host and the device.
2.  **Initialize Host Data:** Populate the host memory with initial data.
3.  **Transfer Data:** Copy data from the host to the device.
4.  **Execute Kernel(s):** Run one or more kernels on the device.
5.  **Transfer Results:** Copy results back from the device to the host.

**What are `malloc` and `cudaMalloc`?**

*   `malloc`: Allocates memory on the **host** and returns the memory address.
*   `cudaMalloc`:  Allocates memory on the **device** and returns the memory address.

**What is the Information Between the Triple Chevrons `<<< smth1, smth2 >>>`?**

*   This is the **execution configuration**. It determines how many 'device threads' will execute the kernel in parallel.
*   In CUDA, a kernel is launched with a "grid of thread blocks".  Threads reside within these blocks. Grids can be 1, 2, or 3-dimensional.  This requires additional arguments:
    *   `smth1`:  Grid dimensions (the layout of the blocks).
    *   `smth2`: Thread dimensions (the layout of threads within each block).
*   Threads within the same block can share data quickly.

**What is the Ceiling Division Trick?**

*   Regular division (`N / 256`) rounds down.
*   To round up (ceiling division), add (`divisor - 1`) before dividing: `(N + 255) / 256`.

**Memory Management Reminder:**

*   Don't forget to deallocate memory using `cudaFree` (for device memory) and `free` (for host memory) when it's no longer needed.

**What is `__global__`?**

*   This is a declaration specifier that must be added before a kernel function definition to indicate that it will run on the GPU.

**Where do Threads Store Their Variables?**

*   Threads primarily store their variables in registers.

**How are Function Arguments Passed to a Kernel?**

*   Function arguments to kernels are passed by value by default, similar to C/C++.

**How to Allocate Work to Threads (Workers)?**

*   Each thread needs a unique global location to work on, to avoid conflicts.  CUDA provides three built-in variables of type `dim3` to help with this:
    *   `blockDim`: Dimensions of each thread block (thread dimensions).
    *   `blockIdx`: Index of the current thread block within the grid.
    *   `threadIdx`: Index of the current thread within its thread block.

**What is `dim3`?**

*   A simple struct defined by CUDA with `x`, `y`, and `z` members.

**Why is Checking Bounds Important?**

*   It's possible that the number of elements you're processing isn't evenly divisible by the thread block size.  This can lead to launching more threads than necessary, potentially exceeding array bounds.  Always check bounds to prevent errors.

**Summary: Porting C Code to CUDA**

To port a C code to CUDA, you typically need to:

1.  Add the `__global__` declaration specifier to kernel functions.
2.  Define the kernel function's logic.
3.  Specify the execution configuration (`<<<...>>>`).
4.  Use device variables (`blockDim`, `blockIdx`, `threadIdx`) to distribute work.
5.  Proceed incrementally, porting one kernel at a time.
