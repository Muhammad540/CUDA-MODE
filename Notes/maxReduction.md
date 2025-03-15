# Finding the Maximum Value in an Array with Parallel Reduction

*   We have a huge array, and we need to find the single largest number in it. 

*   The idea is to get each thread within a block to work on a small piece, or "chunk," of the array and find the maximum value *within that chunk*.

*   To figure out the size of each chunk, we can use this formula:

    $$\text{Chunk Size} = \frac{N}{\text{threadsPerBlock} \times \text{blocksPerGrid}}$$

    Where:
    *   `N` is the total number of elements in the array.
    *   `threadsPerBlock` is the number of threads in each block, which in CUDA is `blockDim.x`.
    *   `blocksPerGrid` is the number of blocks in the grid, which is `gridDim.x`.

*   Let's think about how this works block by block.  Each thread block gets its own little workspace called **shared memory**.

*   **Shared memory** is super fast and only accessible to the threads within that *specific* block. It's like a local scratchpad for each group of threads.

*   If you want different blocks to talk to each other, you'd typically need to use **global memory** (which is slower) or some clever synchronization tricks. But for now, we'll focus on what happens inside each block and then how blocks coordinate to get the final answer.

*   So, in our max reduction:
    *   We'll give each thread a **subset** of the input array to look at.
    *   Each thread will find the **maximum value** in its assigned little piece of the data.
    *   Then, each thread will store its own **local maximum** in a location within the shared memory of its block.

*   **Example to make it clearer:**

    Let's say we have an array with **1024 elements** and we're using a single block with **256 threads**.

    *   Each thread will be responsible for a chunk of `1024 elements / 256 threads = 4 elements`.

    *   **Thread 0** will check elements 0, 1, 2, and 3 and find the biggest one among them.
    *   **Thread 1** will check elements 4, 5, 6, and 7 and find its maximum.
    *   And so on...

    *   After every thread in the block has figured out the max of its small chunk, we're ready to move on to the **parallel reduction** step within the block.
### Parallel Reduction

*   So, we've got to the point where each thread in a block has found the maximum value within its assigned chunk of the array and stored it in shared memory.  Let's call this shared memory array `cache`.  At this point, `cache` within each block holds a bunch of *local* maximums, one from each thread (or maybe fewer if threads processed multiple elements initially).

*   Now, the goal of **parallel reduction** is to efficiently combine these local maximums within each block to find the *overall* maximum for that block. This happens in two stages:

    1.  **Intra-Block Parallel Reduction (Within Each Block):**  Reducing the local maximums *within* the shared memory of each block to get a single maximum *per block*.
    2.  **Inter-Block Parallel Reduction (Final Stage):**  Taking the maximums calculated by each block and reducing them to find the final, overall maximum of the entire array.

#### 1. Intra-Block Parallel Reduction (Within Each Block)

*   **The Problem:**  Inside each block's shared memory (`cache`), we have multiple local maximums. We need to reduce these down to just *one* maximum value that represents the biggest number found within that entire block's assigned portion of the original array.

*   **Steps:**

    1.  **Initialization (already done):** At this point, each thread has already calculated its local maximum and written it to `cache`.  So, `cache[threadIdx.x]` holds the local maximum calculated by thread `threadIdx.x`.

    2.  **Reduction in Steps (Iterative Halving):** We'll perform reduction in multiple steps. In each step, we'll reduce the number of active threads and compare pairs of values in `cache`.

        *   **First Step (Stride = 1/2 the block size):**
            *   Threads with even thread IDs (0, 2, 4, ...) will compare their `cache` value with the `cache` value of the thread immediately next to them (1, 3, 5, ...).
            *   If the value in `cache[even_thread_id + stride]` is greater, the even thread will update its `cache[even_thread_id]` with the larger value.  Otherwise, it keeps its current value.
            *   **Example (block size 8):**
                *   Thread 0 compares `cache[0]` and `cache[4]` (stride = 4) and stores the max in `cache[0]`.
                *   Thread 2 compares `cache[2]` and `cache[6]` (stride = 4) and stores the max in `cache[2]`.
                *   Threads 1, 3, 5, 7 become inactive in this step.
            *   **`__syncthreads();`**  Very important!  We synchronize after each step to make sure all updates to `cache` are complete before moving to the next step.

        *   **Second Step (Stride = 1/4 the block size):**
            *   Now, threads with thread IDs that are multiples of 4 (0, 4, 8, ...) are active.
            *   They compare their `cache` value with the value `stride` positions away (which is now half of the previous stride).
            *   **Example (block size 8):**
                *   Thread 0 compares `cache[0]` and `cache[2]` (stride = 2) and stores the max in `cache[0]`.
                *   Thread 4 becomes inactive.
            *   **`__syncthreads();`**

        *   **Continue Halving the Stride:** We keep repeating this process, halving the `stride` in each step (stride becomes block size / 8, block size / 16, and so on) and reducing the number of active threads by half in each step.

        *   **Final Step (Stride = 1):**
            *   Eventually, the stride becomes 1. In the last step, thread 0 (and only thread 0) will compare `cache[0]` with `cache[1]` and store the final block maximum in `cache[0]`.
            *   **Example (block size 8, after several steps):**
                *   Thread 0 compares `cache[0]` and `cache[1]` (stride = 1) and stores the max in `cache[0]`.

    3.  **Result:** After all these reduction steps, `cache[0]` in each block will hold the **maximum value** found within the entire portion of the array assigned to that block!

*   **Why Shared Memory?**  This intra-block reduction is done in shared memory because shared memory is much faster than global memory.  We want to do as much computation as possible within the fast shared memory to maximize performance.

#### 2. Inter-Block Parallel Reduction (Final Stage)

*   **The Problem:** Now, each block has computed its own maximum and stored it (likely in `cache[0]`).  We have multiple block maximums, and we need to find the single, overall maximum of the entire array.

*   **Solution:** We use the `atomicMax` operation on a global output variable.

*   **Steps:**

    1.  **First Thread of Each Block Writes to Global Memory:**  The very first thread (thread 0) of each block is responsible for taking the maximum value it calculated for its block (which is now in `cache[0]`) and writing it to a global memory location (`output`).

    2.  **`atomicMax` for Safe Updates:**  We use `atomicMax(output, cache[0])` to ensure that updates to the `output` variable are done safely and correctly, even if multiple blocks try to update it at almost the same time.


##### How `atomicMax` Helps

*   `atomicMax` makes sure that only one block updates the output variable at a time, in a safe way.

*   Let's say:
    *   **Block 0** finishes and thread 0 from Block 0 tries to update the global `output` with its block's maximum (let's call it `block0_max`).  `atomicMax` lets this update happen.
    *   Now, **Block 1** finishes, and thread 0 from Block 1 tries to update `output` with `block1_max`. `atomicMax` *compares* `output` (which now holds `block0_max`) with `block1_max`.
        *   If `block1_max` is greater than `block0_max`, `atomicMax` updates `output` to `block1_max`.
        *   If `block1_max` is *not* greater, `output` stays as `block0_max`.

*   This process repeats for all blocks.  By the time all blocks are done, the `output` variable will hold the **absolute maximum value** found in the entire input array!

#### Key Learnings Explained

---

        extern __shared__ float cache[];

*   The keyword `**extern**` in front of `__shared__ float cache[];` is important because it tells CUDA: "this `cache` array is in shared memory, and its size isn't fixed at compile time. We're going to decide how big it is when we actually launch the kernel."

*   I think of `extern` as , "The size is determined *outside* of this kernel function."  It's determined when you launch the kernel using the `<<<gridSize, blockSize, sharedMemoryBytes>>>` syntax.

*   This gives you **flexibility**. You can make your shared memory array just the right size you need for each kernel launch, instead of being stuck with a size decided when you wrote the code.

*   For example, when you launch your kernel, you'll specify the shared memory size like this (the third parameter):

        `your_Kernel<<<gridSize, blockSize, sharedMemoryBytes>>>(arguments);`

---

        int index = blockDim.x * blockIdx.x + threadIdx.x;

*   This line calculates a **global index** for each thread.  It's computes the unique ID number for each thread in the entire grid.

*   `blockIdx.x` tells you which block the thread is in (in the x-dimension of the grid).
*   `blockDim.x` is the number of threads in each block (in the x-dimension).
*   `threadIdx.x` tells you which thread it is *within* its block.

*   By combining these, you get a number that represents the thread's position in the overall **grid of threads**.  This is used to figure out which part of the input array a thread should work on.

---

        stride = gridDim.x * blockDim.x;

*   The `**stride**` is a way to make each thread process **multiple elements** of the input array, but in a way that spreads the work evenly across all threads and blocks.

*   `stride = (blockDim.x * gridDim.x)`  is simply the total number of threads in your entire grid.

*   **How it works (Example):**

    Let's say your array has **20 elements**, you have **2 blocks**, and **2 threads per block**.
    *   `stride = (blockDim.x * gridDim.x) = (2 * 2) = 4`

    *   **Thread Responsibility:**

        *   **Block 0, Thread 0** is responsible for indices: `[0, 0+4, 0+8, 0+12, 0+16]` which is `[0, 4, 8, 12, 16]`
        *   **Block 0, Thread 1** is responsible for indices: `[1, 1+4, 1+8, 1+12, 1+16]` which is `[1, 5, 9, 13, 17]`
        *   **Block 1, Thread 0** is responsible for indices: `[2, 2+4, 2+8, 2+12, 2+16]` which is `[2, 6, 10, 14, 18]`
        *   **Block 1, Thread 1** is responsible for indices: `[3, 3+4, 3+8, 3+12, 3+16]` which is `[3, 7, 11, 15, 19]`

*   So, instead of each thread just processing a *contiguous* chunk, they process elements that are `stride` positions apart. This helps when the array is larger than the number of threads and you want to make sure all threads are kept busy and work on elements spread across the whole array.

---

        __syncthreads();

*   `**__syncthreads()**` is a super important synchronization command in CUDA. It acts like a **barrier** within a thread block.

*   When a thread reaches `__syncthreads()`, it **stops and waits** until *all* other threads in the *same block* have also reached `__syncthreads()`.

*   Once *every* thread in the block has arrived at `__syncthreads()`, *then* they can all proceed together to the code *after* `__syncthreads()`. 

*   **In our max reduction example:**  `__syncthreads()` is crucial to make sure that *all* threads in a block have finished calculating and storing their local maximums in the `cache` (shared memory) *before* any thread tries to read or use those values from the `cache` for the next steps of the reduction or writing to global memory. It ensures that the shared memory is in a consistent state before moving on.
