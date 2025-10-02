# CudaMatrixMult

Custom Matmul experiment to attempt to reach close to SOTA to learn CUDA, following:
https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
https://siboehm.com/articles/22/CUDA-MMM

Project Setup:
Visual Studio cuda project setup
    Likely would rather change to a makefile approach if I started again
    as reviewing this project without has every file in one folder :/
Note that a column major approach was taken similar to cublas

MatrixMultiplication.cu:
    Launching point of app, contains main

kernels.cuh:
    Strategy / Registry of the various kernels to launch
    Rigid func defs, but chosen mostly to easily change what to compare

GpuMatrix.h:
    RAII class, abstraction over matrices on device with various functions to change them

/matmuls
    contains various matmuls over various iterations

/policies
    contains template policies to abstract out data types and operations for modularity and reuse
    e.i. GemmPolicy
    Decided to go for this approach after reviewing Cutlass docs, wanting to develop more akin to 
    Will extend later to potentially test on various datatypes and etc

Various Specs:

Compute Capability: 8.6

Optimizing for:
Name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Compute Capability: 8.6
Max threads per block: 1024
Max threads per multiprocessor: 1536
Threads per warp: 32
Registers per block: 65536
Registers per multiprocessor: 65536
Global memory (MB): 6143
Shared mem per block: 48 KB
Shared mem per SM: 102400 B
SM count: 20
Max warps per SM: 48

