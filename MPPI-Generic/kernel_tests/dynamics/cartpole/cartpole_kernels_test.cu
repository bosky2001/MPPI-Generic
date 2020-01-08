//
// Created by mgandhi3 on 1/7/20.
//
#include <dynamics/cartpole/cartpole.cuh>

__global__ void ParameterTestKernel(Cartpole* CP) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    printf("\nEntering the kernel!\n");
    printf("The thread id is: %i\n", tid);
    if (tid == 0) {
        printf("This is gravity: %f\n", CP->getGravity());
        printf("This is the mass of the cart: %f\n", CP->getCartMass());
        printf("This is the mass of the pole: %f\n", CP->getPoleMass());
        printf("This is the length of the pole: %f\n", CP->getPoleLength());
        printf("This is the value of GPUMemstatus on the GPU: %d\n", CP->GPUMemStatus_);
        printf("This is the value of CP_device on the GPU: %d\n", CP->CP_device);
    }
}

void launchParameterTestKernel(const Cartpole& CP) {
    ParameterTestKernel<<<1,1>>>(CP.CP_device);
    CudaCheckError();
}