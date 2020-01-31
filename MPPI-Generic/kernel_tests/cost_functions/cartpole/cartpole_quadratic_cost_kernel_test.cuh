#ifndef MPPIGENERIC_CARTPOLE_QUADRATIC_COST_KERNEL_TEST_CUH
#define MPPIGENERIC_CARTPOLE_QUADRATIC_COST_KERNEL_TEST_CUH

#include <cost_functions/cartpole/cartpole_quadratic_cost.cuh>

/**
 * In this cost function, we are supplied with a device side cost class, and a device side parameter structure
 */
__global__ void parameterTestKernel(CartpoleQuadraticCost* cost_d, CartpoleQuadraticCost::Params* params_d);

/**
 *
 */
void launchParameterTestKernel(const CartpoleQuadraticCost& cost, CartpoleQuadraticCost::Params& param_check);

#endif //!MPPIGENERIC_CARTPOLE_QUADRATIC_COST_KERNEL_TEST_CUH