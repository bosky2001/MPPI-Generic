#pragma once

#ifndef DUBINS_QUADRATIC_COST_CUH_
#define DUBINS_QUADRATIC_COST_CUH_

#include <mppi/cost_functions/cost.cuh>
#include <mppi/dynamics/dubins/dubins.cuh>
#include <mppi/utils/file_utils.h>

struct DubinsQuadraticCostParams : public CostParams<2>
{
  float position_coeff = 1000;
  float yaw_coeff = 100;
  float terminal_cost_coeff = 2;

  float desired_terminal_state[3] = { 0, 0, M_PI};

  DubinsQuadraticCostParams()
  { 

    this->control_cost_coeff[0] = 0.0;
    this->control_cost_coeff[1] = 0.0;
  }
};

class DubinsQuadraticCost : public Cost<DubinsQuadraticCost, DubinsQuadraticCostParams, DubinsParams>
{
public:
  /**
   * Constructor
   * @param width
   * @param height
   */
  DubinsQuadraticCost(cudaStream_t stream = 0);

  /**
   * @brief Compute the state cost
   */
  __device__ float computeStateCost(float* s, int timestep = 0, float* theta_c = nullptr, int* crash_status = nullptr);

  /**
   * @brief Compute the state cost on the CPU
   */
  float computeStateCost(const Eigen::Ref<const output_array> s, int timestep = 0, int* crash_status = nullptr);

  /**
   * @brief Compute the terminal cost of the system
   */
  __device__ float terminalCost(float* s, float* theta_c);

  float terminalCost(const Eigen::Ref<const output_array> s);

protected:
};

#if __CUDACC__
#include "dubins_quadratic_cost.cu"
#endif

#endif  // DUBINS_QUADRATIC_COST_CUH_// Include the cart pole cost.
