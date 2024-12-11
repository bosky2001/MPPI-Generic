#include <mppi/cost_functions/dubins/dubins_quadratic_cost.cuh>

DubinsQuadraticCost::DubinsQuadraticCost(cudaStream_t stream)
{
  bindToStream(stream);
}

float DubinsQuadraticCost::computeStateCost(const Eigen::Ref<const output_array> s, int timestep, int* crash_status)
{
  return (s[0] - params_.desired_terminal_state[0]) * (s[0] - params_.desired_terminal_state[0]) *
             params_.position_coeff +
         (s[1] - params_.desired_terminal_state[1]) * (s[1] - params_.desired_terminal_state[1]) *
             params_.position_coeff +
         (s[2] - params_.desired_terminal_state[2]) * (s[2] - params_.desired_terminal_state[2]) *
             params_.yaw_coeff; 
}

__device__ float DubinsQuadraticCost::computeStateCost(float* state, int timestep, float* theta_c, int* crash_status)
{
  return (state[0] - params_.desired_terminal_state[0]) * (state[0] - params_.desired_terminal_state[0]) *
             params_.position_coeff +
         (state[1] - params_.desired_terminal_state[1]) * (state[1] - params_.desired_terminal_state[1]) *
             params_.position_coeff +
         (state[2] - params_.desired_terminal_state[2]) * (state[2] - params_.desired_terminal_state[2]) *
             params_.yaw_coeff; 
}

__device__ float DubinsQuadraticCost::terminalCost(float* state, float* theta_c)
{
  return ((state[0] - params_.desired_terminal_state[0]) * (state[0] - params_.desired_terminal_state[0]) *
             params_.position_coeff +
         (state[1] - params_.desired_terminal_state[1]) * (state[1] - params_.desired_terminal_state[1]) *
             params_.position_coeff +
         (state[2] - params_.desired_terminal_state[2]) * (state[2] - params_.desired_terminal_state[2]) *
             params_.yaw_coeff) *
         params_.terminal_cost_coeff;
}
float DubinsQuadraticCost::terminalCost(const Eigen::Ref<const output_array> state)
{
  return ((state[0] - params_.desired_terminal_state[0]) * (state[0] - params_.desired_terminal_state[0]) *
             params_.position_coeff +
         (state[1] - params_.desired_terminal_state[1]) * (state[1] - params_.desired_terminal_state[1]) *
             params_.position_coeff +
         (state[2] - params_.desired_terminal_state[2]) * (state[2] - params_.desired_terminal_state[2]) *
             params_.yaw_coeff) *
         params_.terminal_cost_coeff;
}
