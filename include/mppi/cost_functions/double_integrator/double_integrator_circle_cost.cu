#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>

DoubleIntegratorCircleCost::DoubleIntegratorCircleCost(cudaStream_t stream) {
  bindToStream(stream);
}

__device__ float DoubleIntegratorCircleCost::computeStateCost(float *s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  if ((radial_position < params_.inner_path_radius2) ||
      (radial_position > params_.outer_path_radius2)) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost * (current_velocity - params_.velocity_desired) *
          (current_velocity - params_.velocity_desired);
  cost += params_.velocity_cost *
          (current_angular_momentum - params_.angular_momentum_desired) *
          (current_angular_momentum - params_.angular_momentum_desired);
  return cost;
}

float DoubleIntegratorCircleCost::computeStateCost(const Eigen::Ref<const state_array> s) {
  float radial_position = s[0]*s[0] + s[1]*s[1];
  float current_velocity = sqrtf(s[2]*s[2] + s[3]*s[3]);
  float current_angular_momentum = s[0]*s[3] - s[1]*s[2];

  float cost = 0;
  if ((radial_position < params_.inner_path_radius2) ||
      (radial_position > params_.outer_path_radius2)) {
    cost += params_.crash_cost;
  }
  cost += params_.velocity_cost * (current_velocity - params_.velocity_desired) *
          (current_velocity - params_.velocity_desired);
  cost += params_.velocity_cost *
          (current_angular_momentum - params_.angular_momentum_desired) *
          (current_angular_momentum - params_.angular_momentum_desired);
  return cost;
}

float DoubleIntegratorCircleCost::computeRunningCost(const Eigen::Ref<const state_array> s,
                                                     const Eigen::Ref<const control_array> u,
                                                     const Eigen::Ref<const control_array> noise,
                                                     const Eigen::Ref<const control_array> std_dev,
                                                     float lambda, float alpha, int timestep) {
  return computeStateCost(s) +
          this->computeLikelihoodRatioCost(u, noise, std_dev, lambda, alpha);
}

__device__ float DoubleIntegratorCircleCost::computeRunningCost(float *s, float *u, float* noise, float* std_dev, float lambda, float alpha, int timestep) {
  return computeStateCost(s) +
         this->computeLikelihoodRatioCost(u, noise, std_dev, lambda, alpha);
}

float DoubleIntegratorCircleCost::terminalCost(const Eigen::Ref<const state_array> s) {
  return 0;
}

__device__ float DoubleIntegratorCircleCost::terminalCost(float *state) {
  return 0;
}