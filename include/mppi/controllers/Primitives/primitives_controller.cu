#include <mppi/controllers/Primitives/primitives_controller.cuh>
#include <mppi/core/mppi_common.cuh>
#include <algorithm>
#include <iostream>
#include <mppi/sampling_distributions/piecewise_linear/piecewise_linear_noise.cuh>
#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>

#define Primitives PrimitivesController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
Primitives::PrimitivesController(DYN_T* model, COST_T* cost, FB_T* fb_controller, float dt, int max_iter, float lambda,
                                 float alpha, const Eigen::Ref<const control_array>& control_std_dev, int num_timesteps,
                                 const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
        model, cost, fb_controller, dt, max_iter, lambda, alpha, control_std_dev, num_timesteps, init_control_traj,
        stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();
  std::vector<float> tmp_vec(DYN_T::CONTROL_DIM, 0.0);
  colored_noise_exponents_ = std::move(tmp_vec);
  scale_piecewise_noise_ = std::move(tmp_vec);

  // Copy the noise std_dev to the device
  this->copyControlStdDevToDevice();
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
Primitives::~PrimitivesController()
{
  // all implemented in standard controller
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  // this->free_energy_statistics_.real_sys.previousBaseline = this->baseline_;
  state_array local_state = state;
  for (int i = 0; i < DYN_T::STATE_DIM; i++)
  {
    float diff = fabsf(this->state_.col(leash_jump_)[i] - state[i]);
    if (state_leash_dist_[i] < diff)
    {
      local_state[i] = state[i];
    }
    else
    {
      local_state[i] = this->state_.col(leash_jump_)[i];
    }
  }

  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, local_state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));

  /////////////////
  // BEGIN INTERMEDIATE PLANNER
  // Compute intermediate plan using piecewise linear noise, and choosing the best

  int prev_controls_idx = 1;
  float primitives_baseline = 0.0;
  float baseline_prev = 0.0;

  // Send the nominal control to the device
  this->copyNominalControlToDevice();

  for (int opt_iter = 0; opt_iter < num_primitive_iters_; opt_iter++)
  {
    powerlaw_psd_gaussian(colored_noise_exponents_, this->num_timesteps_, NUM_ROLLOUTS, this->control_noise_d_,
                          this->gen_, this->stream_);

    // Generate piecewise linear noise data, update control_noise_d_
    piecewise_linear_noise(this->num_timesteps_, NUM_ROLLOUTS, DYN_T::CONTROL_DIM, num_piecewise_segments_,
                           optimization_stride, scale_piecewise_noise_, frac_add_nominal_traj_,
                           scale_add_nominal_noise_, this->control_d_, this->control_noise_d_, this->control_std_dev_d_,
                           this->gen_, this->stream_);

    // Set nominal controls to zero because we want to use the noise directly
    this->control_ = control_trajectory::Zero();

    // Send the zero nominal control to the device
    this->copyNominalControlToDevice();

    // Launch the rollout kernel
    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
        this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_, /*optimization_stride = */ 0,
        this->lambda_, this->alpha_, this->initial_state_d_, this->control_d_, this->control_noise_d_,
        this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_);

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    primitives_baseline = mppi_common::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS);

    // get previous control cost (at index = 1, since index = 0 is zero control traj)
    baseline_prev = this->trajectory_costs_.data()[prev_controls_idx];
    if (this->debug_)
    {
      std::cerr << "Previous Baseline: " << baseline_prev << "         Baseline: " << this->baseline_ << std::endl;
    }

    // if baseline is too high and trajectory is unsafe, create and issue a stopping trajectory
    if (stopping_cost_threshold_ > 0 && primitives_baseline > stopping_cost_threshold_)
    {
      std::cerr << "Baseline is too high, issuing stopping trajectory!" << std::endl;
      computeStoppingTrajectory(local_state);
      primitives_baseline = std::numeric_limits<float>::min();
    }
    else if (primitives_baseline > baseline_prev - hysteresis_cost_threshold_)
    {
      // baseline is not decreasing enough, use controls from the previous iteration
      if (this->debug_)
      {
        std::cerr << "Not enough improvement, use prev controls." << std::endl;
      }
      HANDLE_ERROR(cudaMemcpyAsync(
          this->control_.data(), this->control_noise_d_ + prev_controls_idx * this->num_timesteps_ * DYN_T::CONTROL_DIM,
          sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToHost, this->stream_));

      primitives_baseline = baseline_prev;
    }
    else
    {  // otherwise, update the nominal control
      // Copy best control from device to the host
      int best_idx = mppi_common::computeBestIndex(this->trajectory_costs_.data(), NUM_ROLLOUTS);
      HANDLE_ERROR(cudaMemcpyAsync(
          this->control_.data(), this->control_noise_d_ + best_idx * this->num_timesteps_ * DYN_T::CONTROL_DIM,
          sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToHost, this->stream_));
    }

    this->copyNominalControlToDevice();

    cudaStreamSynchronize(this->stream_);
  }

  // Copy back sampled trajectories for visualization
  if (visualize_primitives_)
  {
    this->copySampledControlFromDevice();
  }

  //  END INTERMEDIATE PLANNER
  ////////////////

  ////////////////
  // BEGIN MPPI
  for (int opt_iter = 0; opt_iter < this->num_iters_; opt_iter++)
  {
    // Send the nominal control to the device
    copyMPPIControlToDevice();

    // Generate noise data
    powerlaw_psd_gaussian(colored_noise_exponents_, this->num_timesteps_, NUM_ROLLOUTS, this->control_noise_d_,
                          this->gen_, this->stream_);
    // curandGenerateNormal(this->gen_, this->control_noise_d_, NUM_ROLLOUTS * this->num_timesteps_ *
    // DYN_T::CONTROL_DIM,
    //                      0.0, 1.0);
    /*
    std::vector<float> noise = this->getSampledNoise();
    float mean = 0;
    for(int k = 0; k < noise.size(); k++) {
      mean += (noise[k]/noise.size());
    }

    float std_dev = 0;
    for(int k = 0; k < noise.size(); k++) {
      std_dev += powf(noise[k] - mean, 2);
    }
    std_dev = sqrt(std_dev/noise.size());
    printf("CPU 1 side N(%f, %f)\n", mean, std_dev);
     */

    // Launch the rollout kernel
    mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(
        this->model_->model_d_, this->cost_->cost_d_, this->dt_, this->num_timesteps_, optimization_stride,
        this->lambda_, this->alpha_, this->initial_state_d_, control_mppi_d_, this->control_noise_d_,
        this->control_std_dev_d_, this->trajectory_costs_d_, this->stream_);
    /*
    noise = this->getSampledNoise();
    mean = 0;
    for(int k = 0; k < noise.size(); k++) {
      mean += (noise[k]/noise.size());
    }

    std_dev = 0;
    for(int k = 0; k < noise.size(); k++) {
      std_dev += powf(noise[k] - mean, 2);
    }
    std_dev = sqrt(std_dev/noise.size());
    printf("CPU 2 side N(%f, %f)\n", mean, std_dev);
     */

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    this->baseline_ = mppi_common::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS);

    // if (this->baseline_ > baseline_prev + 1)
    // {
    //   // TODO handle printing
    //   if (this->debug_)
    //   {
    //     std::cout << "Previous Baseline: " << baseline_prev << std::endl;
    //     std::cout << "         Baseline: " << this->baseline_ << std::endl;
    //   }
    // }

    // baseline_prev = this->baseline_;

    // Launch the norm exponential kernel
    mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X, this->trajectory_costs_d_, 1.0 / this->lambda_,
                                     this->baseline_, this->stream_);
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->normalizer_ = mppi_common::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS);

    mppi_common::computeFreeEnergy(this->free_energy_statistics_.real_sys.freeEnergyMean,
                                   this->free_energy_statistics_.real_sys.freeEnergyVariance,
                                   this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance,
                                   this->trajectory_costs_.data(), NUM_ROLLOUTS, this->baseline_, this->lambda_);

    // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
    mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(
        this->trajectory_costs_d_, this->control_noise_d_, control_mppi_d_, this->normalizer_, this->num_timesteps_,
        this->stream_);

    /*
    noise = this->getSampledNoise();
    mean = 0;
    for(int k = 0; k < noise.size(); k++) {
      mean += (noise[k]/noise.size());
    }

    std_dev = 0;
    for(int k = 0; k < noise.size(); k++) {
      std_dev += powf(noise[k] - mean, 2);
    }
    std_dev = sqrt(std_dev/noise.size());
    printf("CPU 3 side N(%f, %f)\n", mean, std_dev);
     */

    // Transfer the new control to the host
    HANDLE_ERROR(cudaMemcpyAsync(control_mppi_.data(), control_mppi_d_,
                                 sizeof(float) * this->num_timesteps_ * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToHost,
                                 this->stream_));
    cudaStreamSynchronize(this->stream_);
  }

  this->free_energy_statistics_.real_sys.normalizerPercent = this->normalizer_ / NUM_ROLLOUTS;
  this->free_energy_statistics_.real_sys.increase =
      this->baseline_ - this->free_energy_statistics_.real_sys.previousBaseline;

  // END MPPI
  ////////////////////////

  // decide between using the MPPI control or the primitives control
  if (this->debug_)
  {
    std::cerr << "mppi baseline: " << this->baseline_ << ", primitives baseline: " << primitives_baseline
              << ", prev baseline: " << baseline_prev << std::endl;
  }
  if ((num_primitive_iters_ == 0 && this->num_iters_ > 0) ||
      ((num_primitive_iters_ > 0 && this->num_iters_ > 0) &&
       (this->baseline_ < primitives_baseline - hysteresis_cost_threshold_)))
  {
    this->control_ = control_mppi_;
    this->copyNominalControlToDevice();
    if (this->debug_)
    {
      std::cerr << "Using MPPI control" << std::endl;
    }
  }

  // smoothControlTrajectory();
  computeStateTrajectory(local_state);
  state_array zero_state = state_array::Zero();
  for (int i = 0; i < this->num_timesteps_; i++)
  {
    this->model_->enforceConstraints(zero_state, this->control_.col(i));
    this->model_->enforceConstraints(zero_state, control_mppi_.col(i));
    // this->control_.col(i)[1] =
    //     fminf(fmaxf(this->control_.col(i)[1], this->model_->control_rngs_[1].x), this->model_->control_rngs_[1].y);
  }

  // Copy back sampled trajectories for visualization
  if (!visualize_primitives_)
  {
    this->copySampledControlFromDevice();
  }
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::allocateCUDAMemory()
{
  Controller<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>::allocateCUDAMemoryHelper();
  HANDLE_ERROR(cudaMalloc((void**)&control_mppi_d_, sizeof(float) * DYN_T::CONTROL_DIM * MAX_TIMESTEPS));
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::copyMPPIControlToDevice()
{
  HANDLE_ERROR(cudaMemcpyAsync(control_mppi_d_, control_mppi_.data(), sizeof(float) * control_mppi_.size(),
                               cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::computeStateTrajectory(const Eigen::Ref<const state_array>& x0)
{
  this->computeStateTrajectoryHelper(this->state_, x0, this->control_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::computeStoppingTrajectory(const Eigen::Ref<const state_array>& x0)
{
  state_array xdot;
  state_array state = x0;
  control_array u_i = control_array::Zero();
  this->model_->initializeDynamics(state, u_i, 0, this->dt_);
  for (int i = 0; i < this->num_timesteps_ - 1; ++i)
  {
    this->model_->getStoppingControl(state, u_i);
    this->model_->enforceConstraints(state, u_i);
    this->control_.col(i) = u_i;
    this->model_->computeStateDeriv(state, u_i, xdot);
    this->model_->updateState(state, xdot, this->dt_);
  }
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::slideControlSequence(int steps)
{
  // TODO does the logic of handling control history reasonable?
  leash_jump_ = steps;
  // Save the control history
  this->saveControlHistoryHelper(steps, this->control_, this->control_history_);

  this->slideControlSequenceHelper(steps, this->control_);
  this->slideControlSequenceHelper(steps, control_mppi_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
}

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void Primitives::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  // controls already copied in compute control

  mppi_common::launchStateAndCostTrajectoryKernel<DYN_T, COST_T, FEEDBACK_GPU, BDIM_X, BDIM_Y>(
      this->model_->model_d_, this->cost_->cost_d_, this->fb_controller_->getDevicePointer(), this->sampled_noise_d_,
      this->initial_state_d_, this->sampled_states_d_, this->sampled_costs_d_, this->sampled_crash_status_d_,
      num_sampled_trajectories, this->num_timesteps_, this->dt_, this->vis_stream_);

  for (int i = 0; i < num_sampled_trajectories; i++)
  {
    // set initial state to the first location
    this->sampled_trajectories_[i].col(0) = this->state_.col(0);
    // shifted by one since we do not save the initial state
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_trajectories_[i].data() + (DYN_T::STATE_DIM),
                                 this->sampled_states_d_ + i * this->num_timesteps_ * DYN_T::STATE_DIM,
                                 (this->num_timesteps_ - 1) * DYN_T::STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost,
                                 this->vis_stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(this->sampled_costs_[i].data(), this->sampled_costs_d_ + (i * (this->num_timesteps_ + 1)),
                        (this->num_timesteps_ + 1) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_crash_status_[i].data(),
                                 this->sampled_crash_status_d_ + (i * this->num_timesteps_),
                                 this->num_timesteps_ * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}

#undef Primitives
