#pragma once

#ifndef KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_
#define KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_

#include <mppi_core/mppi_common.cuh>
#include <curand.h>
#include <vector>

// Declare some sizes for the kernel parameters
const int STATE_DIM = 12;
const int CONTROL_DIM = 3;
const int NUM_ROLLOUTS = 100; // .99 times this number has to be an integer... TODO fix how brittle this is
const int BLOCKSIZE_X = 64;
const int BLOCKSIZE_Y = 8; // Blocksize_y has to be greater than the control dim TODO fix how we step through the controls

__global__ void loadGlobalToShared_KernelTest(float* x0_device, float* sigma_u_device,
                                              float* x_thread, float* xdot_thread, float* u_thread, float* du_thread, float* sigma_u_thread);

void launchGlobalToShared_KernelTest(const std::vector<float>& x0_host,const std::vector<float>& u_var_host,
                                     std::vector<float>& x_thread_host, std::vector<float>& xdot_thread_host,
                                     std::vector<float>& u_thread_host, std::vector<float>& du_thread_host, std::vector<float>& sigma_u_thread_host );

__global__ void injectControlNoiseOnce_KernelTest(int num_rollouts, int num_timesteps, int timestep, float* u_traj_device,
                                                  float* ep_v_device, float* sigma_u_device, float* control_compute_device);

void launchInjectControlNoiseOnce_KernelTest(const std::vector<float>& u_traj_host, const int num_rollouts, const int num_timesteps,
                                             std::vector<float>& ep_v_host, std::vector<float>& sigma_u_host, std::vector<float>& control_compute);

__global__ void injectControlNoise_KernelTest();

void launchInjectControlNoise_KernelTest();

#endif // !KERNEL_TESTS_MPPI_CORE_ROLLOUT_KERNEL_TEST_CUH_