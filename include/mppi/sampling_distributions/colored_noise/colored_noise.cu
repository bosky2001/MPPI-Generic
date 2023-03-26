/**
 * Created by Bogdan Vlahov on 3/25/2023
 **/

#define COLORED_TEMPLATE template <class CLASS_T, template <int> class PARAMS_TEMPLATE, class DYN_PARAMS_T>
#define COLORED_NOISE ColoredNoiseDistributionImpl<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>

#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>
#include <mppi/utils/cuda_math_utils.cuh>
#include <mppi/utils/math_utils.h>

COLORED_TEMPLATE
COLORED_NOISE::ColoredNoiseDistributionImpl(cudaStream_t stream) : PARENT_CLASS::GaussianDistributionImpl(stream)
{
}

COLORED_TEMPLATE
COLORED_NOISE::ColoredNoiseDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS::GaussianDistributionImpl(params, stream)
{
}

COLORED_TEMPLATE
__host__ void COLORED_NOISE::freeCudaMem()
{
  if (GPUMemStatus_)
  {
    cudaFree(freq_coeffs_d_);
    cudaFree(samples_in_freq_complex_d_);
    cudaFree(noise_in_time_d_);
    cudaFree(frequency_sigma_d_);
    freq_coeffs_d_ = nullptr;
    frequency_sigma_d_ = nullptr;
    noise_in_time_d_ = nullptr;
    samples_in_freq_complex_d_ = nullptr;
    cufftDestroy(plan_);
  }
  PARENT_CLASS::freeCudaMem();
}

COLORED_TEMPLATE
__host__ void COLORED_NOISE::allocateCUDAMemoryHelper()
{
  if (GPUMemStatus_)
  {
    if (frequency_sigma_d_)
    {
      HANDLE_ERROR(cudaFreeAsync(frequency_sigma_d_, this->stream_));
    }
    if (samples_in_freq_complex_d_)
    {
      HANDLE_ERROR(cudaFreeAsync(samples_in_freq_complex_d_, this->stream_));
    }
    if (noise_in_time_d_)
    {
      HANDLE_ERROR(cudaFreeAsync(noise_in_time_d_, this->stream_));
    }
    if (freq_coeffs_d_)
    {
      HANDLE_ERROR(cudaFreeAsync(freq_coeffs_d_, this->stream_));
    }
    const int freq_size = this->getNumTimesteps() / 2 + 1;
    HANDLE_ERROR(cudaMallocAsync((void**)&freq_coeffs_d_, sizeof(float) * freq_size * CONTROL_DIM, this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&frequency_sigma_d_, sizeof(float) * CONTROL_DIM, this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&samples_in_freq_complex_d_,
                                 sizeof(cufftComplex) * this->getNumRollouts() * CONTROL_DIM * freq_size *
                                     this->getNumDistributions(),
                                 this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&noise_in_time_d,
                                 sizeof(float) * this->getNumRollouts() * CONTROL_DIM * this->getNumTimesteps() *
                                     this->getNumDistributions(),
                                 this->stream_));
    // Recreate FFT Plan
    HANDLE_CUFFT_ERROR(
        cufftPlan1d(&plan_, this->getNumTimesteps(), CUFFT_C2R, this->getNumRollouts() * this->getNumDistributions()));
    HANDLE_CUFFT_ERROR(cufftSetStream(plan_, this->stream_));
  }
  PARENT_CLASS::allocateCUDAMemoryHelper();
}

// COLORED_TEMPLATE
// __device__ void COLORED_NOISE::getControlSample(const int sample_index, const int t, const int distribution_index,
//                                                 const float* state, float* control, float* theta_d,
//                                                 const int block_size, const int thread_index)
// {
//   SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
//   const int distribution_i = distribution_index >= params_p->num_distributions ? 0 : distribution_index;
//   const int control_index =
//       ((params_p->num_rollouts * distribution_i + sample_idx) * params_p->num_timesteps + t) * CONTROL_DIM;
//   const int mean_index = (params_p->num_timesteps * distribution_i + t) * CONTROL_DIM;
//   if (CONTROL_DIM % 4 == 0)
//   {
//     float4* du4 = reinterpret_cast<float4*>(
//         &theta_d[sizeof(SAMPLING_PARAMS_T) / sizeof(float) + CONTROL_DIM * threadIdx.x]);  // TODO: replace with
//         theta_d
//     float4* u4 = reinterpret_cast<float4*>(control);
//     const float4* u4_mean_d = reinterpret_cast<const float4*>(&(this->control_mean_d_[mean_index]));
//     const float4* u4_d = reinterpret_cast<const float4*>(&(this->control_samples_d_[control_index]));
//     for (int i = thread_idx; i < CONTROL_DIM / 4; i += block_size)
//     {
//       u4[j] = u4_d[j];
//       du4[j] = u4[j] - u4_mean_d[j];
//     }
//   }
//   else if (CONTROL_DIM % 2 == 0)
//   {
//     float2* du2 = reinterpret_cast<float2*>(
//         &theta_d[sizeof(SAMPLING_PARAMS_T) / sizeof(float) + CONTROL_DIM * threadIdx.x]);  // TODO: replace with
//         theta_d
//     float2* u2 = reinterpret_cast<float2*>(control);
//     const float2* u2_mean_d = reinterpret_cast<const float2*>(&(this->control_mean_d_[mean_index]));
//     const float2* u2_d = reinterpret_cast<const float2*>(&(this->control_samples_d_[control_index]));
//     for (int i = thread_idx; i < CONTROL_DIM / 2; i += block_size)
//     {
//       u2[j] = u2_d[j];
//       du2[j] = u2[j] - u2_mean_d[j];
//     }
//   }
//   else
//   {
//     float* du = reinterpret_cast<float*>(
//         &theta_d[sizeof(SAMPLING_PARAMS_T) / sizeof(float) + CONTROL_DIM * threadIdx.x]);  // TODO: replace with
//         theta_d
//     float* u = reinterpret_cast<float*>(control);
//     const float* u_mean_d = reinterpret_cast<const float*>(&(this->control_mean_d_[mean_index]));
//     const float* u_d = reinterpret_cast<const float*>(&(this->control_samples_d_[control_index]));
//     for (int i = thread_idx; i < CONTROL_DIM; i += block_size)
//     {
//       u[j] = u_d[j];
//       du[j] = u[j] - u_mean_d[j];
//     }
//   }
// }

COLORED_TEMPLATE
__host__ void COLORED_NOISE::generateSamples(const int& optimization_stride, const int& iteration_num,
                                             curandGenerator_t& gen)
{
  const int BLOCKSIZE_X = this->params_.readControlsBlockDim.x;
  const int BLOCKSIZE_Y = this->params_.readControlsBlockDim.y;
  const int BLOCKSIZE_Z = this->params_.readControlsBlockDim.z;
  const int num_trajectories = this->getNumRollouts() * this->getNumDistributions();

  std::vector<float> sample_freq;
  fftfreq(this->getNumTimesteps(), sample_freq);
  const float cutoff_freq = fmaxf(this->params_.fmin, 1.0 / this->getNumTimesteps());
  const int freq_size = sample_freq.size();

  int smaller_index = 0;
  Eigen::MatrixXf sample_freqs(freq_size, CONTROL_DIM);

  // Adjust the weighting of each frequency by the exponents
  for (int i = 0; i < freq_size; i++)
  {
    if (sample_freq[i] < cutoff_freq)
    {
      smaller_index++;
    }
    else if (smaller_index < freq_size)
    {
      for (int j = 0; j < smaller_index; j++)
      {
        sample_freq[j] = sample_freq[smaller_index];
        for (int k = 0; k < CONTROL_DIM; k++)
        {
          sample_freqs(j, k) = powf(sample_freq[smaller_index], -exponents[k] / 2.0);
        }
      }
    }
    for (int j = 0; j < CONTROL_DIM; j++)
    {
      sample_freqs(i, j) = powf(sample_freq[i], -exponents[j] / 2.0);
    }
  }

  // Calculate variance
  float sigma[CONTROL_DIM] = { 0 };
  for (int i = 0; i < CONTROL_DIM; i++)
  {
    for (int j = 1; j < freq_size - 1; j++)
    {
      sigma[i] += powf(sample_freqs(j, i), 2);
    }
    sigma[i] += powf(sample_freqs(freq_size - 1, i) * ((1.0 + (this->getNumTimesteps() % 2)) / 2.0), 2);
    sigma[i] = 2 * sqrt(sigma[i]) / this->getNumTimesteps();
  }

  // Sample the noise in frequency domain and reutrn to time domain
  const int batch = num_trajectories * CONTROL_DIM;
  // Need 2 * (this->getNumTimesteps() / 2 + 1) * batch of randomly sampled values
  // float* samples_in_freq_d;
  HANDLE_CURAND_ERROR(curandGenerateNormal(gen, (float*)samples_in_freq_complex_d, 2 * batch * freq_size, 0.0, 1.0));
  HANDLE_ERROR(cudaMemcpyAsync(freq_coeffs_d_, sample_freqs.data(), sizeof(float) * freq_size * CONTROL_DIM,
                               cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(
      cudaMemcpyAsync(frequency_sigma_d_, sigma, sizeof(float) * CONTROL_DIM, cudaMemcpyHostToDevice, this->stream_));
  const int num_trajectories_grid_x = mppi::math::int_ceil(num_trajectories, BLOCKSIZE_X);
  const int variance_grid_y = (freq_size - 1) / BLOCKSIZE_Y + 1;
  const int control_grid_z = mppi::math::int_ceil(CONTROL_DIM, BLOCKSIZE_Z);
  dim3 grid(num_trajectories_grid_x, variance_grid_y, control_grid_z);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  configureFrequencyNoise<<<grid, block, 0, this->stream_>>>(samples_in_freq_complex_d_, freq_coeffs_d_,
                                                             num_trajectories, CONTROL_DIM, freq_size);
  HANDLE_ERROR(cudaGetLastError());
  // freq_data needs to be batch number of num_timesteps/2 + 1 cuComplex values
  // time_data needs to be batch * num_timesteps floats
  HANDLE_CUFFT_ERROR(cufftExecC2R(plan_, samples_in_freq_complex_d_, noise_in_time_d_));

  // Change axes ordering from [trajectories, control, time] to [trajectories, time, control]
  const int reorder_grid_y = mppi::math::int_ceil(this->getNumTimesteps(), BLOCKSIZE_Y);
  dim3 reorder_grid(num_trajectories_grid_x, reorder_grid_y, control_grid_z);
  rearrangeNoise<<<reorder_grid, block, 0, this->stream_>>>(noise_in_time_d, control_noise_d_, frequency_sigma_d_,
                                                            num_trajectories, this->getNumTimesteps(), CONTROL_DIM,
                                                            optimization_stride);

  // Rewrite pure noise into actual control samples
  dim3 control_writing_grid;
  control_writing_grid.x = mppi::math::int_ceil(this->getNumRollouts(), BLOCKSIZE_X);
  control_writing_grid.y = mppi::math::int_ceil(this->getNumTimesteps(), BLOCKSIZE_Y);
  control_writing_grid.z = mppi::math::int_ceil(this->getNumDistributions(), BLOCKSIZE_Z);
  unsigned int std_dev_mem_size = this->getNumDistributions() * CONTROL_DIM;
  // Allocate shared memory for std_deviations per timestep or constant across the trajectory
  std_dev_mem_size = mppi::math::nearest_quotient_4(
      this->params_.time_specific_std_dev ? std_dev_mem_size * this->getNumTimesteps() : std_dev_mem_size);
  unsigned int shared_mem_size =
      std_dev_mem_size +
      mppi::math::nearest_quotient_4(this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM) +
      mppi::math::nearest_quotient_4(BLOCKSIZE_X * BLOCKSIZE_Y * BLOCKSIZE_Z * CONTROL_DIM);
  setGaussianControls<<<control_writing_grid, this->params_.readControlsBlockDim, shared_mem_size, this->stream_>>>(
      this->control_mean_d_, this->std_dev_d_, this->control_samples_d_, CONTROL_DIM, this->getNumTimesteps(),
      this->getNumRollouts(), this->getNumDistributions(), optimization_stride,
      this->params_.pure_noise_trajectories_percentage, this->params_.time_specific_std_dev);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}
#undef COLORED_TEMPLATE
#undef COLORED_NOISE