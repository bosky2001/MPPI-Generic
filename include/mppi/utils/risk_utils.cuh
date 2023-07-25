#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
namespace mppi
{
struct risk_measure
{
  enum FUNC_TYPE : int
  {
    MAX = 0,
    MIN,
    VAR,
    CVAR,
    MEAN,
    MEDIAN,
    NUM_FUNCS
  };

  FUNC_TYPE func_ = CVAR;
  float alpha_ = 0.8;

  __host__ __device__ float shaping_func(float* __restrict__ costs, const int num_costs)
  {
    return shaping_func(costs, num_costs, func_, alpha_);
  }

  static __host__ __device__ float shaping_func(float* __restrict__ costs, const int num_costs,
                                                const FUNC_TYPE type = MEAN, const float risk_tolerance = 0.5f)
  {
    float cost = 0;
    if (num_costs == 1)
    {
      return costs[0];
    }
    switch (type)
    {
      case CVAR:
        cost = cvar(costs, num_costs, risk_tolerance);
        break;
      case MAX:
        cost = max_measure(costs, num_costs);
        break;
      case MEAN:
        cost = mean_measure(costs, num_costs);
        break;
      case MEDIAN:
        cost = var(costs, num_costs, 0.5);
        break;
      case MIN:
        cost = min_measure(costs, num_costs);
        break;
      case VAR:
        cost = var(costs, num_costs, risk_tolerance);
        break;
    }
    return cost;
  }

  static __host__ __device__ float max_measure(const float* __restrict__ costs, const int num_costs)
  {
    float max_cost = costs[0];
    for (int i = 1; i < num_costs; i++)
    {
      if (costs[i] > max_cost)
      {
        max_cost = costs[i];
      }
    }
    return max_cost;
  }

  static __host__ __device__ float min_measure(const float* __restrict__ costs, const int num_costs)
  {
    float min_cost = costs[0];
    for (int i = 1; i < num_costs; i++)
    {
      if (costs[i] < min_cost)
      {
        min_cost = costs[i];
      }
    }
    return min_cost;
  }

  static __host__ __device__ float mean_measure(const float* __restrict__ costs, const int num_costs)
  {
    float cost = 0;
    for (int i = 0; i < num_costs; i++)
    {
      cost += costs[i];
    }
    return cost / num_costs;
  }

  static __host__ __device__ float h_index(const int num_costs, const float alpha)
  {
    return alpha * (num_costs - 1);
  }

  static __host__ __device__ float var(float* __restrict__ costs, const int num_costs, float alpha)
  {
    float cost = 0;
#ifdef __CUDA_ARCH__
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(costs);
    thrust::sort(thrust::seq, thrust_ptr, thrust_ptr + num_costs);
#else
    std::sort(costs, costs + num_costs);
#endif
    float h_idx = h_index(num_costs, alpha);
    int next_idx = min((int)ceilf(h_idx), num_costs - 1);
    int prev_idx = max((int)floorf(h_idx), 0);
    cost = costs[prev_idx] + (h_idx - prev_idx) * (costs[next_idx] - costs[prev_idx]);
    return cost;
  }

  static __host__ __device__ float cvar(float* __restrict__ costs, const int num_costs, float alpha)
  {
    float cost = 0;
    float value_at_risk = var(costs, num_costs, alpha);  // also sorts costs
    int num_costs_above = 1;
    float sum_costs_above = value_at_risk;
    float h_idx = h_index(num_costs, alpha);
    for (int i = ceilf(h_idx); i < num_costs; i++)
    {
      num_costs_above++;
      sum_costs_above += costs[i];
    }
    cost = sum_costs_above / num_costs_above;
    return cost;
  }
};
}  // namespace mppi