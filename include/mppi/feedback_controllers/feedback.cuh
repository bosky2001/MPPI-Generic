/*
 * Created on Sun Sep 6 2020 by Bogdan
 */

#ifndef FEEDBACK_BASE_CUH_
#define FEEDBACK_BASE_CUH_

#include <Eigen/Core>
#include <mppi/utils/managed.cuh>

#include <cmath>
#include <memory>
#include <tuple>

template<class GPU_FB_T, class TEMPLATED_DYNAMICS>
class GPUFeedbackController : public Managed {
public:
  /**
   * Type Aliasing
   */
  using DYN_T = TEMPLATED_DYNAMICS;

  GPU_FB_T* feedback_d_ = nullptr;

  explicit GPUFeedbackController(cudaStream_t stream) : Managed(stream) {}

  /**
   * =================== METHODS THAT SHOULD NOT BE OVERWRITTEN ================
   */
  virtual ~GPUFeedbackController() {
    freeCudaMem();
  };

  // Overwrite of Managed->GPUSetup to call allocateCUDAMemory as well
  void GPUSetup();
  void freeCudaMem();

  /**
   * ========================== METHODS TO OVERWRITE ===========================
   */
  // Method to allocate more CUDA memory if needed
  /**
   * Only need to allocate/deallocate additional memory, GPU pointer is already handled.
   */
  void allocateCUDAMemory() {}
  void deallocateCUDAMemory() {}

  __device__ void k(const float * x_act, const float * x_goal,
         const float t, float * theta,
         float* control_output) {
    //CLASS_T* derived = static_cast<CLASS_T*>(this);
    //derived->k(x_act, x_goal, t, theta, control_output);
  }

  // Abstract method to copy information to GPU
  void copyToDevice() {}
  // Method to return potential diagnostic information from GPU
  void copyFromDevice() {}
};

/**
 * Steps to making a new one
 * Create the GPUFeedback class as an impl class like costs but is still templated on DYN
 * The actual GPUFeedback_act class will then be templated on DYN and inherit from the GPUFeedbackImpl
 * Write the feedback controller to use the GPUFeedback_act as thee GPU_FEEDBACK_T template option
 * It will then automatically create the right pointer
 */
template<class GPU_FB_T, class PARAMS_T, int NUM_TIMESTEPS>
class FeedbackController {
public:
  typedef typename GPU_FB_T::DYN_T DYN_T;

  // Type Defintions and aliases
  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM,
                        NUM_TIMESTEPS> control_trajectory; // A control trajectory
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM,
                        NUM_TIMESTEPS> state_trajectory; // A state trajectory

  // Constructors and Generators
  FeedbackController(cudaStream_t stream=0) {
    gpu_controller_ = std::make_shared<GPU_FB_T>(stream);
    gpu_controller_->GPUSetup();
  }

  virtual ~FeedbackController() = default;

  virtual void initTrackingController() = 0;

  virtual void setParams(PARAMS_T& params) {
    params_ = params;
  }

  // CPU Methods
  virtual control_array k(const Eigen::Ref<state_array>& x_act,
                          const Eigen::Ref<state_array>& x_goal, float t) = 0;

  // might not be a needed method
  virtual void computeFeedbackGains(const Eigen::Ref<const state_array>& init_state,
                                    const Eigen::Ref<const state_trajectory>& goal_traj,
                                    const Eigen::Ref<const control_trajectory>& control_traj) = 0;

  // TODO apply feedback
  virtual control_array interpolateFeedback(state_array& state, state_array& target_nominal_state,
                                            feedback_gain_trajectory& gain_traj, double rel_time);

  GPU_FB_T* getDevicePointer() {
    return gpu_controller_->feedback_d_;
  }
protected:
  std::shared_ptr<GPU_FB_T> gpu_controller_;
  PARAMS_T params_;
};

#ifdef __CUDACC__
#include "feedback.cu"
#endif

#endif // FEEDBACK_BASE_CUH_
