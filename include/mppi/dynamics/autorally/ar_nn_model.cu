
#include "ar_nn_model.cuh"

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(std::array<float2, C_DIM> control_rngs,
                                                                   cudaStream_t stream)
  : PARENT_CLASS(control_rngs, stream)
{
  helper_ = new FNNHelper<FNNParams<layer_args...>>(stream);
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(cudaStream_t stream) : PARENT_CLASS(stream)
{
  helper_ = new FNNHelper<FNNParams<layer_args...>>(stream);
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::~NeuralNetModel()
{
  free(helper_);
  freeCudaMem();
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::freeCudaMem()
{
  helper_->freeCudaMem();
  PARENT_CLASS::freeCudaMem();
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::updateModel(std::vector<int> description,
                                                                     std::vector<float> data)
{
  helper_->updateModel(description, data);
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    helper_->paramsToDevice();
  }
  PARENT_CLASS::paramsToDevice();
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::loadParams(const std::string& model_path)
{
  helper_->loadParams(model_path);
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
bool NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeGrad(const Eigen::Ref<const state_array>& state,
                                                                     const Eigen::Ref<const control_array>& control,
                                                                     Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B)
{
  Eigen::Matrix<float, S_DIM, S_DIM + C_DIM> jac;
  jac.setZero();

  // Start with the kinematic and physics model derivatives
  jac.row(0) << 0, 0, -sin(state(2)) * state(4) - cos(state(2)) * state(5), 0, cos(state(2)), -sin(state(2)), 0, 0, 0;
  jac.row(1) << 0, 0, cos(state(2)) * state(4) - sin(state(2)) * state(5), 0, sin(state(2)), cos(state(2)), 0, 0, 0;
  jac.row(2) << 0, 0, 0, 0, 0, 0, -1, 0, 0;

  state_array state_der;

  // First do the forward pass
  computeDynamics(state, control, state_der);

  nn_dfdx ip_delta;
  helper_->computeGrad(ip_delta);

  jac.bottomRightCorner(DYNAMICS_DIM, DYNAMICS_DIM + C_DIM) += ip_delta;
  A = jac.leftCols(S_DIM);
  B = jac.rightCols(C_DIM);
  return true;
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeKinematics(const Eigen::Ref<const state_array>& state,
                                                                           Eigen::Ref<state_array> state_der)
{
  state_der(0) = cosf(state(2)) * state(4) - sinf(state(2)) * state(5);
  state_der(1) = sinf(state(2)) * state(4) + cosf(state(2)) * state(5);
  state_der(2) = -state(6);  // Pose estimate actually gives the negative yaw derivative
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeDynamics(const Eigen::Ref<const state_array>& state,
                                                                         const Eigen::Ref<const control_array>& control,
                                                                         Eigen::Ref<state_array> state_der)
{
  nn_input_array input;
  nn_output_array output;
  int i, j;
  for (i = 0; i < DYNAMICS_DIM; i++)
  {
    input(i) = state(i + (S_DIM - DYNAMICS_DIM));
  }
  for (i = 0; i < C_DIM; i++)
  {
    input(DYNAMICS_DIM + i) = control(i);
  }
  helper_->forward(input, output);
  for (i = 0; i < DYNAMICS_DIM; i++)
  {
    state_der(i + (S_DIM - DYNAMICS_DIM)) = output[i];
  }
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeKinematics(float* state, float* state_der)
{
  state_der[0] = cosf(state[2]) * state[4] - sinf(state[2]) * state[5];
  state_der[1] = sinf(state[2]) * state[4] + cosf(state[2]) * state[5];
  state_der[2] = -state[6];  // Pose estimate actually gives the negative yaw derivative
}

template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
__device__ void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::computeDynamics(float* state, float* control,
                                                                                    float* state_der, float* theta_s)
{
  float* curr_act;
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;
  int tdz = threadIdx.z;
  int i, j, k;
  curr_act = &theta_s[(2 * LARGEST_LAYER) * (blockDim.x * tdz + tdx)];
  // iterate through the part of the state that should be an input to the NN
  for (i = tdy; i < DYNAMICS_DIM; i += blockDim.y)
  {
    curr_act[i] = state[i + (S_DIM - DYNAMICS_DIM)];
  }
  // iterate through the control to put into first layer
  for (i = tdy; i < C_DIM; i += blockDim.y)
  {
    curr_act[DYNAMICS_DIM + i] = control[i];
  }
  __syncthreads();

  curr_act = helper_->forward(nullptr, theta_s);

  // copies results back into state derivative
  for (i = tdy; i < DYNAMICS_DIM; i += blockDim.y)
  {
    state_der[i + (S_DIM - DYNAMICS_DIM)] = curr_act[i];
  }
  __syncthreads();
}
template <int S_DIM, int C_DIM, int K_DIM, int... layer_args>
void NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::GPUSetup()
{
  PARENT_CLASS* derived = static_cast<PARENT_CLASS*>(this);
  helper_->GPUSetup();
  derived->GPUSetup();

  HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->helper_), &(this->helper_->network_d_),
                               sizeof(FNNHelper<FNNParams<layer_args...>>*), cudaMemcpyHostToDevice, this->stream_));
}
