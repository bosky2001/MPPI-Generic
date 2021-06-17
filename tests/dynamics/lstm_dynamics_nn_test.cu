//
// Created by bvlahov3 on 6/15/21.
//

#include <gtest/gtest.h>
#include <mppi/dynamics/LSTM/LSTM_model.cuh>
#include <mppi/dynamics/autorally/ar_nn_dynamics_kernel_test.cuh>
#include <stdio.h>
#include <math.h>

// Auto-generated header file
#include <autorally_test_network.h>


const int CONTROL_DIM = 2;
const int HIDDEN_DIM = 15;
const int BUFFER = 11;
const int INIT_DIM = 200;
// typedef LSTMModel<7, CONTROL_DIM, 3, 32> DYNAMICS;
using DYNAMICS = LSTMModel<7, CONTROL_DIM, 3, HIDDEN_DIM, BUFFER, INIT_DIM>;
using DYN_PARAMS = DYNAMICS::DYN_PARAMS_T;

void assert_float_array_eq(float* pred, float* gt, int max) {
  for (int i = 0; i < max; i++) {
    ASSERT_NEAR(pred[i], gt[i], 0.001 * abs(gt[i])) << "Expected "
      << gt[i] << " but saw " << pred[i] << " at " << i << std::endl;
  }
}

// Struct to look at protected variables of dynamics model
struct ModelExposer : DYNAMICS {
  DYNAMICS* model_;
  ModelExposer(DYNAMICS* model) {
    model_ = model;
  }
  __host__ __device__ DYN_PARAMS* getParamsPointer() {
    return (DYN_PARAMS*) &((ModelExposer*) model_)->params_;
  }
};

__global__ void access_params(ModelExposer* model) {
  DYN_PARAMS* params = model->getParamsPointer();
  printf("Check if copy_everything is true: %d\n", params->copy_everything);
  printf("SHARED_MEM_REQUEST_BLK: %d\n", params->SHARED_MEM_REQUEST_BLK);
  printf("W_im: %p\n", params->W_im);
  printf("W_im[0]: %f\n", params->W_im[0]);
  printf("W_fm[0]: %f\n", params->W_fm[0]);
  printf("dt: %f\n", params->dt);
}

template<class DYN_T, int NUM_ROLLOUTS=1, int BLOCKSIZE_X = 1, int BLOCKSIZE_Z = 1>
__global__ void run_dynamics(DYN_T* dynamics, float* initial_state,
                             float* control, float* state_der) {
  int thread_idx = threadIdx.x;
  int thread_idy = threadIdx.y;
  int thread_idz = threadIdx.z;
  int block_idx = blockIdx.x;
  int global_idx = blockDim.x * block_idx + thread_idx;

  // Create shared state and control arrays
  __shared__ float x_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float xdot_shared[BLOCKSIZE_X * DYN_T::STATE_DIM * BLOCKSIZE_Z];
  __shared__ float u_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float du_shared[BLOCKSIZE_X * DYN_T::CONTROL_DIM * BLOCKSIZE_Z];
  __shared__ float sigma_u[DYN_T::CONTROL_DIM];
  __shared__ int crash_status_shared[BLOCKSIZE_X*BLOCKSIZE_Z];

  // Create a shared array for the dynamics model to use
  __shared__ float theta_s[DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X*BLOCKSIZE_Z];
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    printf("Amount shared memory jsut for theta %d\n",
      DYN_T::SHARED_MEM_REQUEST_GRD + DYN_T::SHARED_MEM_REQUEST_BLK*BLOCKSIZE_X*BLOCKSIZE_Z);
  }
  float* x;
  float* xdot;
  float* u;
  // float* du;
  // int* crash_status;
  if (global_idx < NUM_ROLLOUTS) {
    x = &x_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    xdot = &xdot_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::STATE_DIM];
    u = &u_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    // du = &du_shared[(blockDim.x * thread_idz + thread_idx) * DYN_T::CONTROL_DIM];
    // crash_status = &crash_status_shared[thread_idz*blockDim.x + thread_idx];
    // crash_status[0] = 0; // We have not crashed yet as of the first trajectory.

    //__syncthreads();
    for (int i = threadIdx.y; i < DYN_T::STATE_DIM; i += blockDim.y) {
      x[i] = initial_state[i];
    }
    for (int i = threadIdx.y; i < DYN_T::CONTROL_DIM; i += blockDim.y) {
      u[i] = control[i];
    }
    __syncthreads();
    /*<----Start of simulation loop-----> */
    dynamics->initializeDynamics(x, u, theta_s, 0.0, 0.0);
    dynamics->computeStateDeriv(x, u, xdot, theta_s);

    __syncthreads();
    for (int i = threadIdx.y; i < DYN_T::STATE_DIM; i += blockDim.y) {
      state_der[global_idx * DYN_T::STATE_DIM + i] = xdot[i];
    }
  }
}

class LSTMDynamicsTest: public ::testing::Test {
public:
  cudaStream_t stream;

  std::array<float2, CONTROL_DIM> u_constraint = {};

  virtual void SetUp() {
    HANDLE_ERROR(cudaStreamCreate(&stream));
    u_constraint[0].x = -1.0;
    u_constraint[0].y = 1.0;

    u_constraint[1].x = -2.0;
    u_constraint[1].y = 2.0;
  }
};

TEST_F(LSTMDynamicsTest, BindStreamControlRanges) {
  DYNAMICS model(u_constraint, stream);
  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST_F(LSTMDynamicsTest, CopyParams) {
  DYNAMICS model(u_constraint, stream);

  model.GPUSetup();
  auto dyn_params = model.getParams();
  dyn_params.W_fm[0] = 5.0;
  dyn_params.copy_everything = true;
  dyn_params.dt = 5;
  dyn_params.W_hidden_input.get()[0] = 13;
  model.setParams(dyn_params);
  ModelExposer access_cpu_model(&model);

  std::cout << "Params Size: " << sizeof(dyn_params) << " bytes" << std::endl;
  // std::cout << "Weight Size: " << sizeof(dyn_params.W_im) / sizeof(float) << std::endl;
  std::cout << "Weight_im: " << dyn_params.W_im << std::endl;
  std::cout << "Hidden State Initialization Weight references: "
            << access_cpu_model.getParamsPointer()->W_hidden_input.use_count()
            << std::endl;
  std::cout << "Hidden State Initialization Weight[0]: "
            << access_cpu_model.getParamsPointer()->W_hidden_input.get()[0]
            << std::endl;

  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(1, 1, 1);
  ModelExposer access_gpu_model(model.model_d_);
  ModelExposer* access_gpu_model_d_;
  HANDLE_ERROR(cudaMalloc((void **)&access_gpu_model_d_, sizeof(ModelExposer)));
  HANDLE_ERROR(cudaMemcpyAsync(access_gpu_model_d_, &access_gpu_model, sizeof(ModelExposer),cudaMemcpyHostToDevice, stream));
  access_params<<<dimGrid, dimBlock, 0, stream>>>(access_gpu_model_d_);
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  EXPECT_EQ(model.stream_, stream) << "Stream binding failure.";

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST_F(LSTMDynamicsTest, LoadWeights) {
  DYNAMICS model(u_constraint, stream);

  int BUFFER_SIZE = 11 * 6;
  const int num_rollouts = 10;
  const int blocksize_x = 32;

  model.GPUSetup();
  model.loadParams(mppi::tests::autorally_lstm_network_file,
                   mppi::tests::autorally_hidden_network_file,
                   mppi::tests::autorally_cell_network_file,
                   mppi::tests::autorally_output_network_file);


  // std::cout << "W_hidden_initial network:\n";
  // for (int i = 0; i < dyn_params.BUFFER_INTER_SIZE; i++) {
  //   if (i % (11 * 6) < 7) {
  //     std::cout << dyn_params.W_hidden_input.get()[i] << ", ";
  //   } else if (i % (11 * 6) == 7) {
  //     std::cout << dyn_params.W_hidden_input.get()[i] << "\n";
  //   }
  // }
  // std::cout << std::endl;
  // std::cout << "Size: " << dyn_params.BUFFER_INTER_SIZE / dyn_params.INTER_DIM << ", "
  //           << dyn_params.INTER_DIM << std::endl;

  std::vector<float> x_0 = {2.9642e-04,  5.7054e+00,  1.1859e-03,  1.3721e-01,  2.4944e-02, 1.2798e-01};
  std::vector<float> x_1 = {7.8346e-04,  5.6928e+00, -1.4520e-02,  1.7258e-01, -3.1522e-03,
          8.4512e-02};
  std::vector<float> x_2 = {7.5389e-04,  5.6884e+00, -1.9062e-02,  4.5813e-04, -2.3523e-02,
          6.8172e-02};
  std::vector<float> x_3 = {1.5670e-03,  5.6779e+00,  5.7993e-03, -9.1165e-02, -2.5202e-02,
          9.0036e-02};
  std::vector<float> x_4 = {2.0307e-03,  5.6623e+00,  3.5971e-02, -1.4233e-01, -1.4520e-02,
          1.3751e-01};
  std::vector<float> x_5 = {6.6427e-04,  5.6565e+00,  4.3000e-02, -2.1955e-02, -2.1740e-03,
          1.9203e-01};
  std::vector<float> x_6 = {2.1942e-04,  5.6636e+00,  1.6840e-02, -5.7120e-03,  3.3988e-03,
          2.3751e-01};
  std::vector<float> x_7 = {6.9824e-04,  5.6656e+00, -4.1707e-04, -4.3693e-02,  2.9118e-03,
          2.6795e-01};
  std::vector<float> x_8 = {1.2957e-03,  5.6861e+00,  2.8441e-03, -1.1037e-01, -6.0677e-04,
          2.7991e-01};
  std::vector<float> x_9 = {8.7452e-04,  5.7010e+00,  2.2052e-02, -2.5667e-02, -4.2457e-03,
          2.7212e-01};
  std::vector<float> x_10 = {7.2980e-04,  5.7185e+00,  2.0501e-02, -4.2951e-02, -5.5691e-03,
          2.5228e-01};
  std::vector<int> description = {4, 2};
  model.updateModel(description, x_0);

  DYN_PARAMS dyn_params = model.getParams();
  std::cout << "BUFFER State after 1 update:\n";
  for (int i = 0; i < BUFFER_SIZE; i++) {
    std::cout << dyn_params.buffer[i] << ", ";
    if (i % 6 == 5) {
      std::cout << "\n";
    }
  }
  std::cout << "Initial x: " << x_0[0] << ", " << x_0[1] << std::endl;
  model.updateModel(description, x_1);
  model.updateModel(description, x_2);
  model.updateModel(description, x_3);
  model.updateModel(description, x_4);
  model.updateModel(description, x_5);
  model.updateModel(description, x_6);
  model.updateModel(description, x_7);
  model.updateModel(description, x_8);
  model.updateModel(description, x_9);
  model.updateModel(description, x_10);
  dyn_params = model.getParams();

  std::cout << "BUFFER State:\n";
  for (int i = 0; i < BUFFER_SIZE; i++) {
    std::cout << dyn_params.buffer[i] << ", ";
    if (i % 6 == 5) {
      std::cout << "\n";
    }
  }
  std::cout << std::endl;
  std::cout << "Initial Hidden State:\n";
  for (int i = 0; i < HIDDEN_DIM; i++) {
    std::cout << dyn_params.initial_hidden[i] << ", ";
  }
  std::cout << std::endl;
  float python_initial_hidden[HIDDEN_DIM] = {1.5683, -1.9951,  1.1584, -0.7979,  1.4718,  0.4733, -0.0247, -2.4777,
          0.7693,  0.1656, -1.2776, -0.7229, -0.0814,  0.2607,  0.2293};
  float python_initial_cell[HIDDEN_DIM] = {-1.3588, -0.3638,  0.4151,  0.3086, -0.4807,  0.1482,  0.2792,  0.2688,
         -0.5454, -1.4281,  0.2832,  0.3822, -0.1447, -0.4664,  1.6805};
  assert_float_array_eq(dyn_params.initial_hidden, python_initial_hidden, HIDDEN_DIM);
  assert_float_array_eq(dyn_params.initial_cell, python_initial_cell, HIDDEN_DIM);

  float initial_state_cpu[DYNAMICS::STATE_DIM] = {0.0, 0.0, 0.0, 7.2980e-04,  5.7185e+00,  2.0501e-02, -4.2951e-02};
  float control_cpu[DYNAMICS::CONTROL_DIM] = {-5.5691e-03, 2.5228e-01};
  float state_der_cpu[num_rollouts * DYNAMICS::STATE_DIM] = {0.0};

  float* initial_state_gpu;
  float* control_gpu;
  float* state_der_gpu;
  HANDLE_ERROR(cudaMalloc((void**)&initial_state_gpu,
                           sizeof(float) * DYNAMICS::STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&control_gpu,
                           sizeof(float) * DYNAMICS::CONTROL_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&state_der_gpu,
                           sizeof(float) * num_rollouts * DYNAMICS::STATE_DIM));
  HANDLE_ERROR(cudaMemcpyAsync(initial_state_gpu, initial_state_cpu,
                               sizeof(float) * DYNAMICS::STATE_DIM,
                               cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR(cudaMemcpyAsync(control_gpu, control_cpu,
                               sizeof(float) * DYNAMICS::CONTROL_DIM,
                               cudaMemcpyHostToDevice, stream));
  const int gridsize_x = (num_rollouts - 1) / blocksize_x + 1;
  dim3 dimBlock(blocksize_x, 16, 1);
  dim3 dimGrid(gridsize_x, 1, 1);
  std::cout << "Launching dynamics kernel" << std::endl;
  run_dynamics<DYNAMICS, num_rollouts, blocksize_x><<<dimGrid, dimBlock, 0,
    stream>>>(model.model_d_, initial_state_gpu, control_gpu, state_der_gpu);
  HANDLE_ERROR(cudaMemcpyAsync(state_der_cpu, state_der_gpu,
                               sizeof(float) * num_rollouts * DYNAMICS::STATE_DIM,
                               cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  std::cout << "Finished dynamics kernel" << std::endl;
  std::cout << "State Der:\n";
  for (int i = 0; i < num_rollouts; i++) {
    std::cout << "Rollout " << i << ": ";
    for (int j = 0; j < DYNAMICS::STATE_DIM; j++) {
      std::cout << state_der_cpu[i * DYNAMICS::STATE_DIM + j] << ", ";
    }
    std::cout << std::endl;
  }
  float expected_state_deriv[4] = {-0.2720,  0.8784, -1.1101,  1.3801};
  for (int i = 0; i < num_rollouts; i++) {
    assert_float_array_eq(&state_der_cpu[i * DYNAMICS::STATE_DIM + 3], expected_state_deriv, 4);
  }


  // HANDLE_ERROR(cudaMalloc((void **)&access_gpu_model_d_, sizeof(ModelExposer)));
  // HANDLE_ERROR(cudaMemcpyAsync(access_gpu_model_d_, &access_gpu_model, sizeof(ModelExposer),cudaMemcpyHostToDevice, stream));

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

TEST_F(LSTMDynamicsTest, NewConstructor) {
  DYNAMICS* model = new DYNAMICS(u_constraint, stream);
  std::cout << "Float Size: " << sizeof(float) << " bytes" << std::endl;

  EXPECT_EQ(model->stream_, stream) << "Stream binding failure.";
  // delete model;
  HANDLE_ERROR(cudaStreamDestroy(stream));
}