#include <mppi/dynamics/racer_dubins/racer_dubins_elevation.cuh>
#include <mppi/utils/math_utils.h>

void RacerDubinsElevation::GPUSetup()
{
  RacerDubinsImpl<RacerDubinsElevation>* derived = static_cast<RacerDubinsImpl<RacerDubinsElevation>*>(this);
  CudaCheckError();
  tex_helper_->GPUSetup();
  CudaCheckError();
  derived->GPUSetup();
  CudaCheckError();
}

void RacerDubinsElevation::freeCudaMem()
{
  tex_helper_->freeCudaMem();
}

void RacerDubinsElevation::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // does all the internal texture updates
    tex_helper_->copyToDevice();
    // makes sure that the device ptr sees the correct texture object
    HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                                 sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
  }
  RacerDubinsImpl<RacerDubinsElevation>::paramsToDevice();
}

// void RacerDubinsElevation::updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
//                                        Eigen::Ref<state_array> state_der, const float dt)
// {
//   next_state = state + state_der * dt;
//   next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
//   next_state(S_INDEX(STEER_ANGLE)) -= state_der(S_INDEX(STEER_ANGLE)) * dt;
//   next_state(S_INDEX(STEER_ANGLE)) =
//       state_der(S_INDEX(STEER_ANGLE)) + (next_state(S_INDEX(STEER_ANGLE)) - state_der(S_INDEX(STEER_ANGLE))) *
//                                             expf(-this->params_.steering_constant * dt);
//   state_der.setZero();
// }

// __device__ void RacerDubinsElevation::updateState(float* state, float* next_state, float* state_der, const float dt)
// {
//   int i;
//   int tdy = threadIdx.y;
//   // Add the state derivative time dt to the current state.
//   // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
//   for (i = tdy; i < PARENT_CLASS::STATE_DIM; i += blockDim.y)
//   {
//     next_state[i] = state[i] + state_der[i] * dt;
//     if (i == S_INDEX(YAW))
//     {
//       next_state[i] = angle_utils::normalizeAngle(next_state[i]);
//     }
//     if (i == S_INDEX(STEER_ANGLE))
//     {
//       next_state[i] -= state_der[i] * dt;
//       next_state[i] = state_der[i] + (next_state[i] - state_der[i]) * expf(-this->params_.steering_constant * dt);
//       // next_state[i] += state_der[i] * expf(-this->params_.steering_constant * dt);
//     }
//     state_der[i] = 0;  // Important: reset the next_state derivative to zero.
//   }
// }

void RacerDubinsElevation::computeStateDeriv(const Eigen::Ref<const state_array>& state,
                                             const Eigen::Ref<const control_array>& control,
                                             Eigen::Ref<state_array> state_der)
{
  float pitch = 0;
  float roll = 0;

  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);
  front_left = make_float3(front_left.x * cosf(state(1)) - front_left.y * sinf(state(1)) + state(2),
                           front_left.x * sinf(state(1)) + front_left.y * cosf(state(1)) + state(3), 0);
  front_right = make_float3(front_right.x * cosf(state(1)) - front_right.y * sinf(state(1)) + state(2),
                            front_right.x * sinf(state(1)) + front_right.y * cosf(state(1)) + state(3), 0);
  rear_left = make_float3(rear_left.x * cosf(state(1)) - rear_left.y * sinf(state(1)) + state(2),
                          rear_left.x * sinf(state(1)) + rear_left.y * cosf(state(1)) + state(3), 0);
  rear_right = make_float3(rear_right.x * cosf(state(1)) - rear_right.y * sinf(state(1)) + state(2),
                           rear_right.x * sinf(state(1)) + rear_right.y * cosf(state(1)) + state(3), 0);
  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;

  if (this->tex_helper_->checkTextureUse(0))
  {
    front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
    front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
    rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
    rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);

    float front_diff = front_left_height - front_right_height;
    front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
    float rear_diff = rear_left_height - rear_right_height;
    rear_diff = max(min(rear_diff, 0.736 * 2), -0.736 * 2);
    float front_roll = asinf(front_diff / (0.737 * 2));
    float rear_roll = asinf(rear_diff / (0.737 * 2));
    if (abs(front_roll) > abs(rear_roll))
    {
      roll = front_roll;
    }
    else
    {
      roll = rear_roll;
    }

    float left_diff = rear_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98), -2.98);
    float right_diff = rear_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98), -2.98);
    float left_pitch = asinf((left_diff) / 2.981);
    float right_pitch = asinf((right_diff) / 2.981);
    if (abs(left_pitch) > abs(right_pitch))
    {
      pitch = left_pitch;
    }
    else
    {
      pitch = right_pitch;
    }
    if (isnan(roll) || isinf(roll) || abs(roll) > M_PI)
    {
      roll = 0;
    }
    if (isnan(pitch) || isinf(pitch) || abs(pitch) > M_PI)
    {
      pitch = 0;
    }
  }

  bool enable_brake = control(0) < 0;
  // applying position throttle
  state_der(0) = (!enable_brake) * this->params_.c_t * control(0) +
                 (enable_brake) * this->params_.c_b * control(0) * (state(0) >= 0 ? 1 : -1) -
                 this->params_.c_v * state(0) + this->params_.c_0;

  state_der[0] -= this->params_.gravity * sinf(pitch);
  state_der(1) = (state(0) / this->params_.wheel_base) * tan(state(4));
  float yaw = state[S_INDEX(YAW)];
  state_der(2) = state(0) * cosf(yaw);
  state_der(3) = state(0) * sinf(yaw);
  state_der(4) = control(1) / this->params_.steer_command_angle_scale;
}
void RacerDubinsElevation::step(Eigen::Ref<state_array>& state, Eigen::Ref<state_array>& next_state,
                                Eigen::Ref<state_array>& state_der, const Eigen::Ref<const control_array>& control,
                                Eigen::Ref<output_array>& output, const float t, const float dt)
{
  computeStateDeriv(state, control, state_der);
  updateState(state, next_state, state_der, dt);

  output[O_INDEX(BASELINK_VEL_B_X)] = state[S_INDEX(VEL_X)];
  output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
  output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
  output[O_INDEX(BASELINK_POS_I_X)] = state[S_INDEX(POS_X)];
  output[O_INDEX(BASELINK_POS_I_Y)] = state[S_INDEX(POS_Y)];
  output[O_INDEX(BASELINK_POS_I_Z)] = 0;
  output[O_INDEX(OMEGA_B_X)] = 0;
  output[O_INDEX(OMEGA_B_Y)] = 0;
  output[O_INDEX(OMEGA_B_Z)] = 0;
  output[O_INDEX(YAW)] = yaw;
  output[O_INDEX(PITCH)] = pitch;
  output[O_INDEX(ROLL)] = roll;
  Eigen::Quaternionf q;
  mppi::math::Euler2QuatNWU(roll, pitch, yaw, q);
  output[O_INDEX(ATTITUDE_QW)] = q.w();
  output[O_INDEX(ATTITUDE_QX)] = q.x();
  output[O_INDEX(ATTITUDE_QY)] = q.y();
  output[O_INDEX(ATTITUDE_QZ)] = q.z();
  output[O_INDEX(STEER_ANGLE)] = state[S_INDEX(STEER_ANGLE)];
  output[O_INDEX(STEER_ANGLE_RATE)] = 0;
  output[O_INDEX(WHEEL_POS_I_FL_X)] = front_left.x;
  output[O_INDEX(WHEEL_POS_I_FL_Y)] = front_left.y;
  output[O_INDEX(WHEEL_POS_I_FR_X)] = front_right.x;
  output[O_INDEX(WHEEL_POS_I_FR_Y)] = front_right.y;
  output[O_INDEX(WHEEL_POS_I_RL_X)] = rear_left.x;
  output[O_INDEX(WHEEL_POS_I_RL_Y)] = rear_left.y;
  output[O_INDEX(WHEEL_POS_I_RR_X)] = rear_right.x;
  output[O_INDEX(WHEEL_POS_I_RR_Y)] = rear_right.y;
  output[O_INDEX(WHEEL_FORCE_B_FL_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FL_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FL_Z)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_FR_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FR_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_FR_Z)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RL_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RL_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RL_Z)] = 10000;
  output[O_INDEX(WHEEL_FORCE_B_RR_X)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RR_Y)] = 0;
  output[O_INDEX(WHEEL_FORCE_B_RR_Z)] = 10000;
  output[O_INDEX(CENTER_POS_I_X)] = output[O_INDEX(BASELINK_POS_I_X)];  // TODO
  output[O_INDEX(CENTER_POS_I_Y)] = output[O_INDEX(BASELINK_POS_I_Y)];
  output[O_INDEX(CENTER_POS_I_Z)] = 0;
}

__device__ void RacerDubinsElevation::computeStateDeriv(float* state, float* control, float* state_der, float* theta_s)
{
  float pitch = 0;
  float roll = 0;

  float3 front_left = make_float3(2.981, 0.737, 0);
  float3 front_right = make_float3(2.981, -0.737, 0);
  float3 rear_left = make_float3(0, 0.737, 0);
  float3 rear_right = make_float3(0, -0.737, 0);
  front_left = make_float3(front_left.x * cosf(state[1]) - front_left.y * sinf(state[1]) + state[2],
                           front_left.x * sinf(state[1]) + front_left.y * cosf(state[1]) + state[3], 0);
  front_right = make_float3(front_right.x * cosf(state[1]) - front_right.y * sinf(state[1]) + state[2],
                            front_right.x * sinf(state[1]) + front_right.y * cosf(state[1]) + state[3], 0);
  rear_left = make_float3(rear_left.x * cosf(state[1]) - rear_left.y * sinf(state[1]) + state[2],
                          rear_left.x * sinf(state[1]) + rear_left.y * cosf(state[1]) + state[3], 0);
  rear_right = make_float3(rear_right.x * cosf(state[1]) - rear_right.y * sinf(state[1]) + state[2],
                           rear_right.x * sinf(state[1]) + rear_right.y * cosf(state[1]) + state[3], 0);
  float front_left_height = 0;
  float front_right_height = 0;
  float rear_left_height = 0;
  float rear_right_height = 0;

  if (this->tex_helper_->checkTextureUse(0))
  {
    front_left_height = this->tex_helper_->queryTextureAtWorldPose(0, front_left);
    front_right_height = this->tex_helper_->queryTextureAtWorldPose(0, front_right);
    rear_left_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_left);
    rear_right_height = this->tex_helper_->queryTextureAtWorldPose(0, rear_right);

    // max magnitude
    float front_diff = front_left_height - front_right_height;
    front_diff = max(min(front_diff, 0.736 * 2), -0.736 * 2);
    float rear_diff = rear_left_height - rear_right_height;
    rear_diff = max(min(rear_diff, 0.736 * 2), -0.736 * 2);
    float front_roll = asinf(front_diff / (0.737 * 2));
    float rear_roll = asinf(rear_diff / (0.737 * 2));
    if (abs(front_roll) > abs(rear_roll))
    {
      roll = front_roll;
    }
    else
    {
      roll = rear_roll;
    }
    float left_diff = rear_left_height - front_left_height;
    left_diff = max(min(left_diff, 2.98), -2.98);
    float right_diff = rear_right_height - front_right_height;
    right_diff = max(min(right_diff, 2.98), -2.98);
    float left_pitch = asinf((left_diff) / 2.981);
    float right_pitch = asinf((right_diff) / 2.981);
    if (abs(left_pitch) > abs(right_pitch))
    {
      pitch = left_pitch;
    }
    else
    {
      pitch = right_pitch;
    }

    if (isnan(roll) || isinf(roll) || abs(roll) > M_PI)
    {
      roll = 0;
    }
    if (isnan(pitch) || isinf(pitch) || abs(pitch) > M_PI)
    {
      pitch = 0;
    }
  }

  bool enable_brake = control[0] < 0;
  // applying position throttle
  state_der[0] = (!enable_brake) * this->params_.c_t * control[0] +
                 (enable_brake) * this->params_.c_b * control[0] * (state[0] >= 0 ? 1 : -1) -
                 this->params_.c_v * state[0] + this->params_.c_0;
  state_der[0] -= this->params_.gravity * sinf(pitch);
  float yaw = state[S_INDEX(YAW)];
  state_der[1] = (state[0] / this->params_.wheel_base) * tan(state[4]);
  state_der[2] = state[0] * cosf(yaw);
  state_der[3] = state[0] * sinf(yaw);
  state_der[4] = control[1] / this->params_.steer_command_angle_scale;
}

__device__ inline void RacerDubinsElevation::step(float* state, float* next_state, float* state_der, float* control,
                                                  float* output, float* theta_s, const float t, const float dt)
{
  computeStateDeriv(state, control, state_der, theta_s);
  __syncthreads();
  updateState(state, next_state, state_der, dt);
  __syncthreads();
  if (output)
  {
    output[O_INDEX(BASELINK_VEL_B_X)] = state[S_INDEX(VEL_X)];
    output[O_INDEX(BASELINK_VEL_B_Y)] = 0;
    output[O_INDEX(BASELINK_VEL_B_Z)] = 0;
    output[O_INDEX(BASELINK_POS_I_X)] = state[S_INDEX(POS_X)];
    output[O_INDEX(BASELINK_POS_I_Y)] = state[S_INDEX(POS_Y)];
    output[O_INDEX(BASELINK_POS_I_Z)] = 0;
    output[O_INDEX(OMEGA_B_X)] = 0;
    output[O_INDEX(OMEGA_B_Y)] = 0;
    output[O_INDEX(OMEGA_B_Z)] = 0;
    output[O_INDEX(YAW)] = yaw;
    output[O_INDEX(PITCH)] = pitch;
    output[O_INDEX(ROLL)] = roll;
    Eigen::Quaternionf q;
    mppi::math::Euler2QuatNWU(roll, pitch, yaw, q);
    output[O_INDEX(ATTITUDE_QW)] = q.w();
    output[O_INDEX(ATTITUDE_QX)] = q.x();
    output[O_INDEX(ATTITUDE_QY)] = q.y();
    output[O_INDEX(ATTITUDE_QZ)] = q.z();
    output[O_INDEX(STEER_ANGLE)] = state[S_INDEX(STEER_ANGLE)];
    output[O_INDEX(STEER_ANGLE_RATE)] = 0;
    output[O_INDEX(WHEEL_POS_I_FL_X)] = front_left.x;
    output[O_INDEX(WHEEL_POS_I_FL_Y)] = front_left.y;
    output[O_INDEX(WHEEL_POS_I_FR_X)] = front_right.x;
    output[O_INDEX(WHEEL_POS_I_FR_Y)] = front_right.y;
    output[O_INDEX(WHEEL_POS_I_RL_X)] = rear_left.x;
    output[O_INDEX(WHEEL_POS_I_RL_Y)] = rear_left.y;
    output[O_INDEX(WHEEL_POS_I_RR_X)] = rear_right.x;
    output[O_INDEX(WHEEL_POS_I_RR_Y)] = rear_right.y;
    output[O_INDEX(WHEEL_FORCE_B_FL_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FL_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FL_Z)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_FR_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FR_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_FR_Z)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_RL_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RL_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RL_Z)] = 10000;
    output[O_INDEX(WHEEL_FORCE_B_RR_X)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RR_Y)] = 0;
    output[O_INDEX(WHEEL_FORCE_B_RR_Z)] = 10000;
    output[O_INDEX(CENTER_POS_I_X)] = output[O_INDEX(BASELINK_POS_I_X)];  // TODO
    output[O_INDEX(CENTER_POS_I_Y)] = output[O_INDEX(BASELINK_POS_I_Y)];
    output[O_INDEX(CENTER_POS_I_Z)] = 0;
  }
}
