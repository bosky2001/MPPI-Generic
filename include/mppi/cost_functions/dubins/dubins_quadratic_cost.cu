#include <mppi/cost_functions/dubins/dubins_quadratic_cost.cuh>

DubinsQuadraticCost::DubinsQuadraticCost(cudaStream_t stream)
{
  bindToStream(stream);
}


float DubinsQuadraticCost::computeStateCost(const Eigen::Ref<const output_array> s, int timestep, int* crash_status)
{
  // Position tracking error 
    float position_cost = (s[0] - params_.desired_terminal_state[0]) * (s[0] - params_.desired_terminal_state[0]) *
                         params_.position_coeff +
                         (s[1] - params_.desired_terminal_state[1]) * (s[1] - params_.desired_terminal_state[1]) *
                         params_.position_coeff;

    // Geometric yaw tracking error 
    float yaw_cost = (sinf(s[2]) - sinf(params_.desired_terminal_state[2])) * 
                     (sinf(s[2]) - sinf(params_.desired_terminal_state[2])) +
                     (cosf(s[2]) - cosf(params_.desired_terminal_state[2])) * 
                     (cosf(s[2]) - cosf(params_.desired_terminal_state[2])) * params_.yaw_coeff;
    

    return position_cost + yaw_cost;
//   const float pos_x = y[O_IND_CLASS(DYN_P, POS_X)]; // get the x position from the state vector y
//   const float pos_y = y[O_IND_CLASS(DYN_P, POS_Y)]; // get the y position from the state vector y
//   const float3 pos = make_float3(pos_x, pos_y, 0);

//   float map_query = 0;
//   // Check that the map has been filled in (set true by enableTexture())
//   if (map_helper_->checkTextureUse(0)) {
//         map_query = map_helper_->queryTextureAtWorldPose(0, pos);
//     }
  
//   float obstacle_cost = 0;
//   obstacle_cost += map_query * params_.obstacle_cost_coeff

//   return obstacle_cost + tracking_cost;
 
}

__device__ float DubinsQuadraticCost::computeStateCost(float* state, int timestep, float* theta_c, int* crash_status)
{
  // Position tracking error 
    float position_cost = (state[0] - params_.desired_terminal_state[0]) * (state[0] - params_.desired_terminal_state[0]) *
                         params_.position_coeff +
                         (state[1] - params_.desired_terminal_state[1]) * (state[1] - params_.desired_terminal_state[1]) *
                         params_.position_coeff;

    // Geometric yaw tracking error 
    float yaw_cost = (sinf(state[2]) - sinf(params_.desired_terminal_state[2])) * 
                     (sinf(state[2]) - sinf(params_.desired_terminal_state[2])) +
                     (cosf(state[2]) - cosf(params_.desired_terminal_state[2])) * 
                     (cosf(state[2]) - cosf(params_.desired_terminal_state[2])) * params_.yaw_coeff;
    

    return position_cost + yaw_cost;
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


// DubinsQuadraticCost::fillMap(const nav_msgs::OccupancyGrid::ConstPtr& ros_grid_msg_ptr)
// {   
//     // To Update resolution of map( 0 means 0th map)
//     map_helper_->updateResolution(0, ros_grid_msg_ptr->info.resolution);

//     // To update origin of map
//     float3 origin = make_float3(ros_grid_msg_ptr->info.origin.position.x,
//     ros_grid_msg_ptr->info.origin.position.y, 0);
//     map_helper_->updateOrigin(0, origin);

//     // To update map rotation
//     Eigen::Quaternionf quat;
//     quat.x() = ros_grid_msg_ptr->info.origin.orientation.x;
//     quat.y() = ros_grid_msg_ptr->info.origin.orientation.y;
//     quat.z() = ros_grid_msg_ptr->info.origin.orientation.z;
//     quat.w() = ros_grid_msg_ptr->info.origin.orientation.w;

//     Eigen::Matrix3f R = quat.toRotationMatrix();
//     std::array<float3, 3> map_helper_R;
//     // You might need to transpose R before filling in map_helper_R
//     map_helper_R[0] = make_float3(R(0,0), R(0,1), R(0,2));
//     map_helper_R[1] = make_float3(R(1,0), R(1,1), R(1,2));
//     map_helper_R[2] = make_float3(R(2,0), R(2,1), R(2,2));
//     map_helper_->updateRotation(0, map_helper_R);

//     // To update map dimensions
//     cudaExtent map_dim = make_cudaExtent(ros_grid_msg_ptr->info.width, ros_grid_msg_ptr->info.height, 0);
//     map_helper_->setExtent(0, map_dim);

//     // Filling in the occupancy grid
//     std::vector<float> map_data;
//     // Put ros_grid_msg_ptr->data into map_data.

//     // Set this bool to true if map_data is stored in column-major order
//     bool column_major = false;
//     map_helper_->updateTexture(0, map_data, column_major);

//     // Enabling the texture is used to assert to the cost function code that the map has been filled with data.
//     map_helper_->enableTexture(0);

// }
