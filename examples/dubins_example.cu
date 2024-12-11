#include <mppi/controllers/MPPI/mppi_controller.cuh>

#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/cost_functions/dubins/dubins_quadratic_cost.cuh>
#include <mppi/dynamics/dubins/dubins.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <iostream>
#include <chrono>



using SAMPLER_T = mppi::sampling_distributions::GaussianDistribution<DubinsDynamics::DYN_PARAMS_T>;



int main(int argc, char** argv)

{

  auto model = new DubinsDynamics();
  std::cout <<":loololololololololol"<< "\n";
  auto cost = new DubinsQuadraticCost();
//   auto test_cost = new DubinsQuadraticCost();

  // delete (cost);
  // delete (model);
  // sets the controls input limits( default -inf to inf)
//   model->control_rngs_->x = -5;
//   model->control_rngs_->y = 5;

  // Set up the cost parameters , horizon 100
  DubinsQuadraticCostParams new_params;



  new_params.position_coeff = 50; // Coefficient for state
  new_params.yaw_coeff = 10;  // Coefficient for yaw

  // Set desired terminal state

  new_params.desired_terminal_state[0] = 5;   // Desired x position
  new_params.desired_terminal_state[1] = 5;    // Desired y position
  new_params.desired_terminal_state[2] = 1.57;  // Desired yaw
  cost->setParams(new_params);

  float dt = 0.02;
  int max_iter = 1;
  float lambda = 0.25;
  float alpha = 0.0;
  const int num_timesteps = 100;

  // Set up Gaussian Distribution
  auto sampler_params = SAMPLER_T::SAMPLING_PARAMS_T();
  for (int i = 0; i < DubinsDynamics::CONTROL_DIM; i++)

  {
    sampler_params.std_dev[i] = 1.0;
  }

  auto sampler = new SAMPLER_T(sampler_params);

  // Feedback Controller
  auto fb_controller = new DDPFeedback<DubinsDynamics, num_timesteps>(model, dt);

  auto DubinsController =

      new VanillaMPPIController<DubinsDynamics, DubinsQuadraticCost, DDPFeedback<DubinsDynamics, num_timesteps>,
                                num_timesteps, 2048>(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha);

  auto controller_params = DubinsController->getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 4, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 4, 1);
  DubinsController->setParams(controller_params);

  DubinsDynamics::state_array current_state = DubinsDynamics::state_array::Zero();
  DubinsDynamics::state_array next_state = DubinsDynamics::state_array::Zero();
  DubinsDynamics::output_array output = DubinsDynamics::output_array::Zero();

  int time_horizon = 5000;
  DubinsDynamics::state_array xdot = DubinsDynamics::state_array::Zero();
  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < time_horizon; ++i)
  {
    // Compute the control
    DubinsController->computeControl(current_state, 1);
    // Increment the state

    DubinsDynamics::control_array control;

    control = DubinsController->getControlSeq().block(0, 0, DubinsDynamics::CONTROL_DIM, 1);
    // model->enforceConstraints(current_state, control);

    model->step(current_state, next_state, xdot, control, output, i, dt);
    current_state = next_state;
    if (i % 50 == 0)

    {

      printf("Current Time: %f    ", i * dt);
      printf("Current Baseline Cost: %f    ", DubinsController->getBaselineCost());
      // model->printState(current_state.data());
      std::cout << control << std::endl;

    }
    // Slide the controls down before calling the optimizer again
    DubinsController->slideControlSequence(1);

  }

  auto time_end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration<double, std::milli>(time_end - time_start);
  printf("The elapsed time is: %f milliseconds\n", diff.count());
  // //    std::cout << "The current control at timestep " << i << " is: " << CartpoleController.get_control_seq()[i] <<
  // //    std::endl;
  // cost->freeCudaMem();

  delete (DubinsController);
  delete (cost);
  delete (model);
  delete (fb_controller);
  delete sampler;

  return 0;

}