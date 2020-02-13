#include "mppi_core/mppi_controller.cuh"

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, NUM_TIMESTEPS, NUM_ROLLOUTS, BDIM_X, BDIM_Y>

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, float dt, int max_iter, float gamma,
                                   const control_array& control_variance, const control_trajectory& init_control_traj,
                                   cudaStream_t stream):
model_(model), cost_(cost), dt_(dt), num_iters_(max_iter), gamma_(gamma),
control_variance_(control_variance), nominal_control_(init_control_traj), stream_(stream) {

    // Create the random number generator
    createAndSeedCUDARandomNumberGen();

    // Bind the model and control to the given stream
    setCUDAStream(stream);

    // Call the GPU setup functions of the model and cost
    model_->GPUSetup();
    cost_->GPUSetup();


    // Allocate CUDA memory for the controller
    allocateCUDAMemory();

    // Copy the noise variance to the device
    copyControlVarianceToDevice();
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
VanillaMPPI::~VanillaMPPIController() {
    // Free the CUDA memory of every object
    model_->freeCudaMem();
    cost_->freeCudaMem();

    // Free the CUDA memory of the controller
    deallocateCUDAMemory();
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::computeControl(state_array state) {

    // Send the initial condition to the device
    HANDLE_ERROR( cudaMemcpyAsync(initial_state_d_, state.data(), DYN_T::STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));

    for (int opt_iter = 0; opt_iter < num_iters_; opt_iter++) {
        // Send the nominal control to the device
        copyNominalControlToDevice();

        //Generate noise data
        curandGenerateNormal(gen_, control_noise_d_, NUM_ROLLOUTS*NUM_TIMESTEPS*DYN_T::CONTROL_DIM, 0.0, 1.0);

        //Launch the rollout kernel
        mppi_common::launchRolloutKernel<DYN_T, COST_T, NUM_ROLLOUTS, BDIM_X, BDIM_Y>(model_, cost_, dt_, NUM_TIMESTEPS,
                initial_state_d_, nominal_control_d_, control_noise_d_, control_variance_d_, trajectory_costs_d_, stream_);

        // Copy the costs back to the host
        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_.data(), trajectory_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR( cudaStreamSynchronize(stream_) );

        float baseline = mppi_common::computeBaselineCost(trajectory_costs_.data(), NUM_ROLLOUTS);

        // Launch the norm exponential kernel
        mppi_common::launchNormExpKernel(NUM_ROLLOUTS, BDIM_X, trajectory_costs_d_, gamma_, baseline);
        HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_.data(), trajectory_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
        HANDLE_ERROR(cudaStreamSynchronize(stream_));

        // Compute the normalizer
        float normalizer = mppi_common::computeNormalizer(trajectory_costs_.data(), NUM_ROLLOUTS);

        // Compute the cost weighted average //TODO SUM_STRIDE is BDIM_X, but should it be its own parameter?
//        mppi_common::launchWeightedReductionKernel<DYN_T, NUM_ROLLOUTS, BDIM_X>(trajectory_costs_d_, control_noise_d_, control_variance_d_, )


    }

}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::allocateCUDAMemory() {
    HANDLE_ERROR(cudaMalloc((void**)&nominal_control_d_, sizeof(float)*DYN_T::CONTROL_DIM*NUM_TIMESTEPS));
    HANDLE_ERROR(cudaMalloc((void**)&nominal_state_d_, sizeof(float)*DYN_T::STATE_DIM*NUM_TIMESTEPS));
    HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d_, sizeof(float)*NUM_ROLLOUTS));
    HANDLE_ERROR(cudaMalloc((void**)&control_variance_d_, sizeof(float)*DYN_T::CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&control_noise_d_, sizeof(float)*DYN_T::CONTROL_DIM*NUM_TIMESTEPS*NUM_ROLLOUTS));
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::deallocateCUDAMemory() {
    cudaFree(nominal_control_d_);
    cudaFree(nominal_state_d_);
    cudaFree(trajectory_costs_d_);
    cudaFree(control_variance_d_);
    cudaFree(control_noise_d_);
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void
VanillaMPPI::setCUDAStream(cudaStream_t stream) {
    stream_ = stream;
    model_->bindToStream(stream);
    cost_->bindToStream(stream);
    curandSetStream(gen_, stream); // requires the generator to be created!
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::updateControlNoiseVariance(const VanillaMPPIController::control_array &sigma_u) {
    control_variance_ = sigma_u;
    copyControlVarianceToDevice();
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::copyControlVarianceToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(control_variance_d_, control_variance_.data(), sizeof(float)*control_variance_.size(), cudaMemcpyHostToDevice, stream_));
    cudaStreamSynchronize(stream_);
}

template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
void VanillaMPPI::copyNominalControlToDevice() {
    HANDLE_ERROR(cudaMemcpyAsync(nominal_control_d_, nominal_control_.data(), sizeof(float)*nominal_control_.size(), cudaMemcpyHostToDevice, stream_));
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
}

//template<class DYN_T, class COST_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X, int BDIM_Y>
//void
//VanillaMPPI::computeNominalStateTrajectory(const state_array& x0) {
//    // Increment the system forward
//    for (int i = 0; i < DYN_T::STATE_DIM; i++) {
//        nominal_state_[i] = x0[i];
//    }
//    for (int i = 1; i < NUM_TIMESTEPS; i++) {
//        for (int j = 0; j < DYN_T::STATE_DIM; j++) {
//            nominal_state_[i*DYN_T::STATE_DIM + j] = model_
//        }
//
//    }
//
//}


#undef VanillaMPPI