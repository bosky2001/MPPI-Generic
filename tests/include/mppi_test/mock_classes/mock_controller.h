//
// Created by jason on 4/14/20.
//

#ifndef MPPIGENERIC_MOCK_CONTROLLER_H
#define MPPIGENERIC_MOCK_CONTROLLER_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mppi/controllers/controller.cuh>
#include <mppi_test/mock_classes/mock_dynamics.h>
#include <mppi_test/mock_classes/mock_costs.h>

// ===== mock controller ====
class MockController : public Controller<MockDynamics, MockCost, 100, 500, 32, 2> {
public:
  MOCK_METHOD0(resetControls, void());
  MOCK_METHOD(void, computeFeedbackGains, (const Eigen::Ref<const state_array>& state), (override));
  MOCK_METHOD1(slideControlSequence, void(int stride));
  MOCK_METHOD5(getCurrentControl, control_array(state_array&, double, state_array, control_trajectory, feedback_gain_trajectory));
  MOCK_METHOD2(computeControl, void(const Eigen::Ref<const state_array>& state, int optimization_stride));
  MOCK_METHOD0(getControlSeq, control_trajectory());
  MOCK_METHOD0(getStateSeq, state_trajectory());
  MOCK_METHOD0(getFeedbackGains, feedback_gain_trajectory());
  MOCK_METHOD1(updateImportanceSampler, void(const Eigen::Ref<const control_trajectory>& nominal_control));
  MOCK_METHOD0(allocateCUDAMemory, void());
  MOCK_METHOD0(computeFeedbackPropagatedStateSeq, void());
//  MOCK_METHOD0(calculateSampledTrajectories, void());
  MOCK_METHOD0(getSampledStateTrajectories, std::vector<state_trajectory>());
  MOCK_METHOD2(setPercentageSampledControlTrajectories, void(float new_perc, int multiplier));
  MOCK_METHOD0(calculateSampledStateTrajectories, void());
};
#endif //MPPIGENERIC_MOCK_CONTROLLER_H
