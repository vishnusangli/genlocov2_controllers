#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <realtime_tools/realtime_box.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "legged_rl_controllers/OnnxController.h"
#include "genlocov2_controllers/ObsShaper.h"
#include "genlocov2_controllers/ActionFilter.h"

namespace legged {
class Genlocov2Controllers : public OnnxController {
 protected:
  vector_t playModel(const vector_t& observations) const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  vector_t getObservations(const rclcpp::Time& time) override;

  // Model Parameters & Objects
  size_t numDOF{12};
  size_t historySize{15};
  size_t tokenSize{32};
  vector_t axisInversion;

  ObsShaper obsShaper;
  ActionFilter actionFilter;

  // Observations
  size_t globalObservationsSize_{0};
  size_t dofObservationsSize_{0};
  vector_t global_observations_;
  vector_t dof_observations_;


  std::vector<float> jointDescriptionsTensor;


  // Command Values
  scalar_t base_height_;
  scalar_t total_height_;
};

}  // namespace legged
