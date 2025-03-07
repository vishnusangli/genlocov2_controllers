//
// Created by qiayuanl on 9/1/24.
//

#include <memory>

#include "legged_rl_controllers/OnnxController.h"

namespace legged {

controller_interface::return_type OnnxController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (ControllerBase::update(time, period) != controller_interface::return_type::OK) {
    return controller_interface::return_type::ERROR;
  }

  std::shared_ptr<Twist> lastCommandMsg;
  receivedVelocityMsg_.get(lastCommandMsg);
  if (time - lastCommandMsg->header.stamp > std::chrono::milliseconds{static_cast<int>(0.5 * 1000.0)}) {
    lastCommandMsg->twist.linear.x = 0.0;
    lastCommandMsg->twist.linear.y = 0.0;
    lastCommandMsg->twist.angular.z = 0.0;
  }
  command_ << lastCommandMsg->twist.linear.x, lastCommandMsg->twist.linear.y, lastCommandMsg->twist.angular.z;

  if (firstUpdate_ || (time - lastPlayTime_).seconds() >= 1. / policyFrequency_) {
    const vector_t observations = getObservations(time);
    lastActions_ = playModel(observations);

    if (actionType_ == "position_absolute") {
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] = lastActions_[i] * actionScale_ + defaultPosition_[hardwareIndex];
      }
    } else if (actionType_ == "position_relative") {
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] += lastActions_[i] * actionScale_;
      }
    } else if (actionType_ == "position_delta") {
      const vector_t currentPosition = leggedModel_->getLeggedModel()->getGeneralizedPosition().tail(lastActions_.size());
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] = currentPosition[hardwareIndex] + lastActions_[i] * actionScale_;
      }
    }
    setPositions(desiredPosition_);

    firstUpdate_ = false;
    lastPlayTime_ = time;

    if (publisherRealtime_->trylock()) {
      auto& msg = publisherRealtime_->msg_;
      msg.data.clear();
      for (const double i : observations) {
        msg.data.push_back(i);
      }
      for (const double i : lastActions_) {
        msg.data.push_back(i);
      }
      publisherRealtime_->unlockAndPublish();
    }
  }

  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn OnnxController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_configure(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // Onnx
  std::string policyPath{};
  get_node()->get_parameter("policy.path", policyPath);
  get_node()->get_parameter("policy.frequency", policyFrequency_);
  onnxEnvPrt_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxController");
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  sessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyPath.c_str(), sessionOptions);
  inputNames_.clear();
  outputNames_.clear();
  inputShapes_.clear();
  outputShapes_.clear();
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < sessionPtr_->GetInputCount(); i++) {
    inputNames_.push_back(sessionPtr_->GetInputName(i, allocator));
    inputShapes_.push_back(sessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < sessionPtr_->GetOutputCount(); i++) {
    outputNames_.push_back(sessionPtr_->GetOutputName(i, allocator));
    outputShapes_.push_back(sessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  printInputsOutputs(inputNames_, inputShapes_, outputNames_, outputShapes_);

  numJoints = leggedModel_->getLeggedModel()->getJointNames().size();

  jointNameInPolicy_ = get_node()->get_parameter("policy.joint_names").as_string_array();
  if (jointNameInPolicy_.size() != numJoints) {
    RCLCPP_ERROR(get_node()->get_logger(), "joint_names size is not equal to joint size.");
    return controller_interface::CallbackReturn::ERROR;
  }

  lastActions_.setZero(numJoints);
  command_.setZero();
  RCLCPP_INFO_STREAM(rclcpp::get_logger("OnnxController"), "Load Onnx model from" << policyPath << " successfully !");

  // Observation
  observationNames_ = get_node()->get_parameter("policy.observations").as_string_array();
  observationSize_ = 0;
  for (const auto& name : observationNames_) {
    if (name == "base_lin_vel") {
      observationSize_ += 3;
    } else if (name == "base_ang_vel") {
      observationSize_ += 3;
    } else if (name == "projected_gravity") {
      observationSize_ += 3;
    } else if (name == "joint_positions") {
      observationSize_ += numJoints;
    } else if (name == "joint_velocities") {
      observationSize_ += numJoints;
    } else if (name == "last_action") {
      observationSize_ += numJoints;
    } else if (name == "phase") {
      observationSize_ += 4;
    } else if (name == "command") {
      observationSize_ += 3;
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Unknown observation name: %s", name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  phase_.setZero(2);
  phase_[1] = M_PI;

  // Action
  get_node()->get_parameter("policy.action_scale", actionScale_);
  get_node()->get_parameter("policy.action_type", actionType_);
  if (actionType_ != "position_absolute" && actionType_ != "position_relative" && actionType_ != "position_delta") {
    RCLCPP_ERROR(get_node()->get_logger(), "Unknown action type: %s", actionType_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  desiredPosition_.setZero(numJoints);

  // ROS Interface
  velocitySubscriber_ =
      get_node()->create_subscription<Twist>("/cmd_vel", rclcpp::SystemDefaultsQoS(), [this](const std::shared_ptr<Twist> msg) -> void {
        if ((msg->header.stamp.sec == 0) && (msg->header.stamp.nanosec == 0)) {
          RCLCPP_WARN_ONCE(get_node()->get_logger(),
                           "Received TwistStamped with zero timestamp, setting it to current "
                           "time, this message will only be shown once");
          msg->header.stamp = get_node()->get_clock()->now();
        }
        receivedVelocityMsg_.set(msg);
      });

  publisher_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/policy_io", rclcpp::SystemDefaultsQoS());
  publisherRealtime_ = std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(publisher_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn OnnxController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  firstUpdate_ = true;
  receivedVelocityMsg_.set(std::make_shared<Twist>());
  lastActions_.setZero();
  desiredPosition_ = defaultPosition_;
  return controller_interface::CallbackReturn::SUCCESS;
}

vector_t OnnxController::getObservations(const rclcpp::Time& time) {
  const auto leggedModel = leggedModel_->getLeggedModel();
  const auto q = leggedModel->getGeneralizedPosition();
  const auto v = leggedModel->getGeneralizedVelocity();

  const quaternion_t quat(q.segment<4>(3));
  const matrix_t inverseRot = quat.toRotationMatrix().transpose();
  const vector3_t baseLinVel = inverseRot * v.segment<3>(0);
  const auto& angVelArray = imu_->get_angular_velocity();
  const vector3_t baseAngVel(angVelArray[0], angVelArray[1], angVelArray[2]);
  const vector3_t gravityVector(0, 0, -1);
  const vector3_t projectedGravity(inverseRot * gravityVector);

  const vector_t jointPositions = q.tail(lastActions_.size());
  const vector_t jointVelocities = v.tail(lastActions_.size());

  vector_t jointPositionsInPolicy(jointNameInPolicy_.size());
  vector_t jointVelocitiesInPolicy(jointNameInPolicy_.size());
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    const size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
    jointPositionsInPolicy[i] = jointPositions[hardwareIndex] - defaultPosition_[hardwareIndex];
    jointVelocitiesInPolicy[i] = jointVelocities[hardwareIndex];
  }


  vector_t global_observations(70);
  vector_t dof_observations(70);

  ObsShaper_.insertObs(quat, jointPositions, lastActions_);
  ObsShaper_.exportFlatObsSep(global_observations, dof_observations);

  // Global Observations
  // quaternion obs (HistorySize * 4)
  // Cmds:
  // local vel (2)
  // sin yaw, cos yaw (2)
  // base height, total height (2)
  int HistorySize = 16;

  // global_observations[(HistorySize) * 4 + 0] = command_.x_vel;
  // global_observations[(HistorySize) * 4 + 1] = command_.y_vel;
  // global_observations[(HistorySize) * 4 + 2] = std::sin(command_.yaw_rate);
  // global_observations[(HistorySize) * 4 + 3] = std::cos(command_.yaw_rate);
  // global_observations[(HistorySize) * 4 + 4] = 0.27;
  // global_observations[(HistorySize) * 4 + 5] = 0.27;

  // vector_t observations(observationSize_);
  // size_t index = 0;
  // for (const auto name : observationNames_) {
  //   if (name == "base_lin_vel") {
  //     observations.segment<3>(index) = baseLinVel;
  //     index += 3;
  //   } else if (name == "base_ang_vel") {
  //     observations.segment<3>(index) = baseAngVel;
  //     index += 3;
  //   } else if (name == "projected_gravity") {
  //     observations.segment<3>(index) = projectedGravity;
  //     index += 3;
  //   } else if (name == "joint_positions") {
  //     observations.segment(index, jointPositionsInPolicy.size()) = jointPositionsInPolicy;
  //     index += jointPositionsInPolicy.size();
  //   } else if (name == "joint_velocities") {
  //     observations.segment(index, jointVelocitiesInPolicy.size()) = jointVelocitiesInPolicy;
  //     index += jointVelocitiesInPolicy.size();
  //   } else if (name == "last_action") {
  //     observations.segment(index, lastActions_.size()) = lastActions_;
  //     index += lastActions_.size();
  //   } else if (name == "phase") {
  //     phase_[0] = std::fmod(phase_[0] + phaseDt_ + M_PI, 2 * M_PI) - M_PI;
  //     phase_[1] = std::fmod(phase_[1] + phaseDt_ + M_PI, 2 * M_PI) - M_PI;
  //     vector_t obs(4);
  //     vector_t phase(2);
  //     // if (command_.norm() > 0.01) {
  //     //   phase = phase_;
  //     // } else {
  //     //   phase = vector_t::Ones(2) * M_PI;
  //     // }
  //     phase = phase_;

  //     obs[0] = std::cos(phase[0]);
  //     obs[1] = std::cos(phase[1]);
  //     obs[2] = std::sin(phase[0]);
  //     obs[3] = std::sin(phase[1]);
  //     observations.segment(index, obs.size()) = obs;
  //     index += obs.size();
  //   } else if (name == "command") {
  //     observations.segment<3>(index) = command_;
  //     index += 3;
  //   }
  // }
  // return observations;
}

vector_t OnnxController::playModel(const vector_t& observations) const {
  // clang-format on
  std::vector<tensor_element_t> observationTensor;
  for (const double i : observations) {
    observationTensor.push_back(static_cast<tensor_element_t>(i));
  }
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observationTensor.data(), observationTensor.size(),
                                                                   inputShapes_[0].data(), inputShapes_[0].size()));
  // run inference
  const Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = sessionPtr_->Run(runOptions, inputNames_.data(), inputValues.data(), 1, outputNames_.data(), 1);

  vector_t actions(lastActions_.size());
  for (size_t i = 0; i < actions.size(); ++i) {
    actions[i] = outputValues[0].At<tensor_element_t>({0, i});
  }
  return actions;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::OnnxController, controller_interface::ControllerInterface)
