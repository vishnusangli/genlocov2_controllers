//
// Created by qiayuanl on 9/1/24.
//

#include <memory>

#include "legged_rl_controllers/OnnxController.h"
#include "genlocov2_controllers/Genlocov2Controllers.h"

namespace legged {

controller_interface::CallbackReturn Genlocov2Controllers::on_configure(const rclcpp_lifecycle::State& previous_state) {
  if (this->OnnxController::on_configure(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
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
  // printInputsOutputs(inputNames_, inputShapes_, outputNames_, outputShapes_);

  const size_t numJoints = leggedModel_->getLeggedModel()->getJointNames().size();

  jointNameInPolicy_ = get_node()->get_parameter("policy.joint_names").as_string_array();
  if (jointNameInPolicy_.size() != numJoints) {
    RCLCPP_ERROR(get_node()->get_logger(), "joint_names size is not equal to joint size.");
    return controller_interface::CallbackReturn::ERROR;
  }

  lastActions_.setZero(numJoints);
  command_.setZero();
  RCLCPP_INFO_STREAM(rclcpp::get_logger("OnnxController"), "Load Onnx model from" << policyPath << " successfully !");

  // Model Parameters
  int hist_size_cfg;
  get_node()->get_parameter("policy.history_size", hist_size_cfg);
  historySize = static_cast<size_t>(hist_size_cfg);
  tokenSize = 2 * (historySize + 1);

  //Observation Sizes
  numDOF = numJoints;
  globalObservationsSize_ = (numDOF * (historySize + 1)) + 6;
  // local xy (2), delta sinyaw cosyaw (2), base height total height (2)
  dofObservationsSize_ = numDOF * (historySize + 1); //TODO: Is it 1d or 2d?
  global_observations_.setZero(globalObservationsSize_);
  dof_observations_.setZero(dofObservationsSize_);
  
  // Joint Descriptions
  scalar_t jointDescriptionSize;
  get_node()->get_parameter("policy.num_joint_descriptions", jointDescriptionSize); 
  jointDescriptionsTensor.resize(jointDescriptionSize*numDOF);
  for (size_t i = 0; i < numDOF; ++i) {
    const auto jointName = leggedModel_->getLeggedModel()->getJointNames()[i];
    const auto jointDesc = get_node()->get_parameter("policy.joint_descriptions" + jointName).as_double_array();
    for (size_t j = 0; j < jointDescriptionSize; ++j) {
      jointDescriptionsTensor[i*jointDescriptionSize + j] = jointDesc[j];
    }
  }

  const auto axis_inv_cfg = get_node()->get_parameter("policy.axis_inversion").as_double_array();
  axisInversion.setZero(numDOF);
  for (size_t i = 0; i < numDOF; ++i) {
    axisInversion[i] = axis_inv_cfg[i];
  }

  // TODO: Get joint descriptions

  // Objects
  obsShaper.numDOF = numDOF;
  obsShaper.tokenSize = tokenSize;
  obsShaper.historySize = historySize;
  obsShaper.resetValues.quaternion.setZero(4);
  obsShaper.resetValues.quaternion[3] = 1;
  obsShaper.resetValues.jointPos.setZero(numDOF);
  obsShaper.resetValues.actions.setZero(numDOF);
  actionFilter.numDOF = numDOF;

  //Commands
  get_node()->get_parameter("base_height", base_height_);
  get_node()->get_parameter("total_height", total_height_);

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

controller_interface::CallbackReturn Genlocov2Controllers::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  firstUpdate_ = true;
  receivedVelocityMsg_.set(std::make_shared<Twist>());
  lastActions_.setZero();
  obsShaper.resetHistoryBuffer();
  actionFilter.resetFilter(defaultPosition_);
  desiredPosition_ = defaultPosition_;
  return controller_interface::CallbackReturn::SUCCESS;
}

vector_t Genlocov2Controllers::getObservations(const rclcpp::Time& time) {
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
  vector_t offset_jointPos(jointNameInPolicy_.size());
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    const size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
    jointPositionsInPolicy[i] = jointPositions[hardwareIndex] - defaultPosition_[hardwareIndex];
    jointVelocitiesInPolicy[i] = jointVelocities[hardwareIndex];
    offset_jointPos[i] = (jointPositions[hardwareIndex] - defaultPosition_[hardwareIndex]) * axisInversion[hardwareIndex];
  }

  obsShaper.insertObs(quat, offset_jointPos, lastActions_);
  obsShaper.exportFlatObsSep(global_observations_, dof_observations_);

  // Global Observations
  // quaternion obs (historySize * 4)
  // Cmds:
  // local vel (2)
  // sin yaw, cos yaw (2)
  // base height, total height (2)

  global_observations_[(historySize) * 4 + 0] = command_[0];
  global_observations_[(historySize) * 4 + 1] = command_[1];
  global_observations_[(historySize) * 4 + 2] = std::sin(command_[2]);
  global_observations_[(historySize) * 4 + 3] = std::cos(command_[2]);
  global_observations_[(historySize) * 4 + 4] = base_height_;
  global_observations_[(historySize) * 4 + 5] = total_height_;

  return global_observations_;
}

vector_t Genlocov2Controllers::playModel(const vector_t& observations) const {
  // clang-format on
  std::vector<tensor_element_t> global_observationTensor;
  for (const double i : global_observations_) {
    global_observationTensor.push_back(static_cast<tensor_element_t>(i));
  }

  std::vector<tensor_element_t> dof_observationTensor;
  for (const double i : dof_observations_) {
    dof_observationTensor.push_back(static_cast<tensor_element_t>(i));
  }
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, dof_observationTensor.data(), dof_observationTensor.size(),
                                                                   inputShapes_[0].data(), inputShapes_[0].size()));
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, global_observationTensor.data(), global_observationTensor.size(),
                                                                   inputShapes_[1].data(), inputShapes_[1].size()));
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, const_cast<tensor_element_t*>(jointDescriptionsTensor.data()), jointDescriptionsTensor.size(),
                                                                   inputShapes_[2].data(), inputShapes_[2].size()));
  // run inference
  const Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = sessionPtr_->Run(runOptions, inputNames_.data(), inputValues.data(), 1, outputNames_.data(), 1);

  vector_t actions(lastActions_.size());
  for (size_t i = 0; i < actions.size(); ++i) {
    actions[i] = outputValues[0].At<tensor_element_t>({0, i});
  }
  return actions;
}

controller_interface::return_type Genlocov2Controllers::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
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
    vector_t preFilteredPositions(lastActions_.size());

    if (actionType_ == "position_absolute") {
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        preFilteredPositions[hardwareIndex] = (lastActions_[i] * actionScale_ + defaultPosition_[hardwareIndex]) * axisInversion[hardwareIndex];
      }
    }
    actionFilter.filterActions(preFilteredPositions, desiredPosition_);
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
}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::OnnxController, controller_interface::ControllerInterface)
