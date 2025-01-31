#include <memory>

#include "genlocov2_controllers/Genlocov2Controllers.h"

namespace legged {
vector_t Genlocov2Controllers::playModel(const vector_t& observations) const {
  std::cerr << "Genlocov2Controllers::playModel" << std::endl;
  return OnnxController::playModel(observations);
}
}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::Genlocov2Controllers, controller_interface::ControllerInterface)
