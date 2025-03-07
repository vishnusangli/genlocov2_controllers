#pragma once

#include "legged_rl_controllers/OnnxController.h"

namespace legged {
class Genlocov2Controllers : public OnnxController {
 protected:
  vector_t playModel(const vector_t& observations) const override;

  // Model Parameters
  size_t numJoints;
  size_t historySize;
  size_t tokenSize;

  // Observation Sizes
  size_t globalObservationsSize_;
  size_t dofObservationsSize_;

  // Command Values
  scalar_t base_height;
  scalar_t total_height;
};

}  // namespace legged
