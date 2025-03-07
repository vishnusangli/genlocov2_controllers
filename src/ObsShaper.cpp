#include "legged_rl_controllers/ObsShaper.h"
#include <iostream>

namespace legged {

#include <vector>
#include <string>
#include <cctype>
#include <stdexcept>

// Constructor implementation
ObsShaper::ObsShaper() 
    : historyBuffer(HistorySize, DataBlock()), // Initialize deque with 16 elements, all default-initialized DataBlock
      resetValues() 
{
    // Initialize default values for resetValues
    resetValues.quaternion.setIdentity(); // Default quaternion
    resetValues.jointPos.setZero();   // Default joint positions
    resetValues.actions.setZero();    // Default actions

    // Fill historyBuffer with resetValues
    std::fill(historyBuffer.begin(), historyBuffer.end(), resetValues);
    // call_count_ = 0;
}

void ObsShaper::resetHistoryBuffer() {
    // Implementation of resetHistoryBuffer
    std::fill(historyBuffer.begin(), historyBuffer.end(), resetValues);
}

void ObsShaper::insertObs(const vector_quat_t& quaternion, const vector_dof_t& jointPos, const vector_dof_t& actions) {
    // Implementation of insertObs
    // Example:
    // std::cout << "Quat w: " << quaternion.w() << ", "
    //         << "x: " << quaternion.x() << ", "
    //         << "y: " << quaternion.y() << ", "
    //         << "z: " << quaternion.z() << std::endl;
    historyBuffer.pop_front();
    historyBuffer.push_back(DataBlock{quaternion, jointPos, actions});
}

void ObsShaper::exportObs(vector_policy_obs_t& policy_obs) {
    // Implementation of exportObs

    for (size_t i = 0; i < HistorySize; i++) {
        // Example:
        const DataBlock& dataBlock = historyBuffer[i];
        for (size_t j = 0; j < NumDOF; j++) {
            policy_obs(j+3, i) = static_cast<float>(dataBlock.jointPos(j));
            policy_obs(j+3, i+16) = static_cast<float>(dataBlock.actions(j));
        }

        policy_obs(1, i) = static_cast<float>(dataBlock.quaternion.w());
        policy_obs(1, i+16) = static_cast<float>(dataBlock.quaternion.x());
        policy_obs(2, i) = static_cast<float>(dataBlock.quaternion.y());
        policy_obs(2, i+16) = static_cast<float>(dataBlock.quaternion.z());
    }
}

void ObsShaper::exportFlatObs(std::vector<float>& flat_policy_obs) {
    // Implementation of exportObs

    for (size_t i = 0; i < HistorySize; i++) {
        // Example:
        const DataBlock& dataBlock = historyBuffer[i];
        for (size_t j = 0; j < NumDOF; j++) {
            flat_policy_obs[j+3 + i*(NumDOF+3)] = static_cast<float>(dataBlock.jointPos(j));
            flat_policy_obs[j+3 + (i+16)*(NumDOF+3)] = static_cast<float>(dataBlock.actions(j));
        }

        flat_policy_obs[1 + i*(NumDOF+3)] = static_cast<float>(dataBlock.quaternion.w());
        flat_policy_obs[1 + (i+16)*(NumDOF+3)] = static_cast<float>(dataBlock.quaternion.x());
        flat_policy_obs[2 + i*(NumDOF+3)] = static_cast<float>(dataBlock.quaternion.y());
        flat_policy_obs[2 + (i+16)*(NumDOF+3)] = static_cast<float>(dataBlock.quaternion.z());

    }
}

void ObsShaper::exportFlatObsSep(std::vector<float>& global_obs, std::vector<float>& dof_obs) {
    // Implementation of exportObs
    for (size_t i = 0; i < HistorySize; i++) {
        // Example:
        const DataBlock& dataBlock = historyBuffer[i];
        for (size_t j = 0; j < NumDOF; j++) {
            dof_obs[(32*j) + (i)] = static_cast<float>(dataBlock.jointPos(j));
            if (i != 0) {
                dof_obs[(32*j) + (15+i)] = static_cast<float>(dataBlock.actions(j));
            }
        }

        global_obs[(32*1) + (i)] = static_cast<float>(dataBlock.quaternion.x());
        global_obs[(32*1) + (16+i)] = static_cast<float>(dataBlock.quaternion.y());
        global_obs[(32*2) + (i)] = static_cast<float>(dataBlock.quaternion.z());
        global_obs[(32*2) + (16+i)] = static_cast<float>(dataBlock.quaternion.w());
    }

}

}
