#include "genlocov2_controllers/ObsShaper.h"
#include <algorithm>  // for std::fill

namespace legged {

ObsShaper::ObsShaper() 
    : historyBuffer(historySize, DataBlock())
{
    // Initialize resetValues with default values.
    // Assumes that quaternion_t has a setIdentity() method and vector_t has setZero().
    resetValues.quaternion.setIdentity();
    resetValues.jointPos.setZero();
    resetValues.actions.setZero();

    // Fill historyBuffer with the reset values.
    std::fill(historyBuffer.begin(), historyBuffer.end(), resetValues);
}

void ObsShaper::resetHistoryBuffer() {
    std::fill(historyBuffer.begin(), historyBuffer.end(), resetValues);
}

void ObsShaper::insertObs(const quaternion_t& quaternion, const vector_t& jointPos, const vector_t& actions) {
    // Remove the oldest observation and push the new one.
    vector_t quat_vec(4);
    quat_vec << quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w();

    historyBuffer.pop_front();
    historyBuffer.push_back(DataBlock{quat_vec, jointPos, actions});
}

void ObsShaper::exportObs(vector_t& policy_obs) {
    // Assumes policy_obs is an Eigen-style matrix with dimensions at least (numDOF+3) x tokenSize,
    // where the first historySize columns (i.e. 0..historySize-1) hold joint positions and quaternion parts,
    // and the next historySize columns (i.e. historySize..tokenSize-1) hold the actions and additional quaternion parts.
    for (size_t i = 0; i < historySize+1; i++) {
        const DataBlock& dataBlock = historyBuffer[i];
        // Write joint positions and actions.
        for (size_t j = 0; j < numDOF; j++) {
            policy_obs(j + 3, i) = static_cast<float>(dataBlock.jointPos(j));
            policy_obs(j + 3, i + historySize+1) = static_cast<float>(dataBlock.actions(j));
        }
        // Write quaternion components.
        // Note: The ordering here replicates the behavior from your original code.
        policy_obs(1, i)           = static_cast<float>(dataBlock.quaternion.w());
        policy_obs(1, i + historySize+1) = static_cast<float>(dataBlock.quaternion.x());
        policy_obs(2, i)           = static_cast<float>(dataBlock.quaternion.y());
        policy_obs(2, i + historySize+1) = static_cast<float>(dataBlock.quaternion.z());
    }
}

void ObsShaper::exportFlatObs(vector_t& flat_policy_obs) {
    // The flat_policy_obs vector is assumed to be sized as 
    // (numDOF + 3) * tokenSize, where tokenSize == 2 * historySize.
    for (size_t i = 0; i < historySize+1; i++) {
        const DataBlock& dataBlock = historyBuffer[i];
        for (size_t j = 0; j < numDOF; j++) {
            flat_policy_obs[j + 3 + i * (numDOF + 3)] = static_cast<float>(dataBlock.jointPos(j));
            flat_policy_obs[j + 3 + (i + historySize+1) * (numDOF + 3)] = static_cast<float>(dataBlock.actions(j));
        }
        flat_policy_obs[1 + i * (numDOF + 3)] = static_cast<float>(dataBlock.quaternion.w());
        flat_policy_obs[1 + (i + historySize+1) * (numDOF + 3)] = static_cast<float>(dataBlock.quaternion.x());
        flat_policy_obs[2 + i * (numDOF + 3)] = static_cast<float>(dataBlock.quaternion.y());
        flat_policy_obs[2 + (i + historySize+1) * (numDOF + 3)] = static_cast<float>(dataBlock.quaternion.z());
    }
}

void ObsShaper::exportFlatObsSep(vector_t& global_obs, vector_t& dof_obs) {
    // Assumes that:
    // - dof_obs is arranged with tokenSize entries per joint,
    // - global_obs holds quaternion-related data with two blocks of historySize each.
    for (size_t i = 0; i < historySize+1; i++) {
        const DataBlock& dataBlock = historyBuffer[i];
        for (size_t j = 0; j < numDOF; j++) {
            // Write joint positions into dof_obs.
            dof_obs[(tokenSize * j) + i] = static_cast<float>(dataBlock.jointPos(j));
            // For actions, skip the first row as in the original logic.
            if (i != 0) {
                dof_obs[(tokenSize * j) + (historySize + i)] = static_cast<float>(dataBlock.actions(j));
            }
        }
        // Write quaternion components into global_obs.
        global_obs[(tokenSize * 1) + i]           = static_cast<float>(dataBlock.quaternion.x());
        global_obs[(tokenSize * 1) + (historySize+1 + i)] = static_cast<float>(dataBlock.quaternion.y());
        global_obs[(tokenSize * 2) + i]           = static_cast<float>(dataBlock.quaternion.z());
        global_obs[(tokenSize * 2) + (historySize+1 + i)] = static_cast<float>(dataBlock.quaternion.w());
    }
}

}  // namespace legged
