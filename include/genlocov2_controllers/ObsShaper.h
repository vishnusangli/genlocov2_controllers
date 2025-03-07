#pragma once

#include "legged_rl_controllers/OnnxController.h"
#include <deque>

namespace legged {

struct DataBlock
{
    vector_t quaternion;
    vector_t jointPos;
    vector_t actions;
};


class ObsShaper {
    private:
        std::deque<DataBlock> historyBuffer;

    public:
        DataBlock resetValues;
        ObsShaper(); // = default;
        ~ObsShaper() = default;

        size_t numDOF;
        size_t tokenSize;
        size_t historySize;

        void resetHistoryBuffer();
        void insertObs(const quaternion_t& quaternion, const vector_t& jointPos, const vector_t& actions);
        void exportObs(vector_t& policy_obs);
        void exportFlatObs(vector_t& flat_policy_obs);
        void exportFlatObsSep(vector_t& global_obs, vector_t& dof_obs);
};
}