#pragma once

#include <deque>

constexpr int NumDOF = 12;
constexpr int TokenSize = 32;
constexpr int HistorySize = 16;


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

        void resetHistoryBuffer();
        void insertObs(const quaternion_t& quaternion, const vector_t& jointPos, const vector_t& actions);
        void exportObs(vector_t& policy_obs);
        void exportFlatObs(const vector_t& flat_policy_obs);
        void exportFlatObsSep(const vector_t& global_obs, const vector_t& dof_obs);
};
}