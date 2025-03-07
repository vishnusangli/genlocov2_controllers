#pragma once

#include "legged_rl_controllers/OnnxController.h"

namespace legged {
class ActionFilter {
    public:
        ActionFilter();
        ~ActionFilter() = default;
        
        void filterActions(const vector_t& pd_tars, vector_t& pd_tars_filtered);

        void resetFilter(const vector_t& defaultAngles);

        size_t numDOF;

    private:
        
        vector_t xhist;
        vector_t yhist;

        std::vector<double> a_coeffs;
        std::vector<double> b_coeffs;
};

}