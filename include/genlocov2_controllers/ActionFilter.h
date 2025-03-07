#pragma once

#include "genlocov2_controllers/ObsShaper.h"


namespace legged {
class ActionFilter {
    public:
        ActionFilter();
        ~ActionFilter() = default;
        
        void filterActions(const vector_t& pd_tars, const vector_t& pd_tars_filtered);

        void resetFilter(const vector_t& defaultAngles);

        private:
        
        vector_t xhist;
        vector_t yhist;

        std::vector<double> a_coeffs;
        std::vector<double> b_coeffs;
};

}