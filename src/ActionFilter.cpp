#include "legged_rl_controllers/ActionFilter.h"

namespace legged {
ActionFilter::ActionFilter() {
    // Initialize filter coefficients
    // These coefficients are alrady pre-computed as per
    // a lowpass butter filter with order 2, highcut 4, fs 1/0.03
    a_coeffs = {1.0000, -0.9824,  0.3477};
    b_coeffs = {0.0913, 0.1826, 0.0913};

    xhist.resize(2*NumDOF);
    yhist.resize(2*NumDOF);
}

void ActionFilter::filterActions(std::vector<double>& pd_tars, std::vector<double>& pd_tars_filtered) {
    // Implementation of filterActions
    // Example:
    for (size_t i = 0; i < pd_tars.size(); i++) {
        pd_tars_filtered[i] = (pd_tars[i]*b_coeffs[0]) + 
                                (xhist[i] * b_coeffs[1]) + 
                                (xhist[i+NumDOF] * b_coeffs[2]) - 
                                ((yhist[i] * a_coeffs[1]) + (yhist[i+NumDOF] * a_coeffs[2]));

        xhist[i+NumDOF] = xhist[i];
        yhist[i+NumDOF] = yhist[i];
        xhist[i] = pd_tars[i];
        yhist[i] = pd_tars_filtered[i];
    }
}

void ActionFilter::resetFilter(vector_t& defaultAngles) {
    // Implementation of resetFilter
    // Example:

    for (int i = 0; i < NumDOF; i++) {
        xhist[i] = static_cast<double>(defaultAngles[i]);
        xhist[NumDOF+i] = static_cast<double>(defaultAngles[i]);

        yhist[i] = static_cast<double>(defaultAngles[i]);
        yhist[NumDOF+i] = static_cast<double>(defaultAngles[i]);
    }
}


} // namespace legged
