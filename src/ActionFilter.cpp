#include "genlocov2_controllers/ActionFilter.h"

namespace legged {

ActionFilter::ActionFilter() {
    // Initialize filter coefficients
    // These coefficients are pre-computed as per a lowpass Butterworth filter
    // with order 2, highcut 4, fs 1/0.03.
    a_coeffs = {1.0000, -0.9824,  0.3477};
    b_coeffs = {0.0913, 0.1826, 0.0913};

    // Initialize history vectors with zero values
    xhist = vector_t::Zero(2 * numDOF);
    yhist = vector_t::Zero(2 * numDOF);
}

void ActionFilter::filterActions(const vector_t& pd_tars, vector_t& pd_tars_filtered) {
    // Ensure the input vector has the expected size.
    if (pd_tars.size() != numDOF) {
        throw std::runtime_error("Input pd_tars size does not match numDOF.");
    }
    // Resize the filtered output if necessary.
    if (pd_tars_filtered.size() != numDOF) {
        pd_tars_filtered.resize(numDOF);
    }

    // Apply the filter for each degree of freedom.
    for (int i = 0; i < numDOF; i++) {
        pd_tars_filtered[i] = (pd_tars[i] * b_coeffs[0]) +
                              (xhist[i] * b_coeffs[1]) +
                              (xhist[i + numDOF] * b_coeffs[2]) -
                              ((yhist[i] * a_coeffs[1]) + (yhist[i + numDOF] * a_coeffs[2]));

        // Update the history: shift older samples.
        xhist[i + numDOF] = xhist[i];
        yhist[i + numDOF] = yhist[i];
        xhist[i] = pd_tars[i];
        yhist[i] = pd_tars_filtered[i];
    }
}

void ActionFilter::resetFilter(const vector_t& defaultAngles) {
    // Ensure the default angles vector has the correct size.
    if (defaultAngles.size() != numDOF) {
        throw std::runtime_error("defaultAngles size does not match numDOF.");
    }
    
    // Reset both histories to the default angles.
    for (int i = 0; i < numDOF; i++) {
        xhist[i] = defaultAngles[i];
        xhist[i + numDOF] = defaultAngles[i];

        yhist[i] = defaultAngles[i];
        yhist[i + numDOF] = defaultAngles[i];
    }
}

} // namespace legged