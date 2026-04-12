#pragma once

/**
 * @file test_fir_basic.hpp
 * @brief Shared FIR filter test coefficients for filters tests
 */

#include <vector>

namespace filters {
namespace tests {

// scipy.signal.firwin(64, 0.1, window='hamming')
static const std::vector<float> kTestFirCoeffs64 = {
  -0.000157f, -0.000332f, -0.000459f, -0.000399f, -0.000000f,
   0.000850f,  0.002169f,  0.003849f,  0.005627f,  0.007078f,
   0.007634f,  0.006647f,  0.003540f, -0.002061f, -0.010202f,
  -0.020558f, -0.032375f, -0.044476f, -0.055375f, -0.063450f,
  -0.067076f, -0.064848f, -0.055800f, -0.039588f, -0.016579f,
   0.012488f,  0.046355f,  0.083311f,  0.121273f,  0.157982f,
   0.191169f,  0.218728f,  0.238881f,  0.250291f,  0.252151f,
   0.244226f,  0.226902f,  0.201177f,  0.168596f,  0.131141f,
   0.091041f,  0.050605f,  0.012058f, -0.022485f, -0.051250f,
  -0.073004f, -0.087189f, -0.093864f, -0.093644f, -0.087573f,
  -0.076953f, -0.063233f, -0.047833f, -0.032032f, -0.016901f,
  -0.003260f,  0.008352f,  0.017614f,  0.024418f,  0.028835f,
   0.031065f,  0.031385f,  0.030094f,  0.027480f
};

}  // namespace tests
}  // namespace filters
