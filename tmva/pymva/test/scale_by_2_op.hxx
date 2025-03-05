#include <algorithm>
#include "ROOT/RSpan.hxx"

namespace Scale_by_2 {
void Compute(std::span<const float> input, std::span<float> output) {
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * 2;
    }
}
}
