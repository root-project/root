/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RLinearizedIndex
#define ROOT_RLinearizedIndex

#include <cstddef>
#include <cstdint>

namespace ROOT {
namespace Experimental {

/**
A linearized index that can be invalid.

For example, when an argument is outside the axis and underflow / overflow bins are disabled.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
struct RLinearizedIndex final {
   // We use std::uint64_t instead of std::size_t for the index because for sparse histograms, not all bins have to be
   // allocated in memory. However, we require that the index has at least that size.
   static_assert(sizeof(std::uint64_t) >= sizeof(std::size_t), "index type not large enough to address all bins");

   std::uint64_t fIndex = 0;
   bool fValid = false;
};

} // namespace Experimental
} // namespace ROOT

#endif
