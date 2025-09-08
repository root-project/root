/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RLinearizedIndex
#define ROOT_RLinearizedIndex

#include <cstddef>

namespace ROOT {
namespace Experimental {

/**
A linearized index that can be invalid.

For example, when an argument is outside the axis and underflow / overflow bins are disabled.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
struct RLinearizedIndex final {
   std::size_t fIndex = 0;
   bool fValid = false;
};

} // namespace Experimental
} // namespace ROOT

#endif
