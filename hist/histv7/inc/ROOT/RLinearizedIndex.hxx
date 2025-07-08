#ifndef ROOT_RLinearizedIndex
#define ROOT_RLinearizedIndex

#include <cstddef>

namespace ROOT {
namespace Experimental {

/**
A linearized index that can be invalid.

For example, when an argument is outside the axis and underflow / overflow bins are disabled.
*/
struct RLinearizedIndex final {
   std::size_t index;
   bool valid;
};

} // namespace Experimental
} // namespace ROOT

#endif
