// @(#)root/foundation:
// Author: Philippe Canal, April 2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_BitUtils
#define ROOT_BitUtils

#include <cassert>
#include <cstddef>

namespace ROOT {
namespace Internal {

/// Return true if \p align is a valid C++ alignment value: strictly positive
/// and a power of two.  This is the set of values accepted by
/// `::operator new[](n, std::align_val_t(align))`.
inline constexpr bool IsValidAlignment(std::size_t align) noexcept
{
   return align > 0 && (align & (align - 1)) == 0;
}

/// Round \p value up to the next multiple of \p align.
/// \p align must be a power of two (asserted at runtime in debug builds).
template <typename T>
inline constexpr T AlignUp(T value, T align) noexcept
{
   assert(IsValidAlignment(static_cast<std::size_t>(align))); // must be a power of two
   return (value + align - 1) & ~(align - 1);
}

} // namespace Internal
} // namespace ROOT

#endif // ROOT_BitUtils
