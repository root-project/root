// @(#)root/foundation:
// Author: Philippe Canal, April 2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RAlignmentUtils
#define ROOT_RAlignmentUtils

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

} // namespace Internal
} // namespace ROOT

#endif // ROOT_RAlignmentUtils
