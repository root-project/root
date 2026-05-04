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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

#ifdef _MSC_VER
#include <intrin.h> // for _BitScan*
#endif

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

/// Given an integer `x`, returns the number of leading 0-bits starting at the most significant bit position.
/// If `x` is 0, it returns the size of `x` in bits.
///
/// Example:
///
/// if x is a std::uint32_t with value 42 (0b0...0101010), then LeadingZeroes(x) == 26
template <typename T>
inline std::size_t LeadingZeroes(T x)
{
   constexpr std::size_t maxBits = sizeof(T) * 8;
   static_assert(std::is_integral_v<T> && (maxBits == 32 || maxBits == 64));

   if (x == 0)
      return maxBits;

#ifdef _MSC_VER
   unsigned long idx = 0;
   [[maybe_unused]] unsigned char nonZero;
   if constexpr (maxBits == 32) {
      nonZero = _BitScanReverse(&idx, x);
   } else {
#ifdef _WIN64
      // 64-bit machine
      nonZero = _BitScanReverse64(&idx, x);
#else
      // 32-bit machine
      std::uint32_t low = (x & 0xFFFF'FFFF);
      std::uint32_t high = (x >> 32) & 0xFFFF'FFFF;
      unsigned long lowIdx, highIdx;
      unsigned char lowNonZero = _BitScanReverse(&lowIdx, low);
      unsigned char highNonZero = _BitScanReverse(&highIdx, high);
      assert(lowNonZero | highNonZero);
      if (high == 0)
         idx = 63 - lowIdx;
      else
         idx = 31 - highIdx;
      return static_cast<std::size_t>(idx);

#endif // _WIN64
   }

   assert(nonZero);
   // NOTE: _BitScanReverse return the 0-based index of the leftmost non-zero bit.
   // To convert it to the number of zeroes we need to "flip" it from [0, maxBits) to [maxBits, 0)
   // (e.g. _BitScanReverse == 0  <=>  LeadingZeroes == maxBits)
   return static_cast<std::size_t>(maxBits - 1 - idx);
#else
   if constexpr (maxBits == 32) {
      return static_cast<std::size_t>(__builtin_clz(x));
   } else {
      return static_cast<std::size_t>(__builtin_clzl(x));
   }
#endif // _MSC_VER
}

/// Given an integer `x`, returns the number of trailing 0-bits starting at the least significant bit position.
/// If `x` is 0, it returns the size of `x` in bits.
///
/// Example:
///
/// if x is a std::uint32_t with value 42 (0b0...0101010), then TrailingZeroes(x) == 1
template <typename T>
inline std::size_t TrailingZeroes(T x)
{
   constexpr std::size_t maxBits = sizeof(T) * 8;
   static_assert(std::is_integral_v<T> && (maxBits == 32 || maxBits == 64));

   if (x == 0)
      return maxBits;

#ifdef _MSC_VER
   unsigned long idx = 0;
   [[maybe_unused]] unsigned char nonZero;
   if constexpr (maxBits == 32) {
      nonZero = _BitScanForward(&idx, x);
   } else {
#ifdef _WIN64
      // 64-bit machine
      nonZero = _BitScanForward64(&idx, x);
#else
      // 32-bit machine
      std::uint32_t low = (x & 0xFFFF'FFFF);
      std::uint32_t high = (x >> 32) & 0xFFFF'FFFF;
      unsigned long lowIdx, highIdx;
      unsigned char lowNonZero = _BitScanForward(&lowIdx, low);
      unsigned char highNonZero = _BitScanForward(&highIdx, high);
      nonZero = lowNonZero | highNonZero;
      if (low == 0)
         idx = highIdx + 32;
      else
         idx = lowIdx;

#endif // _WIN64
   }
   assert(nonZero);
   // Differently from LeadingZeroes, in this case the bit index returned by _BitScanForward is
   // already equivalent to the number of trailing zeroes, so we don't need any transformation.
   return static_cast<std::size_t>(idx);
#else
   if constexpr (maxBits == 32) {
      return static_cast<std::size_t>(__builtin_ctz(x));
   } else {
      return static_cast<std::size_t>(__builtin_ctzl(x));
   }
#endif // _MSC_VER
   }

} // namespace Internal
} // namespace ROOT

#endif // ROOT_BitUtils
