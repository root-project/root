// @(#)root/base

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstdint>
#include <cstring>

#ifndef ROOT_RFloat16
#define ROOT_RFloat16

/**
 * Conversion functions between full- and half-precision floats. The code used here is taken (with some modifications)
 * from the `half` C++ library (https://half.sourceforge.net/index.html), distributed under the MIT license.
 *
 * Original license:
 *
 * The MIT License
 *
 * Copyright (c) 2012-2021 Christian Rau
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef HALF_ENABLE_F16C_INTRINSICS
/// Enable F16C intruction set intrinsics.
/// Defining this to 1 enables the use of [F16C compiler intrinsics](https://en.wikipedia.org/wiki/F16C) for converting
/// between half-precision and single-precision values which may result in improved performance. This will not perform
/// additional checks for support of the F16C instruction set, so an appropriate target platform is required when
/// enabling this feature.
///
/// Unless predefined it will be enabled automatically when the `__F16C__` symbol is defined, which some compilers do on
/// supporting platforms.
#define HALF_ENABLE_F16C_INTRINSICS __F16C__
#endif
#if HALF_ENABLE_F16C_INTRINSICS
#include <immintrin.h>
#endif

namespace ROOT {
namespace Internal {
////////////////////////////////////////////////////////////////////////////////
/// \brief Get the half-precision overflow.
///
/// \param[in] value Half-precision value with sign bit only
///
/// \return Rounded overflowing half-precision value
constexpr std::uint16_t GetOverflowedValue(std::uint16_t value = 0)
{
   return (value | 0x7C00);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Round the given half-precision number to the nearest representable value.
///
/// \param[in] value The finite half-precision number to round
/// \param[in] guardBit The most significant discarded bit
/// \param[in] stickyBit Logical OR of all but the most significant discarded bits
///
/// \return The nearest-rounded half-precision value
constexpr std::uint16_t GetRoundedValue(std::uint16_t value, int guardBit, int stickyBit)
{
   return (value + (guardBit & (stickyBit | value)));
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Convert an IEEE single-precision float to half-precision.
///
/// Credit for this goes to [Jeroen van der Zijp](http://fox-toolkit.org/ftp/fasthalffloatconversion.pdf).
///
/// \param[in] value The single-precision value to convert
///
/// \return The converted half-precision value
inline std::uint16_t FloatToHalf(float value)
{
#if HALF_ENABLE_F16C_INTRINSICS
   return _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT));
#else
   std::uint32_t fbits;
   std::memcpy(&fbits, &value, sizeof(float));

   std::uint16_t sign = (fbits >> 16) & 0x8000;
   fbits &= 0x7FFFFFFF;
   if (fbits >= 0x7F800000)
      return sign | 0x7C00 | ((fbits > 0x7F800000) ? (0x200 | ((fbits >> 13) & 0x3FF)) : 0);
   if (fbits >= 0x47800000)
      return GetOverflowedValue(sign);
   if (fbits >= 0x38800000)
      return GetRoundedValue(sign | (((fbits >> 23) - 112) << 10) | ((fbits >> 13) & 0x3FF), (fbits >> 12) & 1,
                             (fbits & 0xFFF) != 0);
   if (fbits >= 0x33000000) {
      int i = 125 - (fbits >> 23);
      fbits = (fbits & 0x7FFFFF) | 0x800000;
      return GetRoundedValue(sign | (fbits >> (i + 1)), (fbits >> i) & 1,
                             (fbits & ((static_cast<std::uint32_t>(1) << i) - 1)) != 0);
   }

   return sign;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Convert an IEEE half-precision float to single-precision.
///
/// Credit for this goes to [Jeroen van der Zijp](http://fox-toolkit.org/ftp/fasthalffloatconversion.pdf).
///
/// \param[in] value The half-precision value to convert
///
/// \return The converted single-precision value
inline float HalfToFloat(std::uint16_t value)
{
#if HALF_ENABLE_F16C_INTRINSICS
   return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
#else
   std::uint32_t fbits = static_cast<std::uint32_t>(value & 0x8000) << 16;
   int abs = value & 0x7FFF;
   if (abs) {
      fbits |= 0x38000000 << static_cast<unsigned>(abs >= 0x7C00);
      for (; abs < 0x400; abs <<= 1, fbits -= 0x800000)
         ;
      fbits += static_cast<std::uint32_t>(abs) << 13;
   }
   float out;
   std::memcpy(&out, &fbits, sizeof(float));
   return out;
#endif
}
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RFloat16
