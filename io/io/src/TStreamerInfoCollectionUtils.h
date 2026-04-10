// @(#)root/io:$Id$
// Author: Philippe Canal 04/2026

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStreamerInfoCollectionUtils
#define ROOT_TStreamerInfoCollectionUtils

#include "TBuffer.h"
#include "RtypesCore.h"
#include <cmath>    // std::signbit
#include <limits>   // std::numeric_limits

namespace TStreamerInfoUtils {

/// Read the number-of-objects count from the buffer using the encoding introduced
/// in TStreamerInfo version 11: when the high bit of the 32-bit word is set the
/// remaining 31 bits form the high part of a 64-bit count, and the following
/// UInt_t carries the low 32 bits.
///
/// \param b    The input buffer (operator>> must be available for Int_t/UInt_t).
/// \param vers The TStreamerInfo version that was used when the data was written.
/// \return     The number of objects encoded in the stream.
inline ULong64_t ReadCollectionSize(TBuffer &b, Version_t vers)
{
   Int_t nobjects;
   b >> nobjects;
   ULong64_t nobjects64;
   if (std::signbit(nobjects) && vers >= 11) {
      nobjects64 = (static_cast<ULong64_t>(nobjects) & 0x7fffffff) << 32;
      UInt_t nobjectsLow;
      b >> nobjectsLow;
      nobjects64 |= nobjectsLow;
   } else {
      nobjects64 = nobjects;
   }
   return nobjects64;
}

/// Write the number-of-objects count to the buffer using the encoding introduced
/// in TStreamerInfo version 11: counts that fit in a non-negative Int_t are written
/// as a single Int_t; larger counts set the high bit of the first Int_t (carrying
/// bits 32–62 of the count) and follow with a UInt_t carrying the low 32 bits.
///
/// \param b The output buffer (WriteInt/WriteUInt must be available).
/// \param n The number of objects to encode.
inline void WriteCollectionSize(TBuffer &b, ULong64_t n)
{
   if (n <= static_cast<ULong64_t>(std::numeric_limits<Int_t>::max())) {
      b.WriteInt(static_cast<Int_t>(n));
   } else {
      // Encode high 31 bits with the sign bit set, then the low 32 bits.
      Int_t high = static_cast<Int_t>((n >> 32) | 0x80000000u);
      b.WriteInt(high);
      b.WriteUInt(static_cast<UInt_t>(n & 0xffffffffu));
   }
}

} // namespace TStreamerInfoUtils

#endif // ROOT_TStreamerInfoCollectionUtils
