/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Byteswap
#define ROOT_Byteswap

/* Originally (mid-1990s), this file contained copy/pasted assembler from RH6.0's
 * version of <bits/byteswap.h>.  Hence, we keep a copy of the FSF copyright below.
 * I believe all the original code has been excised, perhaps with exception of the
 * R__bswap_constant_* functions.  To be on the safe side, we are keeping the
 * copyright below.
 *   -- Brian Bockelman, August 2018
 */

/* Copyright (C) 1997 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

#include <cstdint>

#ifndef R__USEASMSWAP
#if (defined(__linux) || defined(__APPLE__)) &&   \
    (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__))
# define R__USEASMSWAP
#endif

#if defined(_WIN32) && (_MSC_VER >= 1300)
# include <stdlib.h>
# pragma intrinsic(_byteswap_ushort,_byteswap_ulong,_byteswap_uint64)
# define R__USEASMSWAP
#endif
#endif /* R__USEASMSWAP */

/* Swap bytes in 16 bit value.  */
#define R__bswap_constant_16(x) \
     ((((x) >> 8) & 0xff) | (((x) & 0xff) << 8))

#if defined(R__USEASMSWAP)
# if defined(__GNUC__)
#  define R__bswap_16(x) __builtin_bswap16(x)
# elif defined(_MSC_VER)
#  define R__bswap_16(x) _byteswap_ushort(x)
# endif
#else
# define R__bswap_16(x) R__bswap_constant_16(x)
#endif

/* Swap bytes in 32 bit value.  */
#define R__bswap_constant_32(x) \
     ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |               \
      (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))

#if defined(R__USEASMSWAP)
# if defined(__GNUC__)
#  define R__bswap_32(x) __builtin_bswap32(x)
# elif defined(_MSC_VER)
#  define R__bswap_32(x) _byteswap_ulong(x)
# endif
#else
# define R__bswap_32(x) R__bswap_constant_32(x)
#endif

/* Swap bytes in 64 bit value.  */
static inline uint64_t R__bswap_constant_64(uint64_t x) {
   x = ((x & 0x00000000ffffffff) << 32) | ((x & 0xffffffff00000000) >> 32);
   x = ((x & 0x0000ffff0000ffff) << 16) | ((x & 0xffff0000ffff0000) >> 16);
   x = ((x & 0x00ff00ff00ff00ff) <<  8) | ((x & 0xff00ff00ff00ff00) >>  8);
   return x;
}

#if defined(R__USEASMSWAP)
# if defined(__GNUC__)
#  define R__bswap_64(x) __builtin_bswap64(x)
# elif defined(_MSC_VER)
#  define R__bswap_64(x) _byteswap_uint64(x)
# endif
#else
# define R__bswap_64(x) R__bswap_constant_64(x)
#endif


/* Return a value with all bytes in the 16 bit argument swapped.  */
#define Rbswap_16(x) R__bswap_16(x)

/* Return a value with all bytes in the 32 bit argument swapped.  */
#define Rbswap_32(x) R__bswap_32(x)

/* Return a value with all bytes in the 64 bit argument swapped.  */
#define Rbswap_64(x) R__bswap_64(x)

/// \brief Helper templated class for swapping bytes; specializations for `N={2,4,8}`
/// are provided below.  This class can be used to byteswap any other type, e.g. in a
/// templated function (see example below).
/// ```
/// template <typename T>
/// void byteswap_arg(T &x) {
///    using value_type = typename RByteSwap<sizeof(T)>::value_type;
///    x = RByteSwap<sizeof(T)>::bswap(reinterpret_cast<value_type>(x));
/// }
/// ```
template <unsigned N>
struct RByteSwap {
};

template <>
struct RByteSwap<2> {
   // Signed integers can be safely byteswapped if they are reinterpret_cast'ed to unsigned
   using value_type = std::uint16_t;
   static value_type bswap(value_type x) { return Rbswap_16(x); }
};

template <>
struct RByteSwap<4> {
   using value_type = std::uint32_t;
   static value_type bswap(value_type x) { return Rbswap_32(x); }
};

template <>
struct RByteSwap<8> {
   using value_type = std::uint64_t;
   static value_type bswap(value_type x) { return Rbswap_64(x); }
};

#endif /* Byteswap.h */
