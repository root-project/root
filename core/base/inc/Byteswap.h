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

#if (defined(__linux) || defined(__APPLE__)) && \
    (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__))
#ifndef R__USEASMSWAP
#define R__USEASMSWAP
#endif
#endif

/* Swap bytes in 16 bit value.  */
#define R__bswap_constant_16(x) \
     ((((x) >> 8) & 0xff) | (((x) & 0xff) << 8))

#if defined R__USEASMSWAP
# define R__bswap_16(x) __builtin_bswap16(x)
#else
# define R__bswap_16(x) R__bswap_constant_16 (x)
#endif


/* Swap bytes in 32 bit value.  */
#define R__bswap_constant_32(x) \
     ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |               \
      (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))

#if defined R__USEASMSWAP
# define R__bswap_32(x) __builtin_bswap32(x)
#else
# define R__bswap_32(x) R__bswap_constant_32 (x)
#endif


/* Return a value with all bytes in the 16 bit argument swapped.  */
#define Rbswap_16(x) R__bswap_16 (x)

/* Return a value with all bytes in the 32 bit argument swapped.  */
#define Rbswap_32(x) R__bswap_32 (x)

/* Return a value with all bytes in the 64 bit argument swapped.  */

/* For reasons that were lost to history, Rbswap_64 used to only
 * be defined wherever GNUC was available.  To simplify the macro
 * definitions, we extend this to wherever we can use the gcc-like
 * builtins.
 *    -- Brian Bockelman, August 2018
 */
#ifdef R__USEASMSWAP
# define Rbswap_64(x) __builtin_bswap64(x)
#endif

#endif /* Byteswap.h */
