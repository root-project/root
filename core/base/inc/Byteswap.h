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
    (defined(__GNUC__) && __GNUC__ >= 2)
#ifndef R__USEASMSWAP
#define R__USEASMSWAP
#endif
#endif

/* Get the machine specific, optimized definitions.  */
/* The following is copied from <bits/byteswap.h> (only from RH6.0 and above) */

/* Swap bytes in 16 bit value.  */
#define R__bswap_constant_16(x) \
     ((((x) >> 8) & 0xff) | (((x) & 0xff) << 8))

#if defined R__USEASMSWAP
# define R__bswap_16(x) \
     (__extension__                                                           \
      ({ unsigned short int __v;                                              \
         if (__builtin_constant_p (x))                                        \
           __v = R__bswap_constant_16 (x);                                    \
         else                                                                 \
           __asm__ __volatile__ ("rorw $8, %w0"                               \
                                 : "=r" (__v)                                 \
                                 : "0" ((unsigned short int) (x))             \
                                 : "cc");                                     \
         __v; }))
#else
/* This is better than nothing.  */
# define R__bswap_16(x) R__bswap_constant_16 (x)
#endif


/* Swap bytes in 32 bit value.  */
#define R__bswap_constant_32(x) \
     ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |               \
      (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))

#if defined R__USEASMSWAP
/* To swap the bytes in a word the i486 processors and up provide the
   `bswap' opcode.  On i386 we have to use three instructions.  */
# if !defined __i486__ && !defined __pentium__ && !defined __pentiumpro__ &&  \
     !defined __pentium4__ && !defined __x86_64__
#  define R__bswap_32(x) \
     (__extension__                                                           \
      ({ unsigned int __v;                                                    \
         if (__builtin_constant_p (x))                                        \
           __v = R__bswap_constant_32 (x);                                    \
         else                                                                 \
           __asm__ __volatile__ ("rorw $8, %w0;"                              \
                                 "rorl $16, %0;"                              \
                                 "rorw $8, %w0"                               \
                                 : "=r" (__v)                                 \
                                 : "0" ((unsigned int) (x))                   \
                                 : "cc");                                     \
         __v; }))
# else
#  define R__bswap_32(x) \
     (__extension__                                                           \
      ({ unsigned int __v;                                                    \
         if (__builtin_constant_p (x))                                        \
           __v = R__bswap_constant_32 (x);                                    \
         else                                                                 \
           __asm__ __volatile__ ("bswap %0"                                   \
                                 : "=r" (__v)                                 \
                                 : "0" ((unsigned int) (x)));                 \
         __v; }))
# endif
#else
# define R__bswap_32(x) R__bswap_constant_32 (x)
#endif


#if defined __GNUC__ && __GNUC__ >= 2
/* Swap bytes in 64 bit value.  */
# define R__bswap_64(x) \
     (__extension__                                                           \
      ({ union { __extension__ unsigned long long int __ll;                   \
                 UInt_t __l[2]; } __w, __r;                                   \
         __w.__ll = (x);                                                      \
         __r.__l[0] = R__bswap_32 (__w.__l[1]);                               \
         __r.__l[1] = R__bswap_32 (__w.__l[0]);                               \
         __r.__ll; }))
#endif /* bits/byteswap.h */


/* The following definitions must all be macros since otherwise some
   of the possible optimizations are not possible.  */

/* Return a value with all bytes in the 16 bit argument swapped.  */
#define Rbswap_16(x) R__bswap_16 (x)

/* Return a value with all bytes in the 32 bit argument swapped.  */
#define Rbswap_32(x) R__bswap_32 (x)

#if defined __GNUC__ && __GNUC__ >= 2
/* Return a value with all bytes in the 64 bit argument swapped.  */
# define Rbswap_64(x) R__bswap_64 (x)
#endif

#endif /* Byteswap.h */
