/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_Bswapcpy
#define ROOT_Bswapcpy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Bswapcpy                                                             //
//                                                                      //
// Initial version: Apr 22, 2000                                        //
//                                                                      //
// A set of inline byte swapping routines for arrays.                   //
//                                                                      //
// The bswapcpy16() and bswapcpy32() routines are used for packing      //
// arrays of basic types into a buffer in a byte swapped order. Use     //
// of asm and the `bswap' opcode (available on i486 and up) reduces     //
// byte swapping overhead on linux.                                     //
//                                                                      //
// Use of routines is similar to that of memcpy.                        //
//                                                                      //
// ATTENTION:                                                           //
//                                                                      //
//    n - is a number of array elements to be copied and byteswapped.   //
//        (It is not the number of bytes!)                              //
//                                                                      //
// For arrays of short type (2 bytes in size) use bswapcpy16().         //
// For arrays of of 4-byte types (int, float) use bswapcpy32().         //
//                                                                      //
//                                                                      //
// Author: Alexandre V. Vaniachine <AVVaniachine@lbl.gov>               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(__CINT__)
#include <sys/types.h>
#endif

extern inline void * bswapcpy16(void * to, const void * from, size_t n)
{
int d0, d1, d2, d3;
__asm__ __volatile__(
        "cld\n"
        "1:\tlodsw\n\t"
        "rorw $8, %%ax\n\t"
        "stosw\n\t"
        "loop 1b\n\t"
        :"=&c" (d0), "=&D" (d1), "=&S" (d2), "=&a" (d3)
        :"0" (n), "1" ((long) to),"2" ((long) from)
        :"memory");
return (to);
}

extern inline void * bswapcpy32(void * to, const void * from, size_t n)
{
int d0, d1, d2, d3;
__asm__ __volatile__(
        "cld\n"
        "1:\tlodsl\n\t"
#if !defined __i486__ && !defined __pentium__ && !defined __pentiumpro__ && \
    !defined __pentium4__ && !defined __x86_64__
        "rorw $8, %%ax\n\t"
        "rorl $16, %%eax\n\t"
        "rorw $8, %%ax\n\t"
#else
        "bswap %%eax\n\t"
#endif
        "stosl\n\t"
        "loop 1b\n\t"
        :"=&c" (d0), "=&D" (d1), "=&S" (d2), "=&a" (d3)
        :"0" (n), "1" ((long) to),"2" ((long) from)
        :"memory");
return (to);
}
#endif
