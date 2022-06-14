// @(#)root/base:$Id$
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Match
#define ROOT_Match


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Declarations for regular expression routines.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(__CINT__)
#include <sys/types.h>
#endif

typedef unsigned short Pattern_t;

int         Makepat(const char*, Pattern_t*, int);
const char* Matchs(const char*, size_t len, const Pattern_t*,
                   const char**);

#endif
