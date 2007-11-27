// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Hack proposed by Philippe.
//
// See also Module.mk where this file is put on rootcint's command
// line just in front of the LinkDef.h file.

#if defined(G__DICTIONARY) && defined(R__SOLARIS)
// Force the inclusion of rw/math.h
#include <limits>
// Work around interaction between a struct named exception in math.h,
// std::exception and the use of using namespace std;
#define exception std::exception
#endif
