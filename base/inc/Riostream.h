// @(#)root/base:$Id$
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Riostream
#define ROOT_Riostream


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Riostream                                                            //
//                                                                      //
// This headers is only supposed to be used in implementation files.    //
// Never in headers, since it has "using namespace std".                //
// In headers use the companion Riosfwd.h.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#if defined(R__ANSISTREAM)
#   include <fstream>
#   include <iostream>
#   include <iomanip>
    using namespace std;
#else
#   include <fstream.h>
#   include <iostream.h>
#   include <iomanip.h>
#endif

#if defined(_MSC_VER) && (_MSC_VER <= 1200)
static std::ostream& operator<<(std::ostream& os, __int64 i)
{
   char buf[20];
   sprintf(buf,"%I64d", i);
   os << buf;
   return os;
}

static std::ostream& operator<<(std::ostream& os, unsigned __int64 i)
{ return os << (__int64) i; }
#endif

#endif
