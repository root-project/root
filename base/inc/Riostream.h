// @(#)root/base:$Name:  $:$Id: Riostream.h,v 1.1 2002/01/24 11:39:26 rdm Exp $
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
std::ostream& operator<<(std::ostream& os, __int64 i)
{
   char buf[20];
   sprintf(buf,"%I64d", i);
   os << buf;
   return os;
}

std::ostream& operator<<(std::ostream& os, unsigned __int64 i)
{ return os << (__int64) i; }
#endif

#endif
