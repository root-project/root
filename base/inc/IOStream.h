// @(#)root/base:$Name:$:$Id:$
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOStream
#define ROOT_IOStream

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

#endif
