// @(#)root/base:$Name:  $:$Id: Riostream.h,v 1.1 2002/01/23 17:46:06 rdm Exp $
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

#endif
