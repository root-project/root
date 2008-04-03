// @(#)root/base:$Id$
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Riosfwd
#define ROOT_Riosfwd


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Riosfwd                                                              //
//                                                                      //
// This headers is only supposed to be used in header files.            //
// Never in sources, in source files use the companion Riostream.h.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#if defined(R__ANSISTREAM)
#   if defined(R__TMPLTSTREAM)
#      include <iostream>
#   else
#      include <iosfwd>
#   endif
using std::istream;
using std::ostream;
using std::fstream;
using std::ifstream;
using std::ofstream;
#else
class istream;
class ostream;
class fstream;
class ifstream;
class ofstream;
#endif

#endif
