// @(#)root/base:$Id$
// Author: Fons Rademakers   23/1/02

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
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
// Backward compatibility header, #includes fstream, iostream, iomanip. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
// Workaround for https://github.com/llvm/llvm-project/issues/138683
// Include <chrono> before <fstream> to ensure _FilesystemClock is defined
// Can be removed once the upstream issue is fixed.
#include <chrono>
#endif
#include <fstream>
#include <iostream>
#include <iomanip>

#endif
