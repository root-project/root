// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov 6/12/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "CocoaUtils.h"

namespace ROOT {
namespace MacOSX {
namespace Util {

//______________________________________________________________________________
AutoreleasePool::AutoreleasePool(bool delayCreation /* = false*/)
                  : fPool(delayCreation ? nil : [[NSAutoreleasePool alloc] init])
{
}

//______________________________________________________________________________
AutoreleasePool::~AutoreleasePool()
{
   [fPool release];
}

//______________________________________________________________________________
void AutoreleasePool::Reset()
{
   if (fPool)
      [fPool release];

   fPool = [[NSAutoreleasePool alloc] init];
}

}//Util
}//MacOSX
}//ROOT
