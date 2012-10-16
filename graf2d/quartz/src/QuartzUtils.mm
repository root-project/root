// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 11/06/2012

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cassert>

#include "QuartzUtils.h"

namespace ROOT {
namespace Quartz {

//______________________________________________________________________________
CGStateGuard::CGStateGuard(CGContextRef ctx)
               : fCtx(ctx)
{
   assert(ctx != 0 && "CGStateGuard, ctx parameter is null");
   CGContextSaveGState(ctx);
}

//______________________________________________________________________________
CGStateGuard::~CGStateGuard()
{
   CGContextRestoreGState(fCtx);
}

}//Quartz
}//ROOT
