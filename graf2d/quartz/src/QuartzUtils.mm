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
CGStateGuard::CGStateGuard(MacOSX::Util::CFScopeGuard<CGContextRef> &ctx)
               : fCtx(ctx.Get())
{
   assert(fCtx != 0 && "CGStateGuard, ctx parameter is null");
   CGContextSaveGState(fCtx);
}

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

//Actually, this class does any work only if you disable anti-aliasing, by default I have it on
//and there is nothing to do for a guard if you want it on.

//______________________________________________________________________________
CGAAStateGuard::CGAAStateGuard(CGContextRef ctx, bool enable)
               : fCtx(ctx),
                 fEnable(enable)
{
   assert(ctx != 0 && "CGAAStateGuard, ctx parameter is null");

   if (!enable)
      CGContextSetAllowsAntialiasing(ctx, false);
}

//______________________________________________________________________________
CGAAStateGuard::~CGAAStateGuard()
{
   //Enable it back:
   if (!fEnable)
      CGContextSetAllowsAntialiasing(fCtx, true);
}

}//Quartz
}//ROOT
