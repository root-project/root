// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>

#include "IOSResourceManagement.h"

namespace ROOT {
namespace iOS {
namespace Util {

////////////////////////////////////////////////////////////////////////////////

CFStringGuard::CFStringGuard(const char *text)
                  : fCFString(0)
{
   fCFString = CFStringCreateWithCString(kCFAllocatorDefault, text, kCFStringEncodingMacRoman);
   if (!fCFString)
      std::runtime_error("CFStringGuard: create failed");
}

////////////////////////////////////////////////////////////////////////////////

CFStringGuard::~CFStringGuard()
{
   CFRelease(fCFString);
}

////////////////////////////////////////////////////////////////////////////////

CFStringRef CFStringGuard::Get()const
{
   return fCFString;
}

////////////////////////////////////////////////////////////////////////////////

CGStateGuard::CGStateGuard(CGContextRef ctx)
               : fCtx(ctx)
{
   CGContextSaveGState(ctx);
}

////////////////////////////////////////////////////////////////////////////////

CGStateGuard::~CGStateGuard()
{
   CGContextRestoreGState(fCtx);
}


}//namespace Util
}//namespace iOS
}//namespace ROOT
