// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#ifndef ROOT_ResourceManagement
#define ROOT_ResourceManagement

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Resource management                                                  //
//                                                                      //
// Set of classes to simplify and automate resource and memory          //
// management with Core Foundation, Core Text, Core Graphics etc.       //
// Apple has reference counting system, but it's a good old C,          //
// and you have to remember to release counters yourself.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFBase.h>

#include <CoreGraphics/CGContext.h>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace ROOT {
namespace iOS {
namespace Util {


class NonCopyable {
protected:
   NonCopyable(){};
private:
   NonCopyable(const NonCopyable &rhs);
   NonCopyable &operator = (const NonCopyable &rhs);
};

//Class calls user's function to release resource.
template<class RefType, void (*release)(RefType)>
class RefGuardGeneric : NonCopyable {
public:
   explicit RefGuardGeneric(RefType ref) : fRef(ref), fActive(kTRUE)
   {
   }
   
   ~RefGuardGeneric()
   {
      if (fActive)
         release(fRef);
   }

   RefType Get()const
   {
      return fRef;
   }
   
   RefType Release()
   {
      fActive = kFALSE;
      return fRef;
   }

private:

   RefType fRef;
   Bool_t fActive;
};

//Simple class to call CFRelease on some CFTypeRef.
template<class RefType>
class RefGuard : NonCopyable {
public:
   explicit RefGuard(RefType ref) : fRef(ref), fActive(kTRUE)
   {
   }
   
   ~RefGuard()
   {
      if (fActive)
         CFRelease(fRef);
   }

   RefType Get()const
   {
      return fRef;
   }
   
   RefType Release()
   {
      fActive = kFALSE;
      return fRef;
   }

private:

   RefType fRef;
   bool fActive;
};


//Very similar to RefGuardGeneric, but calls CFRelease
class CFStringGuard : NonCopyable {
public:
   CFStringGuard(const char *text);
   ~CFStringGuard();

   CFStringRef Get()const;
private:
   CFStringRef fCFString;
};

//Save and restore CGContextRef's state.
class CGStateGuard : NonCopyable {
public:
   CGStateGuard(CGContextRef ctx);
   ~CGStateGuard();
   
private:
   CGContextRef fCtx;
};

//Same as RefGuardGeneric, but can
//be copied.
template<class RefType, void (*release)(RefType)>
class SmartRef {
public:
   SmartRef(RefType ref, bool initRetain = kFALSE)
      : fRef(ref)
   {
      if (initRetain)
         CFRetain(fRef);
   }
   
   SmartRef(const SmartRef &rhs)
      : fRef(rhs.fRef)
   {
      CFRetain(fRef);
   }
   
   SmartRef &operator = (const SmartRef &rhs)
   {
      if (fRef != rhs.fRef) {
         release(fRef);
         fRef = rhs.fRef;
         CFRetain(fRef);
      }
      
      return *this;
   }
   
   SmartRef &operator = (RefType ref)
   {
      if (fRef != ref) {
         release(fRef);
         fRef = ref;
         CFRetain(fRef);
      }
      
      return *this;
   }
   
   ~SmartRef()
   {
      release(fRef);
   }
   
   RefType Get()const
   {
      return fRef;
   }
private:
   RefType fRef;
};

}//namespace Util
}//namespace iOS
}//namespace ROOT

#endif
