// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov 6/12/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#ifndef ROOT_CocoaUtils
#define ROOT_CocoaUtils

#include <Foundation/Foundation.h>

namespace ROOT {
namespace MacOSX {
namespace Util {

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  NSStrongReference. Class to keep strong reference to NSObject. //
//  Move ctor and assignment operator are deleted.                 //
//                                                                 //
/////////////////////////////////////////////////////////////////////

//In principle, NS is a prefix for AppKit classes, 
//but I do not want to make it a suffix and
//still have to distinguish between RAII classes 
//for AppKit and for Core Foundation/Core Graphics (suffix CF).
//But in C++ I have namespaces, and I can have NSWhatIWant,
//since it will be ROOT::MacOSX::Util::NSWhatIWant. 
//The same is true for CFWhatIWant (CF is a prefix for
//CoreFoundation in Apple's API).


template<class DerivedType>
class NSStrongReference {
public:
   NSStrongReference()
      : fNSObject(nil)
   {
   }

   NSStrongReference(NSObject *nsObject)
      : fNSObject([nsObject retain])
   {
   }

   NSStrongReference(const NSStrongReference &rhs)
      : fNSObject([rhs.fNSObject retain])
   {
   }


   ~NSStrongReference()
   {
      [fNSObject release];
   }

   NSStrongReference &operator = (const NSStrongReference &rhs)
   {
      if (&rhs != this) {
         //Even if both reference the same NSObject, it's ok to do release.
         [fNSObject release];
         fNSObject = [rhs.fNSObject retain];
      }

      return *this;
   }
   
   NSStrongReference &operator = (NSObject *nsObject)
   {
      if (nsObject != fNSObject) {
         [fNSObject release];
         fNSObject = [nsObject retain];
      }
   
      return *this;
   }
   
   DerivedType *Get()const
   {
      return (DerivedType *)fNSObject;
   }
   
   void Reset(NSObject *object)
   {
      if (fNSObject != object) {
         NSObject *obj = [object retain];
         [fNSObject release];
         fNSObject = obj;
      }
   }

   //Declare as deleted for clarity.
   NSStrongReference &operator = (NSStrongReference &&rhs) = delete;
   NSStrongReference(NSStrongReference &&rhs) = delete;

private:
   NSObject *fNSObject;
};

///////////////////////////////////////////////////////////////////
//                                                               //
// NSScopeGuard. Copy/move operations are deleted.               //
//                                                               //
///////////////////////////////////////////////////////////////////

template<class DerivedType>
class NSScopeGuard {
public:
   explicit NSScopeGuard(NSObject *nsObject)
               : fNSObject(nsObject)
   {   
   }
   ~NSScopeGuard()
   {
      [fNSObject release];//nothing for nil.
   }
   
   NSScopeGuard(const NSScopeGuard &rhs) = delete;
   //Declare as deleted for clarity.
   NSScopeGuard(NSScopeGuard &&rhs) = delete;
   
   NSScopeGuard &operator = (const NSScopeGuard &rhs) = delete;
   //Declare as deleted for clarity.
   NSScopeGuard &operator = (NSScopeGuard &&rhs) = delete;
   
   DerivedType *Get()const
   {
      return (DerivedType *)fNSObject;
   }
   
   void Reset(NSObject *object)
   {
      if (object != fNSObject) {
         [fNSObject release];
         fNSObject = object;
      }
   }
   
   void Release()
   {
      fNSObject = nil;
   }
private:   
   NSObject *fNSObject;
};

//////////////////////////////////////
//                                  //
// RAII class for autorelease pool. //
//                                  //
//////////////////////////////////////

class AutoreleasePool {
public:
   AutoreleasePool();
   ~AutoreleasePool();
   

   AutoreleasePool(const AutoreleasePool &rhs) = delete;
   //Declare as deleted for clarity.
   AutoreleasePool(AutoreleasePool &&rhs) = delete;

   AutoreleasePool &operator = (const AutoreleasePool &rhs) = delete;
   //Declare as deleted for clarity.
   AutoreleasePool &operator = (AutoreleasePool &&rhs) = delete;
private:

   NSAutoreleasePool *fPool;
};

///////////////////////////////////////////////////////////
//                                                       //
// Strong reference for a Core Foundation object.        //
// This class can have specializations for CF object     //
// with it's own version of retain or release.           //
//                                                       //
///////////////////////////////////////////////////////////

template<class RefType>
class CFStrongReference {
public:
   CFStrongReference()
              : fRef(nullptr)
   {
   }
   
   CFStrongReference(RefType ref, bool initRetain)
              : fRef(ref)
   {
      if (initRetain && ref)
         CFRetain(ref);
   }
   
   CFStrongReference(const CFStrongReference &rhs)
   {
      fRef = rhs.fRef;
      if (fRef)
         CFRetain(fRef);
   }

   CFStrongReference &operator = (const CFStrongReference &rhs)
   {
      if (this != &rhs) {
         if (fRef)
            CFRelease(fRef);//Ok even if rhs references the same.
         fRef = rhs.fRef;
         if (fRef)
            CFRetain(fRef);
      }
      
      return *this;
   }
   
   ~CFStrongReference()
   {
      if (fRef)
         CFRelease(fRef);
   }
   
   RefType Get()const
   {
      return fRef;
   }
   
   //Declare as deleted for clarity.
   CFStrongReference(CFStrongReference &&rhs) = delete;
   CFStrongReference &operator = (CFStrongReference &&rhs) = delete;
   
private:
   RefType fRef;
};

///////////////////////////////////////////////////
//                                               //
// Scope guard for Core Foundations objects.     //
// Specializations can be defined to call        //
// something different from CFRetain/CFRelease,  //
// but no need, they usually differ by accepting //
// null pointer (CFRetain/CFRelease will cause   //
// an error.                                     //
//                                               //
///////////////////////////////////////////////////

template<class RefType>
class CFScopeGuard {
public:
   CFScopeGuard()
            : fRef(nullptr)
   {
   }
   
   explicit CFScopeGuard(RefType ref)
               : fRef(ref)
   {
   }
   
   ~CFScopeGuard()
   {
      if (fRef)
         CFRelease(fRef);
   }
   
   CFScopeGuard(const CFScopeGuard &rhs) = delete;
   CFScopeGuard(CFScopeGuard &&rhs) = delete;
   
   //Declare as delete for clarity.
   CFScopeGuard &operator = (const CFScopeGuard &rhs) = delete;
   CFScopeGuard &operator = (CFScopeGuard &&rhs) = delete;
   
   RefType Get()const
   {
      return fRef;
   }
   
   void Reset(RefType ref)
   {
      if (ref != fRef) {
         if (fRef)
            CFRelease(fRef);
         fRef = ref;
      }
   }
   
   void Release()
   {
      fRef = nullptr;
   }

private:
   RefType fRef;
};

}//Util
}//MacOSX
}//ROOT

#endif
