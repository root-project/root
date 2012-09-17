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

#include <cstddef>

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
   NSScopeGuard()
      : fNSObject(nil)
   {
   }

   explicit NSScopeGuard(NSObject *nsObject)
               : fNSObject(nsObject)
   {   
   }
   ~NSScopeGuard()
   {
      [fNSObject release];//nothing for nil.
   }
   
public:
   
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

   NSScopeGuard(const NSScopeGuard &rhs);
   NSScopeGuard &operator = (const NSScopeGuard &rhs);
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
   
private:
   NSAutoreleasePool *fPool;

   AutoreleasePool(const AutoreleasePool &rhs);
   AutoreleasePool &operator = (const AutoreleasePool &rhs);
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
              : fRef(0)
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
            : fRef(0)
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
      fRef = 0;
   }

private:
   RefType fRef;

   CFScopeGuard(const CFScopeGuard &rhs);
   CFScopeGuard &operator = (const CFScopeGuard &rhs);
};

///////////////////////////////////////////////////
//                                               //
// Scoped array - scope guard for array.         //
// Sometimes, I can not use std::vector,         //
// for example, data is allocated in TGCocoa     //
// and must be later freed in Objective-C code.  //
// To make the code exception-safe, I still      //
// have to care about memory, which is already   //
// allocated. Not to have all this explicit      //
// delete [] in error handlers (it's easy        //
// to forget!!!) - I have ScopedArray.           //
// One good day I'll delete this and use         //
// standard library.                             //
//                                               //
///////////////////////////////////////////////////

template<class T>
class ScopedArray {
public:
   explicit ScopedArray(T * p = 0)
      : fData(p)
   {
   }
   
   ~ScopedArray()
   {
      delete [] fData;
   }
   
   void Reset(T * p)
   {
      if (p != fData)
         delete [] fData;
      fData = p;
   }
   
   void Release()
   {
      fData = 0;
   }
   
   T &operator [] (std::ptrdiff_t index)const
   {
      return fData[index];
   }
   
   T *Get()const
   {
      return fData;
   }

private:
   T *fData;

   ScopedArray(const ScopedArray &rhs);
   ScopedArray &operator = (const ScopedArray &rhs);
};

}//Util
}//MacOSX
}//ROOT

#endif
