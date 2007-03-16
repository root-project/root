// @(#)root/cont:$Name:  $:$Id: TVirtualCollectionProxy.h,v 1.12 2007/02/18 14:56:42 brun Exp $
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef Root_TVirtualCollectionProxy
#define Root_TVirtualCollectionProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualCollectionProxy                                              //
//                                                                      //
// Virtual interface of a proxy object for a collection class           //
// In particular this is used to implement splitting, emulation,        //
// and TTreeFormula access to STL containers.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClassRef.h"
#include "TDataType.h"

class TClass;

class TVirtualCollectionProxy {
protected:
   TClassRef fClass;
   virtual void SetValueClass(TClass *newcl) = 0;
   friend class TClass;

public:
   class TPushPop {
      // Helper class that insures that push and pop are done when entering
      // and leaving a C++ context (even in the presence of exceptions)
   public:
      TVirtualCollectionProxy *fProxy;
      inline TPushPop(TVirtualCollectionProxy *proxy, 
         void *objectstart) : fProxy(proxy) { fProxy->PushProxy(objectstart); }
      inline ~TPushPop() { fProxy->PopProxy(); }
   };

   TVirtualCollectionProxy() : fClass() {};
   TVirtualCollectionProxy(TClass *cl) : fClass(cl) {};
  
   virtual TVirtualCollectionProxy* Generate() const = 0; // Returns an object of the actual CollectionProxy class
   virtual ~TVirtualCollectionProxy() {};
   virtual TClass   *GetCollectionClass() { return fClass; } // Return a pointer to the TClass representing the container

   virtual void     *New() const {                // Return a new container object
     return fClass.GetClass()==0 ? 0 : fClass->New();
   }
   virtual void     *New(void *arena) const {     // Execute the container constructor
     return fClass.GetClass()==0 ? 0 : fClass->New(arena);
   }

   virtual void     *NewArray(Int_t nElements) const {                // Return a new container object
     return fClass.GetClass()==0 ? 0 : fClass->NewArray(nElements);
   }
   virtual void     *NewArray(Int_t nElements, void *arena) const {     // Execute the container constructor
     return fClass.GetClass()==0 ? 0 : fClass->NewArray(nElements, arena);
   }

   virtual void      Destructor(void *p, Bool_t dtorOnly = kFALSE) { // Execute the container destructor
     TClass* cl = fClass.GetClass();
     if (cl) cl->Destructor(p, dtorOnly);
   }

   virtual void      DeleteArray(void *p, Bool_t dtorOnly = kFALSE) { // Execute the container array destructor
     TClass* cl = fClass.GetClass();
     if (cl) cl->DeleteArray(p, dtorOnly);
   }

   virtual UInt_t    Sizeof() const = 0; // Return the sizeof the collection object.

   virtual void      PushProxy(void *objectstart) = 0; // Set the address of the container being proxied and keep track of the previous one.
   virtual void      PopProxy() = 0;                   // Reset the address of the container being proxied to the previous container

   virtual Bool_t    HasPointers() const = 0; // Return true if the content is of type 'pointer to'

   virtual TClass   *GetValueClass() = 0;     // Return a pointer to the TClass representing the content.
   virtual EDataType GetType() = 0;           // If the content is a simple numerical value, return its type (see TDataType)
   virtual void     *At(UInt_t idx) = 0;                       // Return the address of the value at index 'idx'
   virtual void      Clear(const char *opt = "") = 0;          // Clear the container
   virtual UInt_t    Size() const = 0;                         // Return the current size of the container
   virtual void*     Allocate(UInt_t n, Bool_t forceDelete) = 0;
   virtual void      Commit(void*) = 0;
           char     *operator[](UInt_t idx) const { return (char*)(const_cast<TVirtualCollectionProxy*>(this))->At(idx); }
};

#endif
