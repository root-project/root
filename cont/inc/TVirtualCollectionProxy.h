// @(#)root/cont:$Name:  $:$Id: TVirtualCollectionProxy.h,v 1.1 2004/01/10 10:52:29 brun Exp $
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef Root_TVirtualCollectionProxy_h
#define Root_TVirtualCollectionProxy_h

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualCollectionProxy                                              //
//                                                                      //
// Virtual interface of a proxy object for a collection class           //
// In particular this is used to implement splitting, emulation,        //
// and TTreeFormula access to STL containers.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClass.h"
#include "TObjArray.h"
#include "TStreamer.h"
#include "TDataType.h"

class TVirtualCollectionProxy {
protected:
   TClass *fClass;

public:
   TVirtualCollectionProxy() : fClass(0) {};
   TVirtualCollectionProxy(TClass *cl) : fClass(cl) {};
  
   virtual TVirtualCollectionProxy* Generate() const = 0; // Returns an object of the actual CollectionProxy class
   virtual ~TVirtualCollectionProxy() {};

   virtual TClass   *GetCollectionClass() { return fClass; } // Return a pointer to the TClass representing the container

   virtual void     *New() const { return fClass==0?0:fClass->New(); }                 // Return a new container object
   virtual void     *New(void *arena) const { return fClass==0?0:fClass->New(arena); } // Execute the container constructor

   virtual UInt_t    Sizeof() const = 0; // Return the sizeof the collection object.

   virtual void      SetProxy(void *objstart) = 0; // Set the address of the container being proxied

   virtual void    **GetPtrArray() = 0;            // Return a contiguous array of pointer to the values in the container (used for splitting)

   virtual Bool_t    HasPointers() const = 0; // Return true if the content is of type 'pointer to'

   virtual TClass   *GetValueClass() = 0;     // Return a pointer to the TClass representing the content.
   virtual EDataType GetType() = 0;           // If the content is a simple numerical value, return its type (see TDataType)

   virtual void     *At(UInt_t idx) = 0;                       // Return the address of the value at index 'idx'
   virtual void      Clear(const char *opt = "") = 0;          // Clear the container
   virtual void      Resize(UInt_t n, Bool_t forceDelete) = 0; // Resize the container
   virtual UInt_t    Size() const = 0;                         // Return the current size of the container

   virtual void      Streamer(TBuffer &b) = 0;                 // Stream the proxied container
};

#endif // Root_TVirtualCollectionProxy_h
