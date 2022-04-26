// @(#)root/io:$Id$
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TEmulatedCollectionProxy
#define ROOT_TEmulatedCollectionProxy

#include "TGenCollectionProxy.h"

#include <vector>

class TEmulatedCollectionProxy : public TGenCollectionProxy  {

   // Friend declaration
   friend class TCollectionProxy;

public:
   // Container type definition
   typedef std::vector<char>  Cont_t;
   // Pointer to container type
   typedef Cont_t            *PCont_t;
protected:

   // Some hack to avoid const-ness
   TGenCollectionProxy* InitializeEx(Bool_t silent) override;

   // Object input streamer
   void ReadItems(int nElements, TBuffer &b);

   // Object output streamer
   void WriteItems(int nElements, TBuffer &b);

   // Shrink the container
   void Shrink(UInt_t nCurr, UInt_t left, Bool_t force);

   // Expand the container
   void Expand(UInt_t nCurr, UInt_t left);

private:
   TEmulatedCollectionProxy &operator=(const TEmulatedCollectionProxy &); // Not implemented.

public:
   // Virtual copy constructor
   TVirtualCollectionProxy* Generate() const override;

   // Copy constructor
   TEmulatedCollectionProxy(const TEmulatedCollectionProxy& copy);

   // Initializing constructor
   TEmulatedCollectionProxy(const char* cl_name, Bool_t silent);

   // Standard destructor
   virtual ~TEmulatedCollectionProxy();

   // Virtual constructor
   void* New() const override             {  return new Cont_t;         }

   // Virtual in-place constructor
   void* New(void* memory) const override {  return new(memory) Cont_t; }

   // Virtual constructor
   TClass::ObjectPtr NewObject() const override             {  return {new Cont_t, nullptr};         }

   // Virtual in-place constructor
   TClass::ObjectPtr NewObject(void* memory) const override {  return {new(memory) Cont_t, nullptr}; }

   // Virtual array constructor
   void* NewArray(Int_t nElements) const override             {  return new Cont_t[nElements]; }

   // Virtual in-place constructor
   void* NewArray(Int_t nElements, void* memory) const override {  return new(memory) Cont_t[nElements]; }

   // Virtual array constructor
   TClass::ObjectPtr NewObjectArray(Int_t nElements) const override  {  return {new Cont_t[nElements], nullptr}; }

   // Virtual in-place constructor
   TClass::ObjectPtr NewObjectArray(Int_t nElements, void* memory) const override {  return {new(memory) Cont_t[nElements], nullptr}; }

   // Virtual destructor
   void  Destructor(void* p, Bool_t dtorOnly = kFALSE) const override;

   // Virtual array destructor
   void  DeleteArray(void* p, Bool_t dtorOnly = kFALSE) const override;

   // TVirtualCollectionProxy overload: Return the sizeof the collection object.
   UInt_t Sizeof() const override { return sizeof(Cont_t); }

   // Return the address of the value at index 'idx'
   void *At(UInt_t idx) override;

   // Clear the container
   void Clear(const char *opt = "") override;

   // Resize the container
   void Resize(UInt_t n, Bool_t force_delete) override;

   // Return the current size of the container
   UInt_t Size() const override;

   // Block allocation of containees
   void* Allocate(UInt_t n, Bool_t forceDelete) override;

   // Block commit of containees
   void Commit(void* env) override;

   // Insert data into the container where data is a C-style array of the actual type contained in the collection
   // of the given size.   For associative container (map, etc.), the data type is the pair<key,value>.
   void  Insert(const void *data, void *container, size_t size) override;

   // Read portion of the streamer
   void ReadBuffer(TBuffer &buff, void *pObj) override;
   void ReadBuffer(TBuffer &buff, void *pObj, const TClass *onfile) override;

   // Streamer for I/O handling
   void Streamer(TBuffer &refBuffer) override;

   // Streamer I/O overload
   void Streamer(TBuffer &buff, void *pObj, int siz) override
   {
      TGenCollectionProxy::Streamer(buff,pObj,siz);
   }

   // Check validity of the proxy itself
   Bool_t IsValid() const;
};

#endif
