// @(#)root/cont:$Name:  $:$Id: TEmulatedVectorProxy.h,v 1.2 2004/02/18 07:28:02 brun Exp $
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef Root_TEmulatedVectorProxy_h
#define Root_TEmulatedVectorProxy_h

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEmulatedVectorProxy                                                 //
//                                                                      //
// Proxy around an emulated stl vector                                  //
//                                                                      //
// In particular this is used to implement splitting, emulation,        //
// and TTreeFormula access to STL vector for which we do not have       //
// access to the compiled copde             .                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualCollectionProxy.h"

class TEmulatedVectorProxy : public TVirtualCollectionProxy, public TClassStreamer {

  typedef std::vector<void*> ProxyList_t;
   
   TString      fProxiedName; // name of the class being proxied.

   TClass      *fValueClass;  //! TClass of object in collection
   void        *fProxied;     //! Address of the proxied vector
   ProxyList_t  fProxyList;   //! Stack of proxied containers
   Int_t        fSize;        //! Sizeof the contained objects
   UInt_t       fCase;        //! type of data
   EDataType    fKind;        //! kind of fundamental type 

   TEmulatedVectorProxy() : 
      fValueClass(0), fProxied(0), fSize(-1), 
      fCase(0), fKind(kNoType_t) {}
   void Init();
   void Destruct(Int_t first,Int_t last,Int_t n);

public:
   TVirtualCollectionProxy* Generate() const  { return new TEmulatedVectorProxy(fProxiedName); }

   TEmulatedVectorProxy(const char *classname);
   TEmulatedVectorProxy(TClass *collectionClass);
   ~TEmulatedVectorProxy() {}

   void   *New() const;
   void   *New(void *arena) const;
   UInt_t  Sizeof() const;
   void    SetProxy(void *objstart) { fProxied = objstart; }
   void    PushProxy(void *objstart) { fProxyList.push_back(fProxied); fProxied = objstart;  }
   void    PopProxy() { fProxied = fProxyList.back(); fProxyList.pop_back(); } 

   virtual Bool_t    HasPointers() const; // Return true if the content is of type 'pointer to'
   virtual TClass   *GetValueClass();     // Return a pointer to the TClass representing the content.
   virtual EDataType GetType();           // If the content is a simple numerical value, return its type (see TDataType)

   void   *At(UInt_t idx);                       // Return the address of the value at index 'idx'
   void    Clear(const char *opt = "");          // Clear the container
   void    Resize(UInt_t n, Bool_t forceDelete); // Resize the container
   UInt_t  Size() const;                         // Return the current size of the container

   void    Streamer(TBuffer &b);
   virtual void operator()(TBuffer &b, void *objp) { fProxied = objp; Streamer(b); }
};

#endif /* Root_TEmulatedVectorProxy_h */
