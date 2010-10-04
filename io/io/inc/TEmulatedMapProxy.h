// @(#)root/io:$Id$
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TEmulatedMapProxy
#define ROOT_TEmulatedMapProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEmulatedMapProxy
//
// Streamer around an arbitrary STL like container, which implements basic
// container functionality.
//
//////////////////////////////////////////////////////////////////////////

#include "TEmulatedCollectionProxy.h"

class TEmulatedMapProxy : public TEmulatedCollectionProxy  {

protected:
   // Map input streamer
   void ReadMap(int nElements, TBuffer &b);

   // Map output streamer
   void WriteMap(int nElements, TBuffer &b);

public:
   // Virtual copy constructor
   virtual TVirtualCollectionProxy* Generate() const;

   // Copy constructor
   TEmulatedMapProxy(const TEmulatedMapProxy& copy);

   // Initializing constructor
   TEmulatedMapProxy(const char* cl_name);

   // Standard destructor
   virtual ~TEmulatedMapProxy();

   // Return the address of the value at index 'idx'
   virtual void *At(UInt_t idx);

   // Return the current size of the container
   virtual UInt_t Size() const;

   // Read portion of the streamer
   virtual void ReadBuffer(TBuffer &buff, void *pObj);
   virtual void ReadBuffer(TBuffer &buff, void *pObj, const TClass *onfile);

   // Streamer for I/O handling
   virtual void Streamer(TBuffer &refBuffer);

   // Streamer I/O overload
   virtual void Streamer(TBuffer &buff, void *pObj, int siz) {
      TEmulatedCollectionProxy::Streamer(buff,pObj,siz);
   }
};

#endif
