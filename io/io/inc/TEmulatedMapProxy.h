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

#include "TEmulatedCollectionProxy.h"

class TEmulatedMapProxy : public TEmulatedCollectionProxy  {

protected:
   // Map input streamer
   void ReadMap(UInt_t nElements, TBuffer &b);

   // Map output streamer
   void WriteMap(UInt_t nElements, TBuffer &b);
private:
   TEmulatedMapProxy &operator=(const TEmulatedMapProxy &rhs); // Not implemented.

public:
   // Virtual copy constructor
   TVirtualCollectionProxy* Generate() const override;

   // Copy constructor
   TEmulatedMapProxy(const TEmulatedMapProxy& copy);

   // Initializing constructor
   TEmulatedMapProxy(const char* cl_name, Bool_t silent);

   // Standard destructor
   virtual ~TEmulatedMapProxy();

   // Return the address of the value at index 'idx'
   void *At(UInt_t idx) override;

   // Return the current size of the container
   UInt_t Size() const override;

   // Read portion of the streamer
   void ReadBuffer(TBuffer &buff, void *pObj) override;
   void ReadBuffer(TBuffer &buff, void *pObj, const TClass *onfile) override;

   // Streamer for I/O handling
   void Streamer(TBuffer &refBuffer) override;

   // Streamer I/O overload
   void Streamer(TBuffer &buff, void *pObj, int siz) override
   {
      TEmulatedCollectionProxy::Streamer(buff,pObj,siz);
   }
};

#endif
