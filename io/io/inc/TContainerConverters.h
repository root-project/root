// @(#)root/io:$Id: 9654411c49ffb811aa71b8ed4287acc53909a9a6 $
// Author: Philippe Canal 11/11/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TContainerConverters
#define ROOT_TContainerConverters

#include "TMemberStreamer.h"

class TVirtualCollectionProxy;
class TGenCollectionStreamer;
class TClassStreamer;

class TConvertClonesArrayToProxy : public TMemberStreamer {
   Bool_t  fIsPointer;
   Bool_t  fIsPrealloc;
   UInt_t  fOffset;
   TClass *fCollectionClass;
public:
   TConvertClonesArrayToProxy(TVirtualCollectionProxy *proxy, Bool_t isPointer, Bool_t isPrealloc);
   ~TConvertClonesArrayToProxy();
   void operator()(TBuffer &b, void *pmember, Int_t size=0) override;
};

class TConvertMapToProxy : public TMemberStreamer {
   Bool_t  fIsPointer;
   Bool_t  fIsPrealloc;
   UInt_t  fSizeOf;
   TClass *fCollectionClass;

public:
   TConvertMapToProxy(TClassStreamer *streamer, Bool_t isPointer, Bool_t isPrealloc);
   void operator()(TBuffer &b, void *pmember, Int_t size=0) override;
   Bool_t IsValid() { return fCollectionClass != nullptr; }
};

#endif
