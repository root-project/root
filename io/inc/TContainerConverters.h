// @(#)root/io:$Name:  $:$Id: TContainerConverters.h,v 1.2 2007/02/07 08:51:41 brun Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Small helper to read a TBuffer containing a TClonesArray into any   //
//  valid collection.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMemberStreamer
#include "TMemberStreamer.h"
#endif
class TVirtualCollectionProxy;

class TConvertClonesArrayToProxy : public TMemberStreamer {
   Bool_t fIsPointer;
   Bool_t fIsPrealloc;
   UInt_t fOffset;
   TVirtualCollectionProxy *fProxy;
public:
   TConvertClonesArrayToProxy(TVirtualCollectionProxy *proxy, Bool_t isPointer, Bool_t isPrealloc);
   void operator()(TBuffer &b, void *pmember, Int_t size=0);
};

#endif
