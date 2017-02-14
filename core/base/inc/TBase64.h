// @(#)root/base:$Id$
// Author: Gerardo Ganis + Fons Rademakers   15/5/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBase64
#define ROOT_TBase64

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBase64                                                              //
//                                                                      //
// This code implements the Base64 encoding and decoding.               //
// Base64 encoded messages are typically used in authentication         //
// protocols and to pack binary data in HTTP messages.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include "TString.h"


class TBase64 {

public:
   virtual ~TBase64() { }

   static TString Encode(const char *data);
   static TString Encode(const char *data, Int_t len);
   static TString Decode(const char *data);

   ClassDef(TBase64,0)  // Base64 encoding/decoding
};

#endif
