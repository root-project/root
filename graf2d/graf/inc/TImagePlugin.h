// @(#)root/graf:$Id$
// Author: Valeriy Onuchin   23/06/05

/*************************************************************************
 * Copyright (C) 2001-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TImagePlugin
#define ROOT_TImagePlugin


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TImagePlugin                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TImagePlugin : public TObject {

protected:
   TString fExtension;  // file extension

public:
   TImagePlugin(const char *ext) { fExtension = ext; }
   virtual ~TImagePlugin() { }

   virtual unsigned char *ReadFile(const char *filename, UInt_t &w,  UInt_t &h) = 0;
   virtual Bool_t WriteFile(const char *filename, unsigned char *argb, UInt_t w,  UInt_t  h) = 0;
   ULong_t Hash() const { return fExtension.Hash(); }

   ClassDef(TImagePlugin, 0)  // base class for different image format handlers(plugins)
};

#endif
