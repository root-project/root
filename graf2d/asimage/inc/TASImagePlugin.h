// @(#)root/graf:$Id$
// Author: Valeriy Onuchin   23/06/05

/*************************************************************************
 * Copyright (C) 2001-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TASImagePlugin
#define ROOT_TASImagePlugin


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TASImagePlugin                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TImagePlugin.h"

struct ASImage;

class TASImagePlugin : public TImagePlugin {

public:
   TASImagePlugin(const char *ext) : TImagePlugin(ext) { }
   virtual ~TASImagePlugin() { }

   unsigned char *ReadFile(const char * /*filename*/, UInt_t & /*w*/,  UInt_t & /*h*/) override { return nullptr; }
   Bool_t WriteFile(const char * /*filename*/, unsigned char * /*argb*/, UInt_t /*w*/,  UInt_t  /*h*/) override { return kFALSE; }
   virtual ASImage *File2ASImage(const char * /*filename*/) {  return nullptr; }
   virtual Bool_t ASImage2File(ASImage * /*asimage*/) { return kFALSE; }

   ClassDefOverride(TASImagePlugin, 0)  // asimage plugin
};

#endif
