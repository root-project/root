// @(#)root/gui:$Id$
// Author: Bertrand Bellenot 23/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGShapedFrame
#define ROOT_TGShapedFrame

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TImage
#include "TImage.h"
#endif

#ifndef ROOT_TGPicture
#include "TGPicture.h"
#endif

class TGShapedFrame : public TGCompositeFrame {

private:
   TGShapedFrame(const TGShapedFrame&); // Not implemented
   TGShapedFrame& operator=(const TGShapedFrame&); // Not implemented

protected:
   const TGPicture      *fBgnd;     // picture used as background/shape
   TImage               *fImage;    // image used as background/shape
   virtual void          DoRedraw() {}

public:
   TGShapedFrame(const char *fname=0, const TGWindow *p=0, UInt_t w=1, UInt_t h=1, UInt_t options=0);
   virtual ~TGShapedFrame();

   const TGPicture   GetPicture() const { return *fBgnd; }
   TImage            GetImage() const { return *fImage; }

   virtual void      SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGShapedFrame, 0) // Shaped composite frame
};

#endif
