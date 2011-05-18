// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLCameraGuide
#define ROOT_TGLCameraGuide

#include "TGLOverlay.h"

class TGLCameraGuide : public TGLOverlayElement
{
private:
   TGLCameraGuide(const TGLCameraGuide&);            // Not implemented
   TGLCameraGuide& operator=(const TGLCameraGuide&); // Not implemented

protected:
   Float_t fXPos;
   Float_t fYPos;
   Float_t fSize;

   Int_t   fSelAxis;
   Bool_t  fInDrag;

public:
   TGLCameraGuide(Float_t x, Float_t y, Float_t s,
                  ERole role=kUser, EState state=kActive);
   virtual ~TGLCameraGuide() {}

   void SetX(Float_t x) { fXPos = x; }
   void SetY(Float_t y) { fYPos = y; }
   void SetXY(Float_t x, Float_t y) { fXPos = x; fYPos = y; }
   void SetSize(Float_t s) { fSize = s; }

   virtual Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                         Event_t* event);
   virtual void   MouseLeave();

   virtual void Render(TGLRnrCtx& rnrCtx);

   ClassDef(TGLCameraGuide, 0); // Short description.
};

#endif
