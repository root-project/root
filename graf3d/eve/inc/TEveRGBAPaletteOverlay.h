// @(#)root/eve:$Id$
// Author: Alja Mrak Tadel 2012

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveRGBAPaletteOverlay
#define ROOT_TEveRGBAPaletteOverlay

#include "TAxis.h"
#include "TGLOverlay.h"
#include "TGLAxisPainter.h"

class TEveRGBAPalette;

class TEveRGBAPaletteOverlay : public TGLOverlayElement
{
private:
   TEveRGBAPaletteOverlay(const TEveRGBAPaletteOverlay&);            // Not implemented
   TEveRGBAPaletteOverlay& operator=(const TEveRGBAPaletteOverlay&); // Not implemented

protected:
   TEveRGBAPalette *fPalette;
   TAxis            fAxis;
   TGLAxisPainter   fAxisPainter;

   Float_t          fPosX;         // x position
   Float_t          fPosY;         // y position
   Float_t          fWidth;        // width
   Float_t          fHeight;       // height

public:
   TEveRGBAPaletteOverlay(TEveRGBAPalette* p, Float_t posx, Float_t posy,
                          Float_t width, Float_t height);
   virtual ~TEveRGBAPaletteOverlay() {}

   virtual void Render(TGLRnrCtx& rnrCtx);

   TAxis&           RefAxis() { return fAxis; }
   TGLAxisPainter&  RefAxisPainter() { return fAxisPainter; }


   void SetPosition(Float_t x, Float_t y) { fPosX = x; fPosY = y; }
   void SetSize(Float_t w, Float_t h) { fWidth = w; fHeight = h; }

   ClassDef(TEveRGBAPaletteOverlay, 0); // Draws TEveRGBAPalette as GL overlay.
};

#endif
