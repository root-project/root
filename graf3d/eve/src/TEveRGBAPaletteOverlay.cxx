// @(#)root/eve:$Id$
// Author: Alja Mrak Tadel 2012

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveRGBAPaletteOverlay.h"
#include "TEveRGBAPalette.h"

#include "TGLIncludes.h"
#include "TGLAxis.h"
#include "TGLRnrCtx.h"
#include "TGLUtil.h"

//______________________________________________________________________________
// Description of TEveRGBAPaletteOverlay
//

ClassImp(TEveRGBAPaletteOverlay);

//______________________________________________________________________________
TEveRGBAPaletteOverlay::TEveRGBAPaletteOverlay(TEveRGBAPalette* p, Float_t posx, Float_t posy,
                                               Float_t width, Float_t height) :
   TGLOverlayElement(),
   fPalette(p),
   fPosX(posx),
   fPosY(posy),
   fWidth(width),
   fHeight(height)
{
   // Constructor.

   fAxis.SetLimits(fPalette->GetMinVal(), fPalette->GetMaxVal());
   fAxis.SetNdivisions(900);
   fAxisPainter.SetUseAxisColors(kFALSE);
   fAxisPainter.SetLabelPixelFontSize(10);
   fAxisPainter.SetFontMode(TGLFont::kPixmap);
   fAxisPainter.SetLabelAlign(TGLFont::kCenterH, TGLFont::kBottom);
}

void TEveRGBAPaletteOverlay::Render(TGLRnrCtx& rnrCtx) 
{ 
   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);

   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);  
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);

   // reset to [0,1] units
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   glOrtho(0, 1, 0, 1, 0, 1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   // postion pallette
   glTranslatef(fPosX, fPosY, 0);

   // colored quads
   {
      TGLCapabilitySwitch move_to_back(GL_POLYGON_OFFSET_FILL, kTRUE);
      glPolygonOffset(0.5f, 0.5f);

      glBegin(GL_QUAD_STRIP);
      TGLUtil::Color4ubv(fPalette->ColorFromValue(fPalette->GetMinVal()));
      glVertex2f(0, 0);
      glVertex2f(0, fHeight);
      Float_t xs = fWidth/(fPalette->GetMaxVal() - fPalette->GetMinVal());
      Float_t x  = xs;
      for (Int_t i = fPalette->GetMinVal() + 1; i < fPalette->GetMaxVal(); i++)
      {
         TGLUtil::Color4ubv(fPalette->ColorFromValue(i));
         glVertex2f(x, 0);
         glVertex2f(x, fHeight);
         x += xs;
      }
      TGLUtil::Color4ubv(fPalette->ColorFromValue(fPalette->GetMaxVal()));
      glVertex2f(fWidth, 0);
      glVertex2f(fWidth, fHeight);
      glEnd();
   }

   // axis
   Float_t sf = fWidth / (fPalette->GetMaxVal() - fPalette->GetMinVal());
   glScalef(sf, 1, 1);
   fAxis.SetTickLength(0.05*fWidth);
   fAxisPainter.RefTMOff(0).Set(0, -1, 0);
   fAxisPainter.PaintAxis(rnrCtx, &fAxis);
   glScalef(1/sf, 1, 1);

   // frame around palette
   glBegin(GL_LINE_LOOP);
   glVertex2f(0, 0);             glVertex2f(fWidth, 0);
   glVertex2f(fWidth, fHeight);  glVertex2f(0, fHeight);
   glEnd();

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();

   glPopAttrib();
}
