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

   fAxis.SetNdivisions(900);
   fAxisPainter.SetUseAxisColors(kFALSE);
   fAxisPainter.SetLabelPixelFontSize(10);
   fAxisPainter.SetFontMode(TGLFont::kPixmap);
   fAxisPainter.SetLabelAlign(TGLFont::kCenterH, TGLFont::kBottom);
}

void TEveRGBAPaletteOverlay::Render(TGLRnrCtx& rnrCtx)
{
   // Render the overlay.

   const Double_t ca_min = fPalette->GetCAMinAsDouble();
   const Double_t ca_max = fPalette->GetCAMaxAsDouble();

   // Uninitialized palette.
   if (ca_min == ca_max) return;

   fAxis.SetLimits(ca_min, ca_max);

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
      TGLUtil::Color4ubv(fPalette->ColorFromValue(fPalette->fCAMin));
      glVertex2f(0, 0);
      glVertex2f(0, fHeight);
      Float_t xs = fWidth / (fPalette->fCAMax - fPalette->fCAMin);
      Float_t x  = xs;
      for (Int_t i = fPalette->fCAMin + 1; i < fPalette->fCAMax; ++i)
      {
         TGLUtil::Color4ubv(fPalette->ColorFromValue(i));
         glVertex2f(x, 0);
         glVertex2f(x, fHeight);
         x += xs;
      }
      TGLUtil::Color4ubv(fPalette->ColorFromValue(fPalette->fCAMax));
      glVertex2f(fWidth, 0);
      glVertex2f(fWidth, fHeight);
      glEnd();
   }

   // axis
   glPushMatrix();
   Float_t sf = fWidth / (ca_max - ca_min);
   glScalef(sf, 1, 1);
   glTranslatef(-ca_min, 0, 0);
   fAxis.SetTickLength(0.05*fWidth);
   fAxisPainter.RefTMOff(0).Set(0, -1, 0);
   fAxisPainter.PaintAxis(rnrCtx, &fAxis);
   glPopMatrix();

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
