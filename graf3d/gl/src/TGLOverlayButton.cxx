// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLOverlayButton.h"
#include "TColor.h"
#include "TMath.h"

#include <TGLRnrCtx.h>
#include <TGLIncludes.h>
#include <TGLSelectRecord.h>
#include <TGLUtil.h>
#include <TGLCamera.h>
#include <TGLViewerBase.h>

/** \class TGLOverlayButton
\ingroup opengl
GL-overlay button.
*/

ClassImp(TGLOverlayButton);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLOverlayButton::TGLOverlayButton(TGLViewerBase *parent, const char *text,
   Float_t posx, Float_t posy, Float_t width, Float_t height) :
   TGLOverlayElement(),
   fText(text),
   fActiveID(-1),
   fBackColor(0x8080ff),
   fTextColor(0xffffff),
   fNormAlpha(0.2),
   fHighAlpha(1.0),
   fPosX(posx),
   fPosY(posy),
   fWidth(width),
   fHeight(height)
{
   if (parent)
      parent->AddOverlayElement(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Render the overlay elements.

void TGLOverlayButton::Render(TGLRnrCtx& rnrCtx)
{
   Float_t r, g, b;
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   if (rnrCtx.Selection())
   {
      TGLRect rect(*rnrCtx.GetPickRectangle());
      rnrCtx.GetCamera()->WindowToViewport(rect);
      gluPickMatrix(rect.X(), rect.Y(), rect.Width(), rect.Height(),
                    (Int_t*) rnrCtx.GetCamera()->RefViewport().CArr());
   }
   const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
   glOrtho(vp.X(), vp.Width(), vp.Y(), vp.Height(), 0, 1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   Float_t offset = (fPosY >= 0.0)? 0.0 : vp.Height()-fHeight;

   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glShadeModel(GL_FLAT);
   glClearColor(0.0, 0.0, 0.0, 0.0);
   glPushName(1);

   // Button rendering
   {
      TGLCapabilitySwitch move_to_back(GL_POLYGON_OFFSET_FILL, kTRUE);
      glPolygonOffset(0.5f, 0.5f);
      glPushMatrix();
      glTranslatef(fPosX, offset+fPosY, 0);
      // First the border, same color as text
      TColor::Pixel2RGB(fTextColor, r, g, b);
      (fActiveID == 1) ? TGLUtil::Color4f(r, g, b, fHighAlpha):TGLUtil::Color4f(r, g, b, fNormAlpha);
      TGLUtil::LineWidth(1);
      glBegin(GL_LINE_LOOP);
      glVertex2f(0.0, 0.0);
      glVertex2f(0.0, fHeight);
      glVertex2f(fWidth, fHeight);
      glVertex2f(fWidth, 0.0);
      glEnd();
      // then the button itself, with its own color
      // decrease a bit the highlight, to avoid bad effects...
      TColor::Pixel2RGB(fBackColor, r, g, b);
      (fActiveID == 1) ? TGLUtil::Color4f(r, g, b, fHighAlpha * 0.8):TGLUtil::Color4f(r, g, b, fNormAlpha);
      glBegin(GL_QUADS);
      glVertex2f(0.0, 0.0);
      glVertex2f(0.0, fHeight);
      glVertex2f(fWidth, fHeight);
      glVertex2f(fWidth, 0.0);
      glEnd();
      glPopMatrix();
   }

   // Text rendering
   {
      rnrCtx.RegisterFontNoScale(TMath::Nint(fHeight*0.8), "arial",  TGLFont::kPixmap, fFont);
      fFont.PreRender(kFALSE);

      TColor::Pixel2RGB(fTextColor, r, g, b);
      (fActiveID == 1) ? TGLUtil::Color4f(r, g, b, fHighAlpha):TGLUtil::Color4f(r, g, b, fNormAlpha);
      glPushMatrix();
      glTranslatef(fPosX+(fWidth/2.0), offset+fPosY+(fHeight/2.0), 0);
      Float_t llx, lly, llz, urx, ury, urz;
      fFont.BBox(fText.Data(), llx, lly, llz, urx, ury, urz);
      glRasterPos2i(0, 0);
      glBitmap(0, 0, 0, 0, -urx*0.5f, -ury*0.5f, 0);
      fFont.Render(fText.Data());
      fFont.PostRender();
      glPopMatrix();
   }
   glPopName();

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////
/// Emits "Clicked(TGLViewerBase*)" signal.
/// Called when user click on the GL button.

void TGLOverlayButton::Clicked(TGLViewerBase *viewer)
{
   Emit("Clicked(TGLViewerBase*)", (Longptr_t)viewer);
}

/******************************************************************************/
// Virtual event handlers from TGLOverlayElement
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Handle overlay event.
/// Return TRUE if event was handled.

Bool_t TGLOverlayButton::Handle(TGLRnrCtx         & rnrCtx,
                               TGLOvlSelectRecord & rec,
                               Event_t            * event)
{
   if (event->fCode != kButton1) {
      return kFALSE;
   }
   switch (event->fType) {
      case kButtonPress:
         if (rec.GetItem(1) == 1) {
            return kTRUE;
         }
         break;
      case kButtonRelease:
         if (rec.GetItem(1) == 1) {
            Clicked(rnrCtx.GetViewer());
            return kTRUE;
         }
         break;
      default:
         break;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has entered overlay area.

Bool_t TGLOverlayButton::MouseEnter(TGLOvlSelectRecord& /*rec*/)
{
   fActiveID = 1;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has left overlay area.

void TGLOverlayButton::MouseLeave()
{
   fActiveID = -1;
}
