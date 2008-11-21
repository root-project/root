// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveLegoOverlay.h"

#include "TEveCalo.h"
#include "TEveCaloData.h"

#include <TGLRnrCtx.h>
#include <TGLIncludes.h>
#include <TGLSelectRecord.h>
#include <TGLUtil.h>
#include <TGLCamera.h>

//______________________________________________________________________________
//
//
// GL-overaly control GUI for TEveCaloLego.
//
//

ClassImp(TEveLegoOverlay);

//______________________________________________________________________________
TEveLegoOverlay::TEveLegoOverlay() :
   TGLCameraOverlay(),
   TEveElementList("Lego Menu", "TEveLegoOverlay", kTRUE),
   fHeaderSelected(kFALSE),

   fCalo(0),
   fMainColor(kGray),

   fShowCamera(kTRUE),
   fShowPlane(kFALSE),

   fMenuW(0.08),
   fButtonW(0.5),
   fSliderH(0.6),
   fSliderPosY(0.15),

   fShowSlider(kFALSE),
   fSliderVal(0),

   fActiveID(-1),
   fActiveCol(kRed-4)
{
   // Constructor.
  fMainColorPtr = &fMainColor;

}


/******************************************************************************/
void TEveLegoOverlay::DrawSlider(TGLRnrCtx& rnrCtx)
{
   // Draw slider and calorimeter Z scale on left side of screen.

   glTranslatef(0, fSliderPosY, 0.5);

   // event handling
   if (rnrCtx.Selection())
   {
      glLoadName(2);
      Float_t w = fButtonW*fMenuW*0.5f;
      glBegin(GL_QUADS);
      glVertex2f(-w, 0);
      glVertex2f( w, 0);
      glVertex2f( w, fSliderH);
      glVertex2f(-w, fSliderH);
      glEnd();
   }

   // drawing
   if ( fCalo->GetData() && fCalo->GetData()->Empty() == kFALSE)
   {
      // axis
      Double_t maxVal = fCalo->GetMaxVal();
      TGLRect& wprt = rnrCtx.RefCamera().RefViewport();
      Float_t fs = wprt.Height()*fSliderH* fAxisAtt.GetLabelSize();
      fAxisAtt.SetAbsLabelFontSize(TGLFontManager::GetFontSize(fs, 12, 36));

      fAxisAtt.RefDir().Set(0, 1, 0);
      fAxisAtt.SetTextAlign(TGLFont::kLeft);
      fAxisAtt.SetRng(0, maxVal);
      fAxisAtt.RefTMOff(0).X() = -maxVal*0.03;
      fAxisAtt.SetAbsLabelFontSize(TMath::Nint(fs));

      glPushMatrix();
      glScalef( fSliderH/maxVal, fSliderH/maxVal, 1.);
      fAxisAtt.SetAxisColor(fMainColor);
      fAxisAtt.SetLabelColor(fMainColor);
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();

      // marker
      TGLUtil::Color((fActiveID == 2) ? fActiveCol : 3);
      glPointSize(8);
      glBegin(GL_POINTS);
      glVertex3f(0, fSliderVal*fSliderH, -0.1);
      glEnd();
   }
}

//______________________________________________________________________________
void TEveLegoOverlay::RenderPlaneInterface(TGLRnrCtx &rnrCtx)
{
   // Render menu for plane-value and the plane if marked.

   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LINE_BIT | GL_POINT_BIT);
   glEnable(GL_POINT_SMOOTH);
   glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(0.1, 1);
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glPushName(0);


   // move to the center of menu
   glPushMatrix();
   glTranslatef(1 -fMenuW, 0, 0); // translate to lower left corner

   // button
   glPushMatrix();
   glTranslatef(0, (1-fButtonW )*fMenuW*0.8, 0);
   glLoadName(1);
   Float_t a=0.6;
   (fActiveID == 1) ? TGLUtil::Color(fActiveCol):TGLUtil::Color4f(0, 1, 0, a);
   Float_t bw = fButtonW*fMenuW*0.5;
   Float_t bwt = bw*0.8;
   Float_t bh = fButtonW*fMenuW;
   glBegin(GL_QUADS);
   glVertex2f(-bw, 0);
   glVertex2f( bw, 0);
   glVertex2f( bwt, bh);
   glVertex2f(-bwt, bh);
   glEnd();

   TGLUtil::Color(4);

   glLineWidth(1);
   glBegin(GL_LINES);
   glVertex2f(0, 0); glVertex2f(0, bh);
   glVertex2f((bw+bwt)*0.5, bh*0.5); glVertex2f(-(bw+bwt)*0.5, bh*0.5);
   glEnd();

   glLineWidth(2);
   glBegin(GL_LINE_LOOP);
   glVertex2f(-bw, 0);
   glVertex2f( bw, 0);
   glVertex2f( bwt, bh);
   glVertex2f(-bwt, bh);
   glEnd();

   glPopMatrix();

   if (fShowSlider) DrawSlider(rnrCtx);

   glPopName();
   glPopAttrib();

   glPopMatrix();

}

/******************************************************************************/
void TEveLegoOverlay::RenderHeader(TGLRnrCtx& rnrCtx)
{
   // Render text on top right corner of the screen.

   TGLRect &vp = rnrCtx.GetCamera()->RefViewport();

   TGLFont font;
   Int_t fs = TGLFontManager::GetFontSize(vp.Height()*0.035, 12, 36);
   rnrCtx.RegisterFont(fs, "arial", TGLFont::kPixmap, font);
   Float_t off = fs*0.2;
   Float_t bb[6];
   font.BBox(fHeaderTxt.Data(), bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
   Float_t x = vp.Width() -bb[3] -off;
   Float_t y = vp.Height() -bb[4] -off;
   if (rnrCtx.Selection())
   {
      glPushName(0);
      glLoadName(3);
      glBegin(GL_QUADS);
      glVertex2f(x/vp.Width(), y/ vp.Height());
      glVertex2f(1,  y/ vp.Height());
      glVertex2f(1, 1);
      glVertex2f(x/vp.Width(), 1);
      glEnd();
      glPopName();
   }
   else
   {
      TGLUtil::Color(fHeaderSelected ? fActiveCol : fCalo->GetFontColor());
      glRasterPos2i(0, 0);
      glBitmap(0, 0, 0, 0, x, y, 0);
      font.Render(fHeaderTxt.Data());
   }
}

/******************************************************************************/
void TEveLegoOverlay::Render(TGLRnrCtx& rnrCtx)
{
   // Render the overlay elements.

   if (fShowPlane || fHeaderTxt.Length())
   {
      // go to normalised coordinates
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      if (rnrCtx.Selection())
      {
         TGLRect rect(*rnrCtx.GetPickRectangle());
         rnrCtx.GetCamera()->WindowToViewport(rect);
         gluPickMatrix(rect.X(), rect.Y(), rect.Width(), rect.Height(), rnrCtx.RefCamera().RefViewport().CArr());
      }
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();

      glTranslatef(-1, -1, 0);

      glScalef(2, 2, 1);
      if (fShowPlane) RenderPlaneInterface(rnrCtx);
      RenderHeader(rnrCtx);

      glPopMatrix();
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
   }


   // call TGLCamera overlay render
   if (fShowCamera)
   {
      TGLCamera& cam = rnrCtx.RefCamera();
      TGLRect& vp = cam.RefViewport();
      Bool_t skip = kFALSE;
      // check if lego axis are already visible
      if (cam.IsOrthographic())
      {
         using namespace TMath;

         Float_t x0 = fCalo->GetEtaMin();
         Float_t y0 = fCalo->GetPhiMin();

         const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
         GLdouble mm[16];
         glGetDoublev(GL_MODELVIEW_MATRIX,  mm);

         GLdouble x, y, z;
         gluProject(x0, y0, 0, mm, pm, (Int_t*)vp.CArr(), &x, &y, &z);
         // viewport height goes from top to bottom
         if ( x > vp.Left() && y > vp.Top())
            skip = kTRUE;
      }

      if (!skip) TGLCameraOverlay::Render(rnrCtx);
   }
}

/******************************************************************************/
// Virtual event handlers from TGLOverlayElement
/******************************************************************************/

Bool_t TEveLegoOverlay::SetSliderVal(Event_t* event, TGLRnrCtx &rnrCtx)
{
   // Set height of horizontal plane in the calorimeter.

   TGLRect& wprt = rnrCtx.RefCamera().RefViewport();
   fSliderVal = (1 -event->fY*1./wprt.Height() -fSliderPosY)/fSliderH;

   if (fSliderVal < 0 )
      fSliderVal = 0;
   else if (fSliderVal > 1)
      fSliderVal = 1;

   fCalo->SetHPlaneVal(fSliderVal);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TEveLegoOverlay::Handle(TGLRnrCtx          & rnrCtx,
                               TGLOvlSelectRecord & rec,
                               Event_t            * event)
{
   // Handle overlay event.
   // Return TRUE if event was handled.

   switch (event->fType)
   {
      case kMotionNotify:
      {
         Int_t item = rec.GetN() < 2 ? -1 : (Int_t)rec.GetItem(1);
         if (fActiveID != item) {
            fActiveID = item;
            return kTRUE;
         } else {
            if (fActiveID == 2 && event->fState == 256)
               return SetSliderVal(event, rnrCtx);

            return kFALSE;
         }
         break;
      }
      case kButtonPress:
      {
         if (event->fCode != kButton1) {
            return kFALSE;
         }
         switch (rec.GetItem(1))
         {
            case 1:
               fShowSlider = !fShowSlider;
               fCalo->SetDrawHPlane(fShowSlider);
               break;
            case 2:
               return SetSliderVal(event, rnrCtx);
            case 3:
               fHeaderSelected = !fHeaderSelected;
            default:
               break;
         }
      }
      default:
         break;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TEveLegoOverlay::MouseEnter(TGLOvlSelectRecord& /*rec*/)
{
   // Mouse has entered overlay area.

   return kTRUE;
}

//______________________________________________________________________________
void TEveLegoOverlay::MouseLeave()
{
   // Mouse has left overlay area.

   fActiveID = -1;
}
