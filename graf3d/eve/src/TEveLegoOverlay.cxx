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

#include <THLimitsFinder.h>


//______________________________________________________________________________
//
//
// GL-overaly control GUI for TEveCaloLego.
//
//

ClassImp(TEveLegoOverlay);

//______________________________________________________________________________
TEveLegoOverlay::TEveLegoOverlay() :
   TGLOverlayElement(),
   TEveElementList("Lego Menu", "TEveLegoOverlay", kTRUE),

   fCalo(0),

   fActiveID(-1),
   fActiveCol(kRed-4),

   fMenuW(0.08),
   fButtonW(0.5),
   fSliderH(0.6),
   fSliderPosY(0.15),

   fShowSlider(kFALSE),
   fSliderVal(0)
{
   // Constructor.

}


/******************************************************************************/
void TEveLegoOverlay::RenderText(const char* txt,Float_t y)
{
   // Render pixmap text at given y.

   glPushMatrix();
   glTranslatef(0, y, 0);

   Float_t llx, lly, llz, urx, ury, urz;
   fNumFont.BBox(txt, llx, lly, llz, urx, ury, urz);
   glRasterPos2i(0, 0);

   glBitmap(0, 0, 0, 0, -urx, -ury*0.5f, 0);
   fNumFont.Render(txt);
   glPopMatrix();
}

/******************************************************************************/
void TEveLegoOverlay::DrawSlider(TGLRnrCtx& rnrCtx)
{
   // Draw slider and calorimeter Z scale on left side of screen.

   TGLUtil::Color(fCalo->GetFontColor());
   Float_t off = -0.01; 

   // font
   TGLRect& wprt = rnrCtx.RefCamera().RefViewport();
   Float_t cfs =   wprt.Height()*fSliderH*0.07;
   Int_t fs = TGLFontManager::GetFontSize(cfs);
   if (fNumFont.GetMode() == TGLFont::kUndef)
   {
      rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fNumFont);
   }
   else if (fNumFont.GetSize() != fs)
   {
      rnrCtx.ReleaseFont(fNumFont);
      rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fNumFont);
   }

   // optimize binning
   Float_t w = fButtonW*fMenuW*0.5f;
   glTranslatef(0, fSliderPosY, 0);
   Int_t nsteps, ndiv;
   Double_t omin, omax, zmax, tickval;
   THLimitsFinder::Optimize(0, fCalo->GetData()->GetMaxVal(), fCalo->GetNZSteps(), omin, omax, ndiv, tickval);
   nsteps = ndiv+1;
   zmax = nsteps*tickval;
   
   // labels
   fNumFont.PreRender(kFALSE);
   glPushMatrix();
   glTranslatef(3*off, 0, 0);
   glScalef(1, fSliderH/zmax , 1.);
   Double_t val = 0;
   for(Int_t i=0; i<=nsteps; i++)
   {
      RenderText(TEveUtil::FormAxisValue(val), val);
      val+= tickval;
   }
   glPopMatrix();
   fNumFont.PostRender();


  // event handling
   if (rnrCtx.Selection())
   {
      glLoadName(2);
      glBegin(GL_QUADS);
      glVertex2f(-w, 0);
      glVertex2f( w, 0);
      glVertex2f( w, fSliderH);
      glVertex2f(-w, fSliderH);
      glEnd();
   }


   glPushMatrix();
   glScalef(1, fSliderH/zmax , 1.);

   // body
   glLineWidth(1);
   glBegin(GL_LINES);
   glVertex2f(0, 0);
   glVertex2f(0, zmax);
   // primary tick-marks
   Double_t tv1= 0;
   for (Int_t i = 0; i <= nsteps; ++i)
   {
      glVertex2d(0,  tv1);
      glVertex2d(2*off , tv1);
      tv1+= tickval;
   }

   // secondary tick-marks
   Double_t omin2, zmax2, tickval2;
   Int_t nsteps2;
   THLimitsFinder::Optimize(0, tickval, fCalo->GetNZSteps(), omin2, zmax2, nsteps2, tickval2);
   Int_t nt2 = Int_t(zmax/tickval2);
   Float_t step2 = zmax/nt2;

   Double_t tv2= 0;
   for (Int_t i = 0; i <= nt2; ++i)
   {
      glVertex2d(0,  tv2);
      glVertex2d(off , tv2);
      tv2+= step2;
   }
   glEnd();

   glPopMatrix();
   

   // marker
   TGLUtil::Color((fActiveID == 2) ? fActiveCol : 3);
   glPointSize(8);
   glBegin(GL_POINTS);
   glVertex3f(0, fSliderVal*fSliderH, -0.1);
   glEnd(); 

}

/******************************************************************************/
void TEveLegoOverlay::Render(TGLRnrCtx& rnrCtx)
{
   // Render the overlay elements.

   if (! fCalo) return;

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
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();
   glScalef(2, 2, 1); // normalised coordinates

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
   glTranslatef(0.5 -fMenuW*0.5, -0.5, 0); // translate to lower left corner

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
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
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
