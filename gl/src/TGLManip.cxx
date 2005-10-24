// @(#)root/gl:$Name:  $:$Id: TGLManip.cxx
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLManip.h"
#include "TGLUtil.h"
#include "TGLCamera.h"
#include "TGLViewer.h"
#include "TGLPhysicalShape.h"
#include "TGLIncludes.h"

TGLQuadric TGLManip::fgQuad;

//TODO: Should really use LOD adjustment on top?
UInt_t TGLManip::fgQuality = 60;

Float_t TGLManip::fgRed[4]    = {1.0, 0.0, 0.0, 1.0 };
Float_t TGLManip::fgGreen[4]  = {0.0, 1.0, 0.0, 1.0 };
Float_t TGLManip::fgBlue[4]   = {0.0, 0.0, 1.0, 1.0 };
Float_t TGLManip::fgYellow[4] = {1.0, 1.0, 0.0, 1.0 };
Float_t TGLManip::fgWhite[4]  = {1.0, 1.0, 1.0, 1.0 };
Float_t TGLManip::fgGrey[4]   = {0.5, 0.5, 0.5, 0.4 };

ClassImp(TGLManip)

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer) : 
   fViewer(viewer), fShape(0), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0), 
   fLastMouse(0, 0)
{
}

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   fViewer(viewer), fShape(shape), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0), 
   fLastMouse(0, 0)
{
}

//______________________________________________________________________________
TGLManip::~TGLManip() 
{
}

//______________________________________________________________________________
void TGLManip::Select(const TGLCamera & camera) 
{
   static UInt_t selectBuffer[4*4];
   glSelectBuffer(4*4, &selectBuffer[0]);
   glRenderMode(GL_SELECT);
   glInitNames();
   Draw(camera);
   Int_t hits = glRenderMode(GL_RENDER);
   TGLUtil::CheckError();
   if (hits < 0) {
      Error("TGLManip::Select", "selection buffer overflow");
      return;
   }

   if (hits > 0) {
      fSelectedWidget = 0;
      UInt_t minDepth = kMaxUInt;
      for (Int_t i = 0; i < hits; i++) {
         // Skip selection on unnamed hits
         if (selectBuffer[i * 4] == 0) {
            continue;
         }
         if (selectBuffer[i * 4 + 1] < minDepth) {
            fSelectedWidget = selectBuffer[i * 4 + 3];
         }
      }
   } else {
      fSelectedWidget = 0;
   }
}

//______________________________________________________________________________
Bool_t TGLManip::HandleButton(const Event_t * event, const TGLCamera & /*camera*/)
{
   // Only interested in Left mouse button actions
   if (event->fCode != kButton1) {
      return kFALSE;
   }

   // Mouse down on selected widget?
   if (event->fType == kButtonPress && fSelectedWidget != 0) {
      fFirstMouse.SetX(event->fX);
      fFirstMouse.SetY(event->fY);
      fLastMouse.SetX(event->fX);
      fLastMouse.SetY(event->fY);
      fActive = kTRUE;
      return kTRUE;
   } else if (event->fType == kButtonRelease && fActive) {
      fActive = kFALSE;
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLManip::HandleMotion(const Event_t * event, const TGLCamera & /*camera*/)
{
   TGLRect selectRect(event->fX, event->fY, 3, 3);
   // Need to do this cross thread under Windows for gVirtualGL context - very ugly...
   UInt_t oldSelection = fSelectedWidget;
   fViewer.RequestSelectManip(selectRect);
   return (fSelectedWidget != oldSelection);
}

//______________________________________________________________________________
Double_t TGLManip::DrawScale(const TGLBoundingBox & box, const TGLCamera & camera) const
{
   TGLVector3 pixelInWorld = camera.ViewportDeltaToWorld(box.Center(), 1, 1);
   Double_t pixelScale = pixelInWorld.Mag();
   Double_t scale = box.Extents().Mag() / 100.0;
   if (scale < pixelScale * 2.0) {
      scale = pixelScale * 2.0;
   } else if (scale > pixelScale * 5.0) {
      scale = pixelScale * 5.0;
   }
   return scale;
}

//______________________________________________________________________________
void TGLManip::DrawAxisWidget(EHeadShape head, Double_t scale, const TGLVertex3 & origin, const TGLVector3 & vector, Float_t rgba[4]) const
{    
   // Draw an axis widget with head type of arrow or box
   SetDrawColors(rgba);
   glPushMatrix();
   TGLMatrix local(origin, vector);
   glMultMatrixd(local.CArr());
   gluCylinder(fgQuad.Get(), scale/4.0, scale/4.0, vector.Mag(), fgQuality, fgQuality); // Line
   gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(fgQuad.Get(), 0.0, scale/4.0, fgQuality, fgQuality); 

   // TODO: axis is longer than vector by head object - this is attached at end
   // doesn't really matter...?
   glTranslated(0.0, 0.0, vector.Mag()); // Shift down local Z to end of vector

   if (head == kArrow) {
      gluDisk(fgQuad.Get(), 0.0, scale, fgQuality, fgQuality); 
      gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_OUTSIDE);
      gluCylinder(fgQuad.Get(), scale, 0.0, scale*2.0, fgQuality, fgQuality); // Arrow head
   } else if (head == kBox) {
      gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_OUTSIDE);
      // TODO: Drawing box should be simplier - maybe make a static helper which BB + others use.
      // This doesn't tesselate properly - ugly lighting 
      TGLBoundingBox box(TGLVertex3(-scale*.7, -scale*.7, 0.0), TGLVertex3(scale*.7, scale*.7, scale*1.4));
      box.Draw(kTRUE);
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLManip::DrawOrigin(const TGLVertex3 & origin, Double_t scale, Float_t rgba[4]) const
{
   SetDrawColors(rgba);
   glPushMatrix();
   glTranslated(origin.X(), origin.Y(), origin.Z());
   gluSphere(fgQuad.Get(), scale*2.0, fgQuality, fgQuality);
   glPopMatrix();
}

//______________________________________________________________________________
void TGLManip::SetDrawColors(Float_t rgba[4]) const 
{
   static Float_t ambient[4] = {0.0, 0.0, 0.0, 1.0};
   static Float_t specular[4] = {0.8, 0.8, 0.8, 1.0};
   static Float_t emission[4] = {0.1, 0.1, 0.1, 1.0};

   glMaterialfv(GL_FRONT, GL_DIFFUSE, rgba);
   glMaterialfv(GL_FRONT, GL_AMBIENT, ambient);
   glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);
   glMaterialf(GL_FRONT, GL_SHININESS, 60.0);
}
