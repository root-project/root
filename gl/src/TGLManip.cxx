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
UInt_t TGLManip::fgQuality = 40;

Float_t TGLManip::fgRed[4]    = {1.0, 0.0, 0.0, 1.0 };
Float_t TGLManip::fgGreen[4]  = {0.0, 1.0, 0.0, 1.0 };
Float_t TGLManip::fgBlue[4]   = {0.0, 0.0, 1.0, 1.0 };
Float_t TGLManip::fgYellow[4] = {1.0, 1.0, 0.0, 1.0 };
Float_t TGLManip::fgWhite[4]  = {1.0, 1.0, 1.0, 1.0 };

ClassImp(TGLManip)

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer) : 
   fViewer(viewer), fShape(0), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouseX(0), fFirstMouseY(0), 
   fLastMouseX(0), fLastMouseY(0)
{
}

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   fViewer(viewer), fShape(shape), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouseX(0), fFirstMouseY(0), 
   fLastMouseX(0), fLastMouseY(0)
{
}

//______________________________________________________________________________
TGLManip::~TGLManip() 
{
}

//______________________________________________________________________________
void TGLManip::Select() 
{
   static UInt_t selectBuffer[4*4];
   glSelectBuffer(4*4, &selectBuffer[0]);
   glRenderMode(GL_SELECT);
   glInitNames();
   Draw();
   Int_t hits = glRenderMode(GL_RENDER);
   TGLUtil::CheckError();
   if (hits < 0) {
      Error("TGLManip::Select", "selection buffer overflow");
      return;
   }

   if (hits > 0) {
      UInt_t minDepth = selectBuffer[1];
      fSelectedWidget = selectBuffer[3];
      for (Int_t i = 1; i < hits; i++) {
         // Skip selection on sphere (unnamed hit)
         // drawn last so doesn't effect array offsets - just terminate
         if (selectBuffer[i * 4] == 0) {
            break;
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
Bool_t TGLManip::HandleButton(Event_t * event)
{
   // Only interested in Left mouse button actions
   if (event->fCode != kButton1) {
      return kFALSE;
   }

   // Mouse down on selected widget?
   if (event->fType == kButtonPress && fSelectedWidget != 0) {
      fFirstMouseX = event->fX;
      fFirstMouseY = event->fY;
      fLastMouseX = event->fX;
      fLastMouseY = event->fY;
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
Bool_t TGLManip::HandleMotion(Event_t * event, const TGLCamera & /*camera*/ )
{
   TGLRect selectRect(event->fX, event->fY, 3, 3);
   // Need to do this cross thread under Windows for gVirtualGL context - very ugly...
   UInt_t oldSelection = fSelectedWidget;
   fViewer.RequestSelectManip(selectRect);
   return (fSelectedWidget != oldSelection);
}

//______________________________________________________________________________
void TGLManip::DrawAxisWidgets(EHeadShape head) const
{
   if (!fShape) {
      return;
   }

   const TGLBoundingBox & box = fShape->BoundingBox();

   Double_t widgetSize = box.Extents().Mag() / 300.0;

   // Draw three axis widgets out of bounding box
   // GL name loading for hit testing - 0 reserved for no selection
   glPushName(1);
   DrawAxisWidget(head, box.Center(), box.Axis(0, kFALSE)*-0.51, widgetSize, fSelectedWidget == 1 ? fgYellow : fgRed);
   glPopName();
   glPushName(2);
   DrawAxisWidget(head, box.Center(), box.Axis(1, kFALSE)*-0.51, widgetSize, fSelectedWidget == 2 ? fgYellow : fgGreen);
   glPopName();
   glPushName(3);
   DrawAxisWidget(head, box.Center(), box.Axis(2, kFALSE)*-0.51, widgetSize, fSelectedWidget == 3 ? fgYellow : fgBlue);
   glPopName();

   // Draw central origin sphere
   DrawOrigin(box.Center(), widgetSize*2.0, fgWhite);
}

//______________________________________________________________________________
void TGLManip::DrawAxisWidget(EHeadShape head, const TGLVertex3 & origin, const TGLVector3 & vector, Double_t size, Float_t rgba[4]) const
{    
   // Draw an axis widget with head type of arrow or box
   SetDrawColors(rgba);
   glPushMatrix();
   TGLMatrix local(origin, vector);
   glMultMatrixd(local.CArr());
   gluCylinder(fgQuad.Get(), size, size, vector.Mag(), fgQuality, fgQuality); // Line
   gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(fgQuad.Get(), 0.0, size, fgQuality, fgQuality); 

   // TODO: axis is longer than vector by head object - this is attached at end
   // doesn't really matter...?
   glTranslated(0.0, 0.0, vector.Mag()); // Shift down local Z to end of vector

   if (head == kArrow) {
      gluDisk(fgQuad.Get(), 0.0, size*4.0, fgQuality, fgQuality); 
      gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_OUTSIDE);
      gluCylinder(fgQuad.Get(), size*4.0, 0.0, size*8.0, fgQuality, fgQuality); // Arrow head
   } else if (head == kBox) {
      gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_OUTSIDE);
      TGLBoundingBox box(TGLVertex3(-size*2.0, -size*2.0, 0.0), TGLVertex3(size*2.0, size*2.0, size*4.0));
      box.Draw(kTRUE);
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLManip::DrawOrigin(const TGLVertex3 & origin, Double_t size, Float_t rgba[4]) const
{
   SetDrawColors(rgba);
   glPushMatrix();
   glTranslated(origin.X(), origin.Y(), origin.Z());
   gluSphere(fgQuad.Get(), size, fgQuality, fgQuality);
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
