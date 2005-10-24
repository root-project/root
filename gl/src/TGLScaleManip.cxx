// @(#)root/gl:$Name:  $:$Id: TGLScaleManip.cxx
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLScaleManip.h"
#include "TGLPhysicalShape.h"
#include "TGLCamera.h"
#include "TGLIncludes.h"

ClassImp(TGLScaleManip)

//______________________________________________________________________________
TGLScaleManip::TGLScaleManip(TGLViewer & viewer) : TGLManip(viewer)
{
}

//______________________________________________________________________________
TGLScaleManip::TGLScaleManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   TGLManip(viewer, shape) 
{
}

//______________________________________________________________________________
TGLScaleManip::~TGLScaleManip() 
{
}

//______________________________________________________________________________
void TGLScaleManip::Draw(const TGLCamera & camera) const
{
   if (!fShape) {
      return;
   }

   const TGLBoundingBox & box = fShape->BoundingBox();
   Double_t widgetScale = DrawScale(box, camera);

   // Get permitted manipulations on shape
   TGLPhysicalShape::EManip manip = fShape->GetManip();

   TGLVector3 scaleAxes[3];
   for (UInt_t i = 0; i<3; i++) {
      if (box.IsEmpty()) {
         scaleAxes[i] = box.Axis(i, kTRUE)*widgetScale*-10.0;
      } else {
         scaleAxes[i] = box.Axis(i, kFALSE)*-0.51;
      }
   }

   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE);

   // Draw three axis widgets out of bounding box where permitted
   // Not drawing will prevent interaction
   // GL name loading for hit testing - 0 reserved for no selection
   if (manip & TGLPhysicalShape::kScaleX) {
      glPushName(1);
      DrawAxisWidget(kBox, widgetScale, box.Center(), scaleAxes[0], fSelectedWidget == 1 ? fgYellow : fgRed);
      glPopName();
   } else {
      DrawAxisWidget(kBox, widgetScale, box.Center(), scaleAxes[0], fgGrey);
   }
   if (manip & TGLPhysicalShape::kScaleY) {
      glPushName(2);
      DrawAxisWidget(kBox, widgetScale, box.Center(), scaleAxes[1], fSelectedWidget == 2 ? fgYellow : fgGreen);
      glPopName();
   } else {
      DrawAxisWidget(kBox, widgetScale, box.Center(), scaleAxes[1], fgGrey);
   }
   if (manip & TGLPhysicalShape::kScaleZ) {
      glPushName(3);
      DrawAxisWidget(kBox, widgetScale, box.Center(), scaleAxes[2], fSelectedWidget == 3 ? fgYellow : fgBlue);
      glPopName();
   } else {
      DrawAxisWidget(kBox, widgetScale, box.Center(), scaleAxes[2], fgGrey);
   }
   // Draw central origin sphere
   DrawOrigin(box.Center(), widgetScale/2.0, fgWhite);

   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
}
 
//______________________________________________________________________________
Bool_t TGLScaleManip::HandleButton(const Event_t * event, const TGLCamera & camera)
{
   if (event->fType == kButtonPress && fSelectedWidget != 0) {
      fStartScale = fShape->GetScale();
   }

   return TGLManip::HandleButton(event, camera);
}

//______________________________________________________________________________
Bool_t TGLScaleManip::HandleMotion(const Event_t * event, const TGLCamera & camera)
{
   if (fActive) {
      // Find mouse delta projected into world at attached object center
      TGLVector3 shift = camera.ViewportDeltaToWorld(fShape->BoundingBox().Center(), 
                                                     event->fX - fFirstMouse.GetX(),
                                                     -event->fY + fFirstMouse.GetY()); // Y inverted

      UInt_t axisIndex = fSelectedWidget - 1; // Ugg sort out axis / widget id mapping
      TGLVector3 widgetAxis = fShape->BoundingBox().Axis(axisIndex, kTRUE);

      // Scale by projected screen factor
      TGLVector3 screenScale = camera.ViewportDeltaToWorld(fShape->BoundingBox().Center(), 500, 500);
      Double_t factor = -5.0*Dot(shift, widgetAxis) / screenScale.Mag();

      TGLVector3 newScale = fStartScale;
      newScale[axisIndex] += factor;
      LimitScale(newScale[axisIndex]);
      fShape->Scale(newScale);

      fLastMouse.SetX(event->fX);
      fLastMouse.SetY(event->fY);

      return kTRUE;
   } else {
      return TGLManip::HandleMotion(event, camera);
   }
}

//______________________________________________________________________________
void TGLScaleManip::LimitScale(Double_t & factor) const
{
   if (factor < 1e-4) {
      factor = 1e-4;
   }
   if (factor > 1e+4) {
      factor = 1e+4;
   }
}
