// @(#)root/gl:$Name:  $:$Id: TGLTransManip.cxx
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLTransManip.h"
#include "TGLPhysicalShape.h"
#include "TGLCamera.h"
#include "TGLIncludes.h"

ClassImp(TGLTransManip)

//______________________________________________________________________________
TGLTransManip::TGLTransManip(TGLViewer & viewer) : TGLManip(viewer)
{
}

//______________________________________________________________________________
TGLTransManip::TGLTransManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   TGLManip(viewer, shape) 
{
}

//______________________________________________________________________________
TGLTransManip::~TGLTransManip() 
{
}
   
//______________________________________________________________________________
void TGLTransManip::Draw(const TGLCamera & camera) const
{
   if (!fShape) {
      return;
   }

   const TGLBoundingBox & box = fShape->BoundingBox();
   Double_t widgetScale = CalcDrawScale(box, camera);

   // Get permitted manipulations on shape
   TGLPhysicalShape::EManip manip = fShape->GetManip();

   TGLVector3 translateAxes[3];
   for (UInt_t i = 0; i<3; i++) {
      if (box.IsEmpty()) {
         translateAxes[i] = box.Axis(i, kTRUE)*widgetScale*-10.0;
      } else {
         translateAxes[i] = box.Axis(i, kFALSE)*-0.51;
      }
   }

   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE);

   // Draw three axis widgets out of bounding box where permitted
   // Not drawing will prevent interaction
   // GL name loading for hit testing - 0 reserved for no selection
   if (manip & TGLPhysicalShape::kTranslateX) {
      glPushName(1);
      TGLUtil::DrawLine(box.Center(), translateAxes[0], TGLUtil::kLineHeadArrow, 
                        widgetScale, fSelectedWidget == 1 ? fgYellow : fgRed);
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), translateAxes[0], TGLUtil::kLineHeadArrow, 
                        widgetScale, fgGrey);
   }
   if (manip & TGLPhysicalShape::kTranslateY) {
      glPushName(2);
      TGLUtil::DrawLine(box.Center(), translateAxes[1], TGLUtil::kLineHeadArrow, 
                        widgetScale, fSelectedWidget == 2 ? fgYellow : fgGreen);
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), translateAxes[1], TGLUtil::kLineHeadArrow, 
                        widgetScale, fgGrey);
   }
   if (manip & TGLPhysicalShape::kTranslateZ) {
      glPushName(3);
      TGLUtil::DrawLine(box.Center(), translateAxes[2], TGLUtil::kLineHeadArrow, 
                        widgetScale, fSelectedWidget == 3 ? fgYellow : fgBlue);
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), translateAxes[2], TGLUtil::kLineHeadArrow, 
                        widgetScale, fgGrey);
   }
   // Draw white center sphere
   TGLUtil::DrawSphere(box.Center(), widgetScale/2.0, fgWhite);

   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
}

//______________________________________________________________________________
Bool_t TGLTransManip::HandleMotion(const Event_t * event, const TGLCamera & camera)
{
   if (fActive) {
      // Find mouse delta projected into world at attached object center
      TGLVector3 shift = camera.ViewportDeltaToWorld(fShape->BoundingBox().Center(), 
                                                     event->fX - fLastMouse.GetX(),
                                                     -event->fY + fLastMouse.GetY()); // Y inverted
      
      // Now project this delta onto the current widget (axis) to give
      // a constrained shift along this
      UInt_t axisIndex = fSelectedWidget - 1; // Ugg sort out axis / widget id mapping
      TGLVector3 widgetAxis = fShape->BoundingBox().Axis(axisIndex, kTRUE);
      TGLVector3 constrainedShift = widgetAxis * Dot(shift, widgetAxis);
      fShape->Translate(constrainedShift);

      fLastMouse.SetX(event->fX);
      fLastMouse.SetY(event->fY);

      return kTRUE;
   } else {
      return TGLManip::HandleMotion(event, camera);
   }
}

