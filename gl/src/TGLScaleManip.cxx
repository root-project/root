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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLScaleManip                                                        //
//                                                                      //
// Scale manipulator - attaches to physical shape and draws local axes  //
// widgets with box heads. User can mouse over (turns yellow) and L     //
// click/drag to scale along this axis.                                 // 
// Widgets use standard 3D package axes colours: X red, Y green, Z blue.//
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLScaleManip)

//______________________________________________________________________________
TGLScaleManip::TGLScaleManip(TGLViewer & viewer) : TGLManip(viewer)
{
   // Construct scale manipulator, attached to supplied TGLViewer 
   // 'viewer', not bound to any physical shape.
}

//______________________________________________________________________________
TGLScaleManip::TGLScaleManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   TGLManip(viewer, shape) 
{
   // Construct scale manipulator, attached to supplied TGLViewer 
   // 'viewer', bound to TGLPhysicalShape 'shape'.
}

//______________________________________________________________________________
TGLScaleManip::~TGLScaleManip() 
{
   // Destory the scale manipulator
}

//______________________________________________________________________________
void TGLScaleManip::Draw(const TGLCamera & camera) const
{
   // Draw scale manipulator - tubes with box heads, in local axes of 
   // attached shape, in red(X), green(Y) and blue(Z), with white center sphere. 
   // If selected widget (mouse over) this is drawn in active colour (yellow).
   if (!fShape) {
      return;
   }

   const TGLBoundingBox & box = fShape->BoundingBox();
   Double_t widgetScale = CalcDrawScale(box, camera);

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
      TGLUtil::DrawLine(box.Center(), scaleAxes[0], TGLUtil::kLineHeadBox, 
                        widgetScale, fSelectedWidget == 1 ? fgYellow : fgRed);
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), scaleAxes[0], TGLUtil::kLineHeadBox, 
                        widgetScale, fgGrey);
   }
   if (manip & TGLPhysicalShape::kScaleY) {
      glPushName(2);
      TGLUtil::DrawLine(box.Center(), scaleAxes[1], TGLUtil::kLineHeadBox, 
                        widgetScale, fSelectedWidget == 2 ? fgYellow : fgGreen);
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), scaleAxes[1], TGLUtil::kLineHeadBox, 
                        widgetScale, fgGrey);
   }
   if (manip & TGLPhysicalShape::kScaleZ) {
      glPushName(3);
      TGLUtil::DrawLine(box.Center(), scaleAxes[2], TGLUtil::kLineHeadBox, 
                        widgetScale, fSelectedWidget == 3 ? fgYellow : fgBlue);
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), scaleAxes[2], TGLUtil::kLineHeadBox, 
                        widgetScale, fgGrey);
   }
   // Draw white center sphere
   TGLUtil::DrawSphere(box.Center(), widgetScale/2.0, fgWhite);

   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
}
 
//______________________________________________________________________________
Bool_t TGLScaleManip::HandleButton(const Event_t * event, const TGLCamera & camera)
{
   // Handle mouse button event over manipulator - returns kTRUE if redraw required 
   // kFALSE otherwise.
   if (event->fType == kButtonPress && fSelectedWidget != 0) {
      fStartScale = fShape->GetScale();
   }

   return TGLManip::HandleButton(event, camera);
}

//______________________________________________________________________________
Bool_t TGLScaleManip::HandleMotion(const Event_t * event, const TGLCamera & camera)
{
   // Handle mouse motion over manipulator - if active (selected widget) scale 
   // physical along selected widget (axis) of the manipulator, so it tracks mouse 
   // action. Returns kTRUE if redraw required kFALSE otherwise.
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
   // Clamp scale to sizable values: 1000 - 1/1000
   // Guards against div by zero problems.
   if (factor < 1e-4) {
      factor = 1e-4;
   }
   if (factor > 1e+4) {
      factor = 1e+4;
   }
}
