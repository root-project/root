// @(#)root/gl:$Name:  $:$Id: TGLRotateManip.cxx
// Author:  Richard Maunder  04/10/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLRotateManip.h"
#include "TGLPhysicalShape.h"
#include "TGLCamera.h"
#include "TGLIncludes.h"
#include "TMath.h"
#include "TError.h"

//______________________________________________________________________________
TGLRotateManip::TGLRotateManip(TGLViewer & viewer) : 
   TGLManip(viewer),
   fRingLine(TGLVertex3(0.0, 0.0, 0.0), TGLVertex3(0.0, 0.0, 0.0)),
   fRingLineOld(TGLVertex3(0.0, 0.0, 0.0), TGLVertex3(0.0, 0.0, 0.0)),
   fDebugProj(TGLVertex3(0.0, 0.0, 0.0), TGLVertex3(0.0, 0.0, 0.0))
{
}

//______________________________________________________________________________
TGLRotateManip::TGLRotateManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   TGLManip(viewer, shape),
   fRingLine(TGLVertex3(0.0, 0.0, 0.0), TGLVertex3(0.0, 0.0, 0.0)),
   fRingLineOld(TGLVertex3(0.0, 0.0, 0.0), TGLVertex3(0.0, 0.0, 0.0)),
   fDebugProj(TGLVertex3(0.0, 0.0, 0.0), TGLVertex3(0.0, 0.0, 0.0))
{
}

//______________________________________________________________________________
TGLRotateManip::~TGLRotateManip() 
{
}
   
//______________________________________________________________________________
void TGLRotateManip::Draw(const TGLCamera & camera) const
{
   if (!fShape) {
      return;
   }

   const TGLBoundingBox & box = fShape->BoundingBox();
   Double_t widgetScale = DrawScale(box, camera);

   // Get permitted manipulations on shape
   TGLPhysicalShape::EManip manip = fShape->GetManip();

   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE);

   // Draw three axis rings where permitted
   // Not drawing will prevent interaction
   // GL name loading for hit testing - 0 reserved for no selection
   if (manip & TGLPhysicalShape::kRotateX) {
      glPushName(1);
      DrawAxisRing(box.Center(), box.Axis(0, kTRUE), widgetScale*10.04, 
                   fSelectedWidget == 1 ? fgYellow : fgRed);
      glPopName();
   } else {
      DrawAxisRing(box.Center(), box.Axis(0, kTRUE), widgetScale*10.04, fgGrey);
   }
   if (manip & TGLPhysicalShape::kRotateY) {
      glPushName(2);
      DrawAxisRing(box.Center(), box.Axis(1, kTRUE), widgetScale*10.02, 
                   fSelectedWidget == 2 ? fgYellow : fgGreen);
      glPopName();
   } else {
      DrawAxisRing(box.Center(), box.Axis(1, kTRUE), widgetScale*10.02, fgGrey);
   }
   if (manip & TGLPhysicalShape::kRotateZ) {
      glPushName(3);
      DrawAxisRing(box.Center(), box.Axis(2, kTRUE), widgetScale*10.00, 
                   fSelectedWidget == 3 ? fgYellow : fgBlue);
      glPopName();
   } else {
      DrawAxisRing(box.Center(), box.Axis(2, kTRUE), widgetScale*10.00, fgGrey);
   }
   // Draw central origin sphere
   DrawOrigin(box.Center(), widgetScale/2.0, fgWhite);

   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
}

//______________________________________________________________________________
void TGLRotateManip::DrawAxisRing(const TGLVertex3 & origin, const TGLVector3 & axis, 
                                  Double_t radius, Float_t rgba[4]) const
{    
   SetDrawColors(rgba);

   Double_t inner = radius;
   Double_t width = radius*0.05;
   Double_t outer = inner+width;

   glPushMatrix();
   TGLMatrix local(origin, axis);
   glMultMatrixd(local.CArr());

   glTranslated(0.0, 0.0, -width/2.0);

   glDisable(GL_CULL_FACE);
   gluCylinder(fgQuad.Get(), inner, inner, width, fgQuality, fgQuality);
   gluCylinder(fgQuad.Get(), outer, outer, width, fgQuality, fgQuality);
   gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(fgQuad.Get(), inner, outer, fgQuality, fgQuality); 
   glTranslated(0.0, 0.0, width);
   gluQuadricOrientation(fgQuad.Get(), (GLenum)GLU_OUTSIDE);
   gluDisk(fgQuad.Get(), inner, outer, fgQuality, fgQuality); 
   glEnable(GL_CULL_FACE);

   glPopMatrix();
}

//______________________________________________________________________________
Bool_t TGLRotateManip::HandleButton(const Event_t * event, const TGLCamera & camera)
{
   Bool_t captured = TGLManip::HandleButton(event, camera);

   if (captured) {
      fRingLineOld = fRingLine = CalculateRingLine(fLastMouse, camera);
   }

   return captured;
}

//______________________________________________________________________________
Bool_t TGLRotateManip::HandleMotion(const Event_t * event, const TGLCamera & camera)
{
   if (fActive) {
      fLastMouse.SetX(event->fX);
      fLastMouse.SetY(event->fY);

      fRingLineOld = fRingLine;
      fRingLine = CalculateRingLine(fLastMouse, camera);

      TGLVector3 widgetAxis = fShape->BoundingBox().Axis(fSelectedWidget - 1, kTRUE); // Ugg sort out axis / widget id mapping
      TGLVertex3 ringCenter = fShape->BoundingBox().Center();

      // Calculate singed angle delta between old and new ring position using
      Double_t angle = Angle(fRingLineOld.Vector(), fRingLine.Vector(), widgetAxis);
      fShape->Rotate(ringCenter, widgetAxis, angle);

      return kTRUE;
   } else {
      return TGLManip::HandleMotion(event, camera);
   }
}

//______________________________________________________________________________
TGLLine3 TGLRotateManip::CalculateRingLine(const TPoint & mouse, const TGLCamera & camera) const
{
   // Find active selected axis
   UInt_t axisIndex = fSelectedWidget - 1; // Ugg sort out axis / widget id mapping
   TGLVector3 widgetAxis = fShape->BoundingBox().Axis(axisIndex, kTRUE);

   // Construct plane for the axis ring, using normal and center point
   TGLVertex3 ringCenter = fShape->BoundingBox().Center();
   TGLPlane ringPlane(widgetAxis, ringCenter);

   // Find mouse position in viewport coords
   TPoint mouseViewport(mouse);
   camera.WindowToViewport(mouseViewport);

   // Find projection of mouse into world
   TGLLine3 viewportProjection = camera.ViewportToWorld(mouseViewport);
   fDebugProj = viewportProjection;

   // Find rotation line from ring center to this intersection on plane
   std::pair<Bool_t, TGLVertex3> ringPlaneInter =  ringPlane.Intersection(viewportProjection);
   if (!ringPlaneInter.first) {
      Error("TGLRotateManip::CalculateRingLine", "projection on ring plane failed");
   }
   return TGLLine3(ringCenter, ringPlaneInter.second);
}

