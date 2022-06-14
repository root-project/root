// @(#)root/gl:$Id$
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

/** \class TGLTransManip
\ingroup opengl
Translation manipulator - attaches to physical shape and draws local
axes widgets with arrow heads. User can mouse over (turns yellow) and
L click/drag to translate along this axis.
Widgets use standard 3D package axes colours: X red, Y green, Z blue.
*/

ClassImp(TGLTransManip);

////////////////////////////////////////////////////////////////////////////////
/// Construct translation manipulator not bound to any physical shape.

TGLTransManip::TGLTransManip()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Construct translation manipulator, attached to supplied TGLViewer
/// 'viewer', bound to TGLPhysicalShape 'shape'.

TGLTransManip::TGLTransManip(TGLPhysicalShape * shape) :
   TGLManip(shape)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy the translation manipulator

TGLTransManip::~TGLTransManip()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draw translation manipulator - tubes with arrow heads, in local axes of
/// attached shape, in red(X), green(Y) and blue(Z), with white center sphere.
/// If selected widget (mouse over) this is drawn in active colour (yellow).

void TGLTransManip::Draw(const TGLCamera & camera) const
{
   if (!fShape) {
      return;
   }

   // Get draw scales
   const TGLBoundingBox & box = fShape->BoundingBox();
   Double_t baseScale;
   TGLVector3 axisScale[3];
   CalcDrawScale(box, camera, baseScale, axisScale);

   // Get permitted manipulations on shape
   TGLPhysicalShape::EManip manip = fShape->GetManip();

   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE);

   // Draw three axis widgets out of bounding box where permitted
   // Not drawing will prevent interaction
   // GL name loading for hit testing - 0 reserved for no selection
   if (manip & TGLPhysicalShape::kTranslateX) {
      glPushName(1);
      TGLUtil::DrawLine(box.Center(), axisScale[0], TGLUtil::kLineHeadArrow,
                        baseScale, ColorFor(1));
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), axisScale[0], TGLUtil::kLineHeadArrow,
                        baseScale, TGLUtil::fgGrey);
   }
   if (manip & TGLPhysicalShape::kTranslateY) {
      glPushName(2);
      TGLUtil::DrawLine(box.Center(), axisScale[1], TGLUtil::kLineHeadArrow,
                        baseScale, ColorFor(2));
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), axisScale[1], TGLUtil::kLineHeadArrow,
                        baseScale, TGLUtil::fgGrey);
   }
   if (manip & TGLPhysicalShape::kTranslateZ) {
      glPushName(3);
      TGLUtil::DrawLine(box.Center(), axisScale[2], TGLUtil::kLineHeadArrow,
                        baseScale, ColorFor(3));
      glPopName();
   } else {
      TGLUtil::DrawLine(box.Center(), axisScale[2], TGLUtil::kLineHeadArrow,
                        baseScale, TGLUtil::fgGrey);
   }
   // Draw white center sphere
   TGLUtil::DrawSphere(box.Center(), baseScale/2.0, TGLUtil::fgWhite);

   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse motion over manipulator - if active (selected
/// widget) translate physical along selected widget (axis) of the
/// manipulator, so it tracks mouse action. Returns kTRUE if redraw
/// required kFALSE otherwise.

Bool_t TGLTransManip::HandleMotion(const Event_t        & event,
                                   const TGLCamera      & camera)
{
   if (fActive) {
      // Find mouse delta projected into world at attached object center
      TGLVector3 shift =
         camera.ViewportDeltaToWorld( fShape->BoundingBox().Center(),
                                      event.fX - fLastMouse.GetX(),
                                     -event.fY + fLastMouse.GetY() ); // Y inverted

      // Now project this delta onto the current widget (axis) to give
      // a constrained shift along this
      UInt_t axisIndex = fSelectedWidget - 1; // Ugg sort out axis / widget id mapping
      TGLVector3 widgetAxis = fShape->BoundingBox().Axis(axisIndex, kTRUE);
      TGLVector3 constrainedShift = widgetAxis * Dot(shift, widgetAxis);
      fShape->Translate(constrainedShift);

      fLastMouse.SetX(event.fX);
      fLastMouse.SetY(event.fY);

      return kTRUE;
   }
   return kFALSE;
}

