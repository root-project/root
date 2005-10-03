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
void TGLTransManip::Draw() const
{
   TGLManip::DrawAxisWidgets(kArrow);
}

//______________________________________________________________________________
Bool_t TGLTransManip::HandleMotion(Event_t * event, const TGLCamera & camera)
{
   if (fActive) {
      // Find mouse delta projected into world at attached object center
      TGLVector3 shift = camera.ProjectedShift(fShape->BoundingBox().Center(), 
                                               event->fX - fLastMouseX,
                                               -event->fY + fLastMouseY); // Y inverted
      
      // Now project this delta onto the current widget (axis) to give
      // a constrained shift along this
      UInt_t axisIndex = fSelectedWidget - 1; // Ugg sort out axis / widget id mapping
      TGLVector3 widgetAxis = fShape->BoundingBox().Axis(axisIndex, kTRUE);
      TGLVector3 constrainedShift = widgetAxis * Dot(shift, widgetAxis);
      fShape->Shift(constrainedShift);

      fLastMouseX = event->fX;
      fLastMouseY = event->fY;

      return kTRUE;
   } else {
      return TGLManip::HandleMotion(event, camera);
   }
}

