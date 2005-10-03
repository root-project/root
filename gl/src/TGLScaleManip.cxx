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
#include "TError.h"

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
void TGLScaleManip::Draw() const
{
   TGLManip::DrawAxisWidgets(kBox);
}
 
//______________________________________________________________________________
Bool_t TGLScaleManip::HandleButton(Event_t * event)
{
   if (event->fType == kButtonPress && fSelectedWidget != 0) {
      fStartScale = fShape->Scale();
   }

   return TGLManip::HandleButton(event);
}

//______________________________________________________________________________
Bool_t TGLScaleManip::HandleMotion(Event_t * event, const TGLCamera & camera)
{
   if (fActive) {
      // Find mouse delta projected into world at attached object center
      TGLVector3 shift = camera.ProjectedShift(fShape->BoundingBox().Center(), 
                                               event->fX - fFirstMouseX,
                                               -event->fY + fFirstMouseY); // Y inverted

      TGLVector3 screenSize = camera.ProjectedShift(fShape->BoundingBox().Center(), 500, 500);
      UInt_t axisIndex = fSelectedWidget - 1; // Ugg sort out axis / widget id mapping
      TGLVector3 widgetAxis = fShape->BoundingBox().Axis(axisIndex, kTRUE);
      Double_t factor = -5.0*Dot(shift, widgetAxis) / screenSize.Mag();
      TGLVector3 newScale = fStartScale;
      newScale[axisIndex] += factor;
      LimitScale(newScale[axisIndex]);
      TGLVector3 oldE = fShape->BoundingBox().Extents();
//      Info("Old Extents", " (%f,%f,%f)", oldE.X(), oldE.Y(), oldE.Z());
//      Info("Scale", "%d by factor %f -> %f", axisIndex, factor, newScale[axisIndex]);
      fShape->SetScale(newScale);
      TGLVector3 newE = fShape->BoundingBox().Extents();
//      Info("New Extents", " (%f,%f,%f)", newE.X(), newE.Y(), newE.Z());

      fLastMouseX = event->fX;
      fLastMouseY = event->fY;

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
