// Author:  Richard Maunder  25/05/2005
// Parts taken from original by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

// CREDITS TO TIMUR!

#include "TGLOrthoCamera.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

#include "TMath.h"
#include "Riostream.h"

ClassImp(TGLOrthoCamera)

UInt_t   TGLOrthoCamera::fgZoomDeltaSens = 1000;
         
//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera(EType type) :
   fType(type)
{
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

//______________________________________________________________________________
TGLOrthoCamera::~TGLOrthoCamera()
{
}

//______________________________________________________________________________
void TGLOrthoCamera::Setup(const TGLBoundingBox & box)
{
   fCenter = box.Center();
   
   static const Double_t rotMatrixXOY[] = {1., 0.,  0., 0., 
                                           0., 1.,  0., 0.,
                                           0., 0.,  1., 0., 
                                           0., 0.,  0., 1.};
   static const Double_t rotMatrixYOZ[] = {1.,  0.,  0.,  0., 
                                           0.,  0., -1.,  0.,
                                           0.,  1.,  0.,  0.,  
                                           0.,  0.,  0.,  1.};
   static const Double_t rotMatrixXOZ[] = { 0.,  0.,  1.,  0., 
                                            0.,  1.,  0.,  0.,
                                           -1.,  0.,  0.,  0., 
                                            0.,  0.,  0.,  1.};
            
   switch (fType) {
      case (kXOY): {
         fWidth = box.XMax() - box.XMin();
         fHeight = box.YMax() - box.YMin();
         fMatrix.Set(rotMatrixXOY);
         break;
      }
      case (kYOZ): {
         fWidth = box.YMax() - box.YMin();
         fHeight = box.ZMax() - box.ZMin();
         fMatrix.Set(rotMatrixYOZ);
         break;
      }
      case (kXOZ): {
         fWidth = box.XMax() - box.XMin();
         fHeight = box.ZMax() - box.ZMin();
         fMatrix.Set(rotMatrixXOZ);
         break;
      }
   }
   fVolumeDiag = box.Extents().Mag();
   fZoomMin = 0.5; 
   fZoomDefault = 0.95;
   fZoomMax = 30.0;  
   Reset();
}

//______________________________________________________________________________
void TGLOrthoCamera::Reset()
{
   fTruck.Set(-fCenter.X(), -fCenter.Y(), -fCenter.Z());
   fZoom   = fZoomDefault;
   fCacheDirty = kTRUE;
}
   
//______________________________________________________________________________
Bool_t TGLOrthoCamera::Dolly(Int_t delta)
{
   return Zoom(delta);
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Zoom (Int_t delta)
{
   Double_t shift = static_cast<Double_t>(delta) / fgZoomDeltaSens;
    
   fZoom += shift * (fZoomMax - fZoomMin);
   if (fZoom < fZoomMin) {
      fZoom = fZoomMin;
   }
   if (fZoom > fZoomMax) {
      fZoom = fZoomMax;
   }
   
   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta)
{
   //TODO: Convert TGLRect so this not required
   GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
   TGLVertex3 start, end;   
   gluUnProject(x, y, 0.0, fModVM.CArr(), fProjM.CArr(), viewport, &start.X(), &start.Y(), &start.Z());
   gluUnProject(x + xDelta, y + yDelta, 0.0, fModVM.CArr(), fProjM.CArr(), viewport, &end.X(), &end.Y(), &end.Z());
   fTruck = fTruck + (end - start);
   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Rotate(Int_t /*xDelta*/, Int_t /*yDelta*/)
{
   // Not allowed at present - could let the user or external code create non-axis
   // ortho projects by adjusting H/V rotations 
   return kFALSE;
}

//______________________________________________________________________________
void TGLOrthoCamera::Apply(const TGLBoundingBox & /*box*/, const TGLRect * pickRect)
{
   glViewport(fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height());   

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // Load up any picking rect
   if (pickRect) {
      //TODO: Convert TGLRect so this not required
      GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
      gluPickMatrix(pickRect->X(), pickRect->Y(), 
                    pickRect->Width(), pickRect->Height(),
                    viewport);
   }

   if(fViewport.Width() == 0 || fViewport.Height() == 0) {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
   } else {
      Double_t biggest = fWidth > fHeight ? fWidth:fHeight;
      glOrtho(-biggest/2.0, biggest/2.0, -biggest/2.0, biggest/2.0, fVolumeDiag, 3.0*fVolumeDiag);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslated(0.0, 0.0, -2.0*fVolumeDiag);
      glScaled(fZoom, fZoom*fViewport.Aspect(), 1.0);
      glMultMatrixd(fMatrix.CArr());
      glTranslated(fTruck.X(), fTruck.Y(), fTruck.Z());
   }

   if (fCacheDirty) { 
      UpdateCache();
   }
}



