// @(#)root/gl:$Name:  $:$Id: TGLOrthoCamera.cxx,v 1.5 2005/06/21 16:54:17 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLOrthoCamera.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

#include "TGLQuadric.h" // REmove

#include "TMath.h"
#include "Riostream.h"

ClassImp(TGLOrthoCamera)

UInt_t   TGLOrthoCamera::fgZoomDeltaSens = 500;

//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera(EType type) :
   fType(type), fZoomMin(0.01), fZoomDefault(0.9), fZoomMax(1000.0), 
	fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
	fZoom(1.0), fTruck(0.0, 0.0, 0.0), fMatrix()
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
   static const Double_t rotMatrixXOY[] = { 1.,  0.,  0.,  0.,
                                            0., -1.,  0.,  0.,
                                            0.,  0., -1.,  0.,
                                            0.,  0.,  0.,  1. };

   static const Double_t rotMatrixXOZ[] = { 1.,  0.,  0.,  0.,
                                            0.,  0., -1.,  0.,
                                            0.,  1.,  0.,  0.,
                                            0.,  0.,  0.,  1. };

   static const Double_t rotMatrixZOY[] = { 0.,  0.,  -1.,  0.,
                                            0.,  1.,  0.,  0.,
                                            1.,  0.,  0.,  0.,
                                            0.,  0.,  0.,  1. };
	
   switch (fType) {
		// Looking down Z axis, X horz, Y vert
      case (kXOY): {
         // X -> X
         // Y -> Y
         // Z -> Z
         fVolume = box;
         fMatrix.Set(rotMatrixXOY);
         break;
      }
		// Looking down Y axis, X horz, Z vert
      case (kXOZ): {
         // X -> X
         // Z -> Y
         // Y -> Z
         fVolume.SetAligned(TGLVertex3(box.XMin(), box.ZMin(), box.YMin()), 
                            TGLVertex3(box.XMax(), box.ZMax(), box.YMax()));
         fMatrix.Set(rotMatrixXOZ);
         break;
      }
		// Looking down X axis, Z horz, Y vert
      case (kZOY): {
         // Z -> X
         // Y -> Y
         // X -> Z
         fVolume.SetAligned(TGLVertex3(box.ZMin(), box.YMin(), box.XMin()), 
                            TGLVertex3(box.ZMax(), box.YMax(), box.XMax()));
         fMatrix.Set(rotMatrixZOY);
         break;
      }
   }
   Reset();
}

//______________________________________________________________________________
void TGLOrthoCamera::Reset()
{
   fTruck.Set(0.0, 0.0, 0.0);
   fZoom   = fZoomDefault;
   fCacheDirty = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   return Zoom(delta, mod1, mod2);
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Zoom (Int_t delta, Bool_t mod1, Bool_t mod2)
{
   if (AdjustAndClampVal(fZoom, fZoomMin, fZoomMax, -delta*2, fgZoomDeltaSens, mod1, mod2))
   {
      fCacheDirty = kTRUE;
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
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
      return;
   }
   
   TGLVector3 extents = fVolume.Extents();
   Double_t width = extents.X();
   Double_t height = extents.Y();
   Double_t halfRange;
   if (width > height) {
      halfRange = width / 2.0;
   } else {
      halfRange = height / 2.0;
   }
   halfRange /= fZoom;

   // For near/far clipping half depth give extra slack so clip objects/manips 
   // are visible 
   Double_t halfDepth = extents.Z();
   const TGLVertex3 & center = fVolume.Center();

   glOrtho(center.X() - halfRange, 
           center.X() + halfRange, 
           center.Y() - halfRange, 
           center.Y() + halfRange, 
           center.Z() - halfDepth, 
           center.Z() + halfDepth);


   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

	glScaled(1.0 / fViewport.Aspect(), 1.0, 1.0); 	

   // Debug aid - show current volume
   /*glDisable(GL_LIGHTING);
   glColor3d(0.0, 0.0, 1.0);
   fVolume.Draw();
   glEnable(GL_LIGHTING);*/

   glMultMatrixd(fMatrix.CArr());
   glTranslated(fTruck.X(), fTruck.Y(), fTruck.Z());

   if (fCacheDirty) {
      UpdateCache();
   }
}
