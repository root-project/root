// @(#)root/gl:$Name:  $:$Id: TGLOrthoCamera.cxx,v 1.13 2006/01/26 11:59:41 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TGLOrthoCamera.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

#include "TMath.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOrthoCamera                                                       //
//                                                                      //
// Orthographic projection camera. Currently limited to three types     //
// defined at construction time - kXOY, kXOZ, kZOY - where this refers  //
// to the viewport plane axis - e.g. kXOY has X axis horizontal, Y      //
// vertical - i.e. looking down Z axis with Y vertical.                 //
//
// The plane types restriction could easily be removed to supported     //
// arbitary ortho projections along any axis/orientation with free      //
// rotations about them.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLOrthoCamera)

UInt_t   TGLOrthoCamera::fgZoomDeltaSens = 500;

//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera(EType type) :
   fType(type), fZoomMin(0.01), fZoomDefault(0.78), fZoomMax(1000.0), 
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0), fTruck(0.0, 0.0, 0.0), fMatrix()
{
   // Construct orthographic camera with 'type' defining fixed view direction
   // & orientation (in world frame):
   //
   // kXOY : X Horz. / Y Vert (looking towards +Z, Y up)
   // kXOZ : X Horz. / Z Vert (looking towards +Y, Z up)
   // kZOY : Z Horz. / Y Vert (looking towards +X, Y up)
   // 
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

//______________________________________________________________________________
TGLOrthoCamera::~TGLOrthoCamera()
{
   // Destroy orthographic camera
}

//______________________________________________________________________________
void TGLOrthoCamera::Setup(const TGLBoundingBox & box, Bool_t reset)
{
   // Setup camera limits suitible to view the world volume defined by 'box'
   // and call Reset() to initialise camera.
   static const Double_t rotMatrixXOY[] = { 1.,  0.,  0.,  0.,
                                            0.,  1.,  0.,  0.,
                                            0.,  0.,  1.,  0.,
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
   if (reset)
      Reset();
}

//______________________________________________________________________________
void TGLOrthoCamera::Reset()
{
   // Reset the camera to defaults - trucking, zooming to reframe the world volume
   // established in Setup(). Note: limits defined in Setup() are not adjusted.
   fTruck.Set(0.0, 0.0, 0.0);
   fZoom   = fZoomDefault;
   fCacheDirty = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Dolly the camera - 'move camera along eye line, retaining lens focal length'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive dolly in, -ive dolly out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // For an orthographic camera dollying and zooming are identical and both equate 
   // logically to a rescaling of the viewport limits - without center shift. 
   // There is no perspective foreshortening or lens 'focal length'.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   // TODO: Bring all mouse handling into camera classes - would simplify interface and
   // remove these non-generic cases.
   return Zoom(delta, mod1, mod2);
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Zoom (Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Zoom the camera - 'adjust lens focal length, retaining camera position'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive zoom in, -ive zoom out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // For an orthographic camera dollying and zooming are identical and both equate 
   // logically to a rescaling of the viewport limits - without center shift. 
   // There is no perspective foreshortening or lens 'focal length'.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   // TODO: Bring all mouse handling into camera classes - would simplify interface and
   // remove these non-generic cases.
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
   // Truck the camera - 'move camera parallel to film plane'. The film 
   // plane is defined by the EyePoint() / EyeDirection() pair. Define motion 
   // using center point (x/y) and delta (xDelta/yDelta) - the mouse motion. 
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   //
   // Note: Trucking is often mistakenly refered to as 'pan' or 'panning'. 
   // Panning is swivelling the camera on it's own axis - the eye point.
   
   //TODO: Convert TGLRect so this not required
   GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
   TGLVertex3 start, end;
   // Trucking done at near clipping plane
   gluUnProject(x, y, 0.0, fModVM.CArr(), fProjM.CArr(), viewport, &start.X(), &start.Y(), &start.Z());
   gluUnProject(x + xDelta, y + yDelta, 0.0, fModVM.CArr(), fProjM.CArr(), viewport, &end.X(), &end.Y(), &end.Z());
   fTruck = fTruck + (end - start);
   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Rotate(Int_t /*xDelta*/, Int_t /*yDelta*/)
{
   // Rotate the camera - 'swivel round the view volume center'.
   // Ignored at present for orthographic cameras - have a fixed direction. 
   // Could let the user or external code create non-axis
   // ortho projects by adjusting H/V rotations in future.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   return kFALSE;
}

//______________________________________________________________________________
void TGLOrthoCamera::Apply(const TGLBoundingBox & /*box*/, const TGLRect * pickRect) const
{
   // Apply the camera to the current GL context, setting the viewport, projection
   // and modelview matricies. After this verticies etc can be directly entered
   // in the world frame. This also updates the cached frustum values, enabling
   // all the projection, overlap tests etc defined in TGLCamera to be used.
   // 
   // Arguments are:
   // 'box' - view volume box - ignored for ortho camera. Assumed to be same
   // as one passed to Setup().
   // 'pickRect' - optional picking rect. If non-null, restrict drawing to this
   // viewport rect.
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
   Double_t halfDepth = extents.Mag();
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

//______________________________________________________________________________
void TGLOrthoCamera::Configure(Double_t left, Double_t right, 
                               Double_t top, Double_t bottom)
{
   // Configure the camera state
   Double_t width = right - left;
   Double_t height = top - bottom;

   Double_t xZoom = width/fVolume.Extents().X();
   Double_t yZoom = height/fVolume.Extents().Y();

   fZoom = (xZoom > yZoom) ? xZoom : yZoom;

   // kXOY : X Horz. / Y Vert (looking towards +Z, Y up)
   // kXOZ : X Horz. / Z Vert (looking towards +Y, Z up)
   // kZOY : Z Horz. / Y Vert (looking towards +X, Y up)
   if (fType == kXOY) {
      fTruck.X() = right - left;
      fTruck.Y() = top - bottom;
   } else if (fType == kXOZ) {
      fTruck.X() = right - left;
      fTruck.Z() = top - bottom;
   } else if (fType == kZOY) {
      fTruck.Z() = right - left;
      fTruck.Y() = top - bottom;
   }
   fCacheDirty = kTRUE;
}
