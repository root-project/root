/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TEveLegoEventHandler.h"
#include "TEveCaloLegoGL.h"

#include "TGLViewer.h"
#include "TGLWidget.h"
#include "TGLOverlay.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLCamera.h"
#include "TGLPerspectiveCamera.h"
#include "TGLOrthoCamera.h"
#include "KeySymbols.h"

#include "TMath.h"
#include "TGLUtil.h"
#include "TEveTrans.h"

#include "TEveCalo.h"


//==============================================================================
//==============================================================================
// TEveLegoEventHandler
//==============================================================================

ClassImp(TEveLegoEventHandler);

//______________________________________________________________________________
//
// A base class of TGLEventHandler. Switches current camera from perspective
// to orthographic bird-view, if camera theta is less than given threshold. It sets back
// perspective camera when accumulated angle is more than transition theta.
//

//______________________________________________________________________________
TEveLegoEventHandler::TEveLegoEventHandler(TGWindow *w, TObject *obj, TEveCaloLego *lego):
   TGLEventHandler(w, obj),

   fMode(kFree),
   fTransTheta(0.5f),
   fTheta(0.f),

   fLego(lego)
{
   // Constructor.
}

//______________________________________________________________________________
Bool_t TEveLegoEventHandler::HandleKey(Event_t *event)
{
   // Virtual from TGLEventHandler.
   // Free the camera when home is pressed.

   if (event->fCode == kKey_Home)
      fMode = kFree;

   return TGLEventHandler::HandleKey(event);
}

//______________________________________________________________________________
Bool_t TEveLegoEventHandler::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Method to handle action TGLViewer::kDragCameraRotate. It switches from standard perspective
   // view to bird-view bellow angle fTransTheta and restores view when accumulated theta is larger
   // than transition angle.

   if ( !fLego ) return TGLEventHandler::Rotate(xDelta, yDelta, mod1, mod2);

   TGLCamera &cam =  fGLViewer->GetRnrCtx()->RefCamera();
   Double_t hRotate = cam.AdjustDelta(-yDelta, TMath::Pi()/cam.RefViewport().Height(), mod1, mod2);

   // get lego bounding box
   Float_t *bb = fLego->AssertBBox();
   TGLBoundingBox box;
   box.SetAligned(TGLVertex3(bb[0], bb[2], bb[4]), TGLVertex3(bb[1], bb[3], bb[5]));
   box.Transform(fLego->RefMainTrans().Array());

   Bool_t camChanged = kFALSE;

   if (cam.IsOrthographic())
   {
      fTheta += hRotate;
      if (fTheta < 0) fTheta = 0;
      if (fTheta > fTransTheta)
      {
         TGLCamera* ortho = &cam;
         Double_t l = -ortho->FrustumPlane(TGLCamera::kLeft).D();
         Double_t r =  ortho->FrustumPlane(TGLCamera::kRight).D();
         Double_t t =  ortho->FrustumPlane(TGLCamera::kTop).D();
         Double_t b = -ortho->FrustumPlane(TGLCamera::kBottom).D();

         fGLViewer->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
         TGLPerspectiveCamera* persp = dynamic_cast<TGLPerspectiveCamera*>(&fGLViewer->GetRnrCtx()->RefCamera());
         persp->Setup(box, kTRUE);

         TGLVector3 extents = box.Extents();
         Int_t sortInd[3];
         TMath::Sort(3, extents.CArr(), sortInd);
         Double_t size = TMath::Hypot(extents[sortInd[0]], extents[sortInd[1]]);
         Double_t dolly  = size / (2.0*TMath::Tan(30*TMath::Pi()/360));
         Double_t fov = TMath::ATan(TMath::Hypot(t-b, r-l)/(2*dolly));

         persp->SetCenterVecWarp(0.5*(l+r), 0.5*(t+b), 0);

         Double_t vR =  -0.5 * TMath::Pi(); // switch XY
         Double_t hR =  -0.5 * TMath::Pi() + fTransTheta; // fix top view angle
         persp->Configure(fov*TMath::RadToDeg(), 0, 0, hR, vR);

         fMode = kFree;
         camChanged = kTRUE;
      }
   }
   else
   {
      Double_t theta  = cam.GetTheta();
      Double_t thetaN = theta + hRotate;
      if (thetaN > TMath::Pi() - cam.GetVAxisMinAngle()) thetaN = TMath::Pi() - cam.GetVAxisMinAngle();
      else if (thetaN < cam.GetVAxisMinAngle())   thetaN = cam.GetVAxisMinAngle();

      fTheta = thetaN;

      if (thetaN < fTransTheta)
      {
         TGLPerspectiveCamera* persp =  (TGLPerspectiveCamera*)(&cam);
         fGLViewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
         TGLOrthoCamera* ortho = dynamic_cast<TGLOrthoCamera*>(& fGLViewer->GetRnrCtx()->RefCamera());
         ortho->Setup(box,  kTRUE);

         // translation to the plane intersect
         const TGLMatrix& mx =  cam.GetCamBase() * cam.GetCamTrans();
         TGLVertex3 d   = mx.GetTranslation();
         TGLVertex3 p = d + mx.GetBaseVec(1);
         TGLLine3  line(d, p);
         const TGLPlane rp = TGLPlane(cam.GetCamBase().GetBaseVec(3), TGLVertex3());
         std::pair<Bool_t, TGLVertex3> intersection;
         intersection = Intersection(rp, line, kTRUE);
         TGLVertex3 v = intersection.second;
         ortho->Truck( v.X() - box.Center().X(), v.Y() - box.Center().Y());

         // zoom
         Double_t t =  persp->FrustumPlane(TGLCamera::kTop).D();
         Double_t b = -persp->FrustumPlane(TGLCamera::kBottom).D();
         Double_t zoom = box.Extents().Y()/(t-b);
         ortho->Configure(zoom, 0, 0, 0, 0);

         fMode = kLocked;
         camChanged = kTRUE;
      }
      else
      {
         camChanged = fGLViewer->CurrentCamera().Rotate(xDelta, -yDelta, mod1, mod2);
      }
   }
   return camChanged;
}
