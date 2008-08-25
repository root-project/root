/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TEveLegoEventHandler.h"

#include "TGLViewer.h"
#include "TGLWidget.h"
#include "TGLOverlay.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "KeySymbols.h"

#include "TMath.h"
#include "TGLUtil.h"

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
TEveLegoEventHandler::TEveLegoEventHandler(const char *name, TGWindow *w, TObject *obj,
                                 const char *title) :
   TGLEventHandler(name, w, obj, title),

   fMode(kFree),
   fTransTheta(0.5f),
   fTheta(0.f),

   fLastPickedLego(0)
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
Bool_t TEveLegoEventHandler::HandleDoubleClick(Event_t *event)
{
   // Virtual from TGLEventHandler.
   // Sets id of the tower with scale.

   if (fGLViewer->IsLocked()) return kFALSE;

   if (event->fCode == kButton1)
   {
      fGLViewer->RequestSelect(event->fX, event->fY);
      TGLPhysicalShape* pshape = fGLViewer->GetSelRec().GetPhysShape();
      if (pshape && fGLViewer->GetSelRec().GetN() > 2)
      {
         TGLLogicalShape& lshape = const_cast<TGLLogicalShape&> (*pshape->GetLogical());
         TGLLogicalShape* f = &lshape;
         TEveCaloLego*  lego   = dynamic_cast<TEveCaloLego*>(f->GetExternal());

         if (lego)
         {
            fLastPickedLego = lego;
            lego->SetTowerPicked(fGLViewer->GetSelRec().GetItem(2));
         }
      }
      else if (fLastPickedLego)
      {
         fLastPickedLego->SetTowerPicked(-1);
      }
      fGLViewer->RequestDraw();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TEveLegoEventHandler::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Method to handle action TGLViewer::kDragCameraRotate. It switches from standard perspective
   // view to bird-view bellow angle fTransTheta and restores view when accumulated theta is larger
   // than transition angle.

   using namespace TMath;

   TGLCamera &cam =  fGLViewer->GetRnrCtx()->RefCamera();
   Double_t hRotate = cam.AdjustDelta(-yDelta, Pi()/cam.RefViewport().Height(), mod1, mod2);

   if (fMode == kLocked)
   {
      fTheta += hRotate;
      if (fTheta < 0) fTheta = 0;
      if (fTheta > fTransTheta)
      {
         fGLViewer->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
         fMode = kFree;
      }
   }
   else
   {
      Double_t theta  = cam.GetTheta();
      Double_t thetaN = theta + hRotate;
      if (thetaN > Pi() - cam.GetVAxisMinAngle()) thetaN = Pi() - cam.GetVAxisMinAngle();
      else if (thetaN < cam.GetVAxisMinAngle())   thetaN = cam.GetVAxisMinAngle();

      fTheta = thetaN;

      if (thetaN < fTransTheta)
      {
         fGLViewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
         fMode = kLocked;
      }
      else
      {
         fGLViewer->CurrentCamera().Rotate(xDelta, -yDelta, mod1, mod2);
      }
   }
   return kTRUE;
}
