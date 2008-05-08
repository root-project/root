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

#include "TEveCaloLegoGL.h"


//==============================================================================
//==============================================================================
// TEveLegoEventHandler
//==============================================================================

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
   // This is virtual method from base-class TGLEventHandler.

   if (event->fCode == kKey_Home)
   {
      fMode = kFree;
      fGLViewer->ResetCurrentCamera();
      return kTRUE;
   }

   return TGLEventHandler::HandleKey(event);
}

//______________________________________________________________________________
Bool_t TEveLegoEventHandler::HandleDoubleClick(Event_t *event)
{
   if (fGLViewer->IsLocked()) return kFALSE;

   if (event->fCode == kButton1)
   {
      fGLViewer->RequestSelect(event->fX, event->fY);
      TGLPhysicalShape* pshape = fGLViewer->GetSelRec().GetPhysShape();
      if (pshape && fGLViewer->GetSelRec().GetN() > 2)
      {
         TGLLogicalShape& lshape = const_cast<TGLLogicalShape&> (*pshape->GetLogical());
         TGLLogicalShape* f = &lshape;
         TEveCaloLegoGL*  lego   = dynamic_cast<TEveCaloLegoGL*>(f);
                 
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
Bool_t TEveLegoEventHandler::HandleMotion(Event_t * event)
{
   // This is virtual method from base-class TGLEventHandler.
   // Handles same actions as base-class, except TGLViewer::kDragCameraRotate.

   fGLViewer->MouseIdle(0, 0, 0);
   if (fGLViewer->IsLocked()) {
      if (gDebug>3) {
         Info("TEveLegoEventHandler::HandleMotion", "ignored - viewer is %s",
            fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }

   assert (event); // was if event==0 return

   Bool_t processed = kFALSE, changed = kFALSE;
   Short_t lod = TGLRnrCtx::kLODMed;

   // Camera interface requires GL coords - Y inverted
   Int_t  xDelta = event->fX - fLastPos.fX;
   Int_t  yDelta = event->fY - fLastPos.fY;
   Bool_t mod1   = event->fState & kKeyControlMask;
   Bool_t mod2   = event->fState & kKeyShiftMask;

   if (fGLViewer->GetDragAction() == TGLViewer::kDragNone)
   {
      changed = fGLViewer->RequestOverlaySelect(event->fX, event->fY);
      if (fGLViewer->GetCurrentOvlElm())
         processed = fGLViewer->GetCurrentOvlElm()->Handle(*fGLViewer->GetRnrCtx(), fGLViewer->GetOvlSelRec(), event);
      lod = TGLRnrCtx::kLODHigh;
   } else if (fGLViewer->GetDragAction() == TGLViewer::kDragCameraRotate) {
      processed = Rotate(xDelta, -yDelta, mod1, mod2);
   } else if (fGLViewer->GetDragAction() == TGLViewer::kDragCameraTruck) {
      processed = fGLViewer->CurrentCamera().Truck(xDelta, -yDelta, mod1, mod2);
   } else if (fGLViewer->GetDragAction() == TGLViewer::kDragCameraDolly) {
      processed = fGLViewer->CurrentCamera().Dolly(xDelta, mod1, mod2);
   } else if (fGLViewer->GetDragAction() == TGLViewer::kDragOverlay) {
      processed = fGLViewer->GetCurrentOvlElm()->Handle(*fGLViewer->GetRnrCtx(), fGLViewer->GetOvlSelRec(), event);
   }

   fLastPos.fX = event->fX;
   fLastPos.fY = event->fY;

   if (processed || changed) {
      if (fGLViewer->GetDev() != -1) {
         gGLManager->MarkForDirectCopy(fGLViewer->GetDev(), kTRUE);
         gVirtualX->SetDrawMode(TVirtualX::kCopy);
      }

      fGLViewer->RequestDraw(lod);
   }

   return processed;
}

//______________________________________________________________________________
Bool_t TEveLegoEventHandler::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Method to handle action TGLViewer::kDragCameraRotate. It switches from standard perspective
   // view to bird-view bellow angle fTransTheta and restores view when accumulated theta is larger
   // than transition angle. 

   using namespace TMath;

   TGLCamera &cam =  fGLViewer->GetRnrCtx()->RefCamera();
   Double_t hRotate = cam.AdjustDelta(yDelta, Pi()/cam.RefViewport().Height(), mod1, mod2);

   if (fMode == kLocked)
   {
      fTheta += hRotate;
      if (fTheta<0) fTheta=0;
      if (fTheta>fTransTheta)
      {
         fGLViewer->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
         fMode = kFree;
      }
   }
   else 
   {  
      Double_t theta = cam.GetTheta();
      Double_t thetaN = theta+hRotate;
      if(thetaN > Pi()-cam.GetVAxisMinAngle()) thetaN = Pi()-cam.GetVAxisMinAngle();
      else if (thetaN <cam.GetVAxisMinAngle()) thetaN = cam.GetVAxisMinAngle();

      fTheta = thetaN; 

      if (thetaN<fTransTheta)
      {
         fGLViewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
         fMode = kLocked;
      }
      else 
      {
         fGLViewer->CurrentCamera().Rotate(xDelta, yDelta, mod1, mod2);      
      }
   }
   return kTRUE;
}
