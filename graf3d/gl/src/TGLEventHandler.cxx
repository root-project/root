// @(#)root/gl:$Id$
// Author: Bertrand Bellenot   29/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLEventHandler                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TGLEventHandler.h"
#include "TGEventHandler.h"
#include "TGLViewer.h"
#include "TGLWidget.h"
#include "TGWindow.h"
#include "TPoint.h"
#include "TVirtualPad.h" // Remove when pad removed - use signal
#include "TVirtualX.h"
#include "TGClient.h"
#include "TVirtualGL.h"
#include "TGLOverlay.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TContextMenu.h"
#include "TGToolTip.h"
#include "KeySymbols.h"
#include "TGLAnnotation.h"

//______________________________________________________________________________
//
// Base-class and default implementation of event-handler for TGLViewer.
//
// This allows for complete disentanglement of GL-viewer from GUI
// event handling. Further, alternative event-handlers can easily be
// designed and set at run-time.
//
// The signals about object being selected or hovered above are
// emitted via the TGLViewer itself.
//
// This class is still under development.

ClassImp(TGLEventHandler);

//______________________________________________________________________________
TGLEventHandler::TGLEventHandler(TGWindow *w, TObject *obj) :
   TGEventHandler      ("TGLEventHandler", w, obj),
   fGLViewer           ((TGLViewer *)obj),
   fMouseTimer         (0),
   fLastPos            (-1, -1),
   fLastMouseOverPos   (-1, -1),
   fLastMouseOverShape (0),
   fTooltip            (0),
   fActiveButtonID     (0),
   fLastEventState     (0),
   fInPointerGrab      (kFALSE),
   fMouseTimerRunning  (kFALSE),
   fTooltipShown       (kFALSE),
   fTooltipPixelTolerance (3),
   fSecSelType(TGLViewer::kOnRequest)
{
   // Constructor.

   fMouseTimer = new TTimer(this, 80);
   fTooltip    = new TGToolTip(0, 0, "", 650);
   fTooltip->Hide();
}

//______________________________________________________________________________
TGLEventHandler::~TGLEventHandler()
{
   // Destructor.

   delete fMouseTimer;
   delete fTooltip;
}

//______________________________________________________________________________
void TGLEventHandler::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Process event of type 'event' - one of EEventType types,
   // occuring at window location px, py
   // This is provided for use when embedding GL viewer into pad

   /*enum EEventType {
   kNoEvent       =  0,
   kButton1Down   =  1, kButton2Down   =  2, kButton3Down   =  3, kKeyDown  =  4,
   kButton1Up     = 11, kButton2Up     = 12, kButton3Up     = 13, kKeyUp    = 14,
   kButton1Motion = 21, kButton2Motion = 22, kButton3Motion = 23, kKeyPress = 24,
   kButton1Locate = 41, kButton2Locate = 42, kButton3Locate = 43,
   kMouseMotion   = 51, kMouseEnter    = 52, kMouseLeave    = 53,
   kButton1Double = 61, kButton2Double = 62, kButton3Double = 63

   enum EGEventType {
   kGKeyPress, kKeyRelease, kButtonPress, kButtonRelease,
   kMotionNotify, kEnterNotify, kLeaveNotify, kFocusIn, kFocusOut,
   kExpose, kConfigureNotify, kMapNotify, kUnmapNotify, kDestroyNotify,
   kClientMessage, kSelectionClear, kSelectionRequest, kSelectionNotify,
   kColormapNotify, kButtonDoubleClick, kOtherEvent*/

   // Map our event EEventType (base/inc/Buttons.h) back to Event_t (base/inc/GuiTypes.h)
   // structure, and call appropriate HandleXyzz() function
   Event_t eventSt;
   eventSt.fX = px;
   eventSt.fY = py;
   eventSt.fState = 0;

   if (event != kKeyPress) {
      eventSt.fY -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
      eventSt.fX -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
   }

   switch (event) {
      case kMouseMotion:
         eventSt.fCode = kMouseMotion;
         eventSt.fType = kMotionNotify;
         HandleMotion(&eventSt);
         break;
      case kButton1Down:
      case kButton1Up:
      {
         eventSt.fCode = kButton1;
         eventSt.fType = event == kButton1Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      break;
      case kButton2Down:
      case kButton2Up:
      {
         eventSt.fCode = kButton2;
         eventSt.fType = event == kButton2Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      break;
      case kButton3Down:
      {
         eventSt.fState = kKeyShiftMask;
         eventSt.fCode = kButton1;
         eventSt.fType = kButtonPress;
         HandleButton(&eventSt);
      }
      break;
      case kButton3Up:
      {
         eventSt.fCode = kButton3;
         eventSt.fType = kButtonRelease;//event == kButton3Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      break;
      case kButton1Double:
      case kButton2Double:
      case kButton3Double:
      {
         eventSt.fCode = kButton1Double ? kButton1 : kButton2Double ? kButton2 : kButton3;
         eventSt.fType = kButtonDoubleClick;
         HandleDoubleClick(&eventSt);
      }
      break;
      case kButton1Motion:
      case kButton2Motion:
      case kButton3Motion:
      {

         eventSt.fCode = event == kButton1Motion ? kButton1 : event == kButton2Motion ? kButton2 : kButton3;
         eventSt.fType = kMotionNotify;
         HandleMotion(&eventSt);
      }
      break;
      case kKeyPress: // We only care about full key 'presses' not individual down/up
      {
         eventSt.fType = kGKeyPress;
         eventSt.fCode = py; // px contains key code - need modifiers from somewhere
         HandleKey(&eventSt);
      }
      break;
      case 6://trick :)
         if (fGLViewer->CurrentCamera().Zoom(+50, kFALSE, kFALSE)) { //TODO : val static const somewhere
            if (fGLViewer->fGLDevice != -1) {
               gGLManager->MarkForDirectCopy(fGLViewer->fGLDevice, kTRUE);
               gVirtualX->SetDrawMode(TVirtualX::kCopy);
            }
            fGLViewer->RequestDraw();
         }
         break;
      case 5://trick :)
         if (fGLViewer->CurrentCamera().Zoom(-50, kFALSE, kFALSE)) { //TODO : val static const somewhere
            if (fGLViewer->fGLDevice != -1) {
               gGLManager->MarkForDirectCopy(fGLViewer->fGLDevice, kTRUE);
               gVirtualX->SetDrawMode(TVirtualX::kCopy);
            }
            fGLViewer->RequestDraw();
         }
         break;
      case 7://trick :)
         eventSt.fState = kKeyShiftMask;
         eventSt.fCode = kButton1;
         eventSt.fType = kButtonPress;
         HandleButton(&eventSt);
         break;
      default:
      {
        // Error("TGLEventHandler::ExecuteEvent", "invalid event type");
      }
   }
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleEvent(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   if (event->fType == kFocusIn) {
      if (fGLViewer->fDragAction != TGLViewer::kDragNone) {
         Error("TGLEventHandler::HandleEvent", "active drag-action at focus-in.");
         fGLViewer->fDragAction = TGLViewer::kDragNone;
      }
      StartMouseTimer();
   }
   if (event->fType == kFocusOut) {
      if (fGLViewer->fDragAction != TGLViewer::kDragNone) {
         Warning("TGLEventHandler::HandleEvent", "drag-action active at focus-out.");
         fGLViewer->fDragAction = TGLViewer::kDragNone;
      }
      StopMouseTimer();
      ClearMouseOver();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleFocusChange(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   fGLViewer->MouseIdle(0, 0, 0);
   if (event->fType == kFocusIn) {
      if (fGLViewer->fDragAction != TGLViewer::kDragNone) {
         Error("TGLEventHandler::HandleFocusChange", "active drag-action at focus-in.");
         fGLViewer->fDragAction = TGLViewer::kDragNone;
      }
      StartMouseTimer();
      fGLViewer->Activated();
   }
   if (event->fType == kFocusOut) {
      if (fGLViewer->fDragAction != TGLViewer::kDragNone) {
         Warning("TGLEventHandler::HandleFocusChange", "drag-action active at focus-out.");
         fGLViewer->fDragAction = TGLViewer::kDragNone;
      }
      StopMouseTimer();
      ClearMouseOver();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleCrossing(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer.

   // Ignore grab and ungrab events.
   if (event->fCode != 0) {
      return kTRUE;
   }

   fGLViewer->MouseIdle(0, 0, 0);
   if (event->fType == kEnterNotify) {
      if (fGLViewer->fDragAction != TGLViewer::kDragNone) {
         Error("TGLEventHandler::HandleCrossing", "active drag-action at enter-notify.");
         fGLViewer->fDragAction = TGLViewer::kDragNone;
      }
      StartMouseTimer();
      // Maybe, maybe not...
      fGLViewer->Activated();
   }
   if (event->fType == kLeaveNotify) {
      if (fGLViewer->fDragAction != TGLViewer::kDragNone) {
         Warning("TGLEventHandler::HandleCrossing", "drag-action active at leave-notify.");
         fGLViewer->fDragAction = TGLViewer::kDragNone;
      }
      StopMouseTimer();
      ClearMouseOver();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleButton(Event_t * event)
{
   // Handle mouse button 'event'.
   static Event_t eventSt = {kOtherEvent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kFALSE, 0, 0, {0,0,0,0,0}};

   if (fGLViewer->IsLocked()) {
      if (gDebug>2) {
         Info("TGLEventHandler::HandleButton", "ignored - viewer is %s",
              fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }

   // Button DOWN
   if (event->fType == kButtonPress && event->fCode <= kButton3)
   {
      // Allow a single action/button down/up pairing - block others
      fGLViewer->MouseIdle(0, 0, 0);
      fGLViewer->Activated();
      if (fGLViewer->fDragAction != TGLViewer::kDragNone)
         return kFALSE;
      eventSt.fX = event->fX;
      eventSt.fY = event->fY;
      eventSt.fCode = event->fCode;

      if ( fGLViewer->GetPushAction() != TGLViewer::kPushStd )
      {
         fGLViewer->RequestSelect(event->fX, event->fY);
         if (fGLViewer->fSelRec.GetN() > 0)
         {
            TGLVector3 v(event->fX, event->fY, 0.5*fGLViewer->fSelRec.GetMinZ());
            fGLViewer->CurrentCamera().WindowToViewport(v);
            v = fGLViewer->CurrentCamera().ViewportToWorld(v);
            if (fGLViewer->GetPushAction() == TGLViewer::kPushCamCenter)
            {
               fGLViewer->CurrentCamera().SetExternalCenter(kTRUE);
               fGLViewer->CurrentCamera().SetCenterVec(v.X(), v.Y(), v.Z());
            }
            else
            {
               TGLSelectRecord& rec = fGLViewer->GetSelRec();
               TObject* obj = rec.GetObject();
               TGLRect& vp = fGLViewer->CurrentCamera().RefViewport();
               new TGLAnnotation(fGLViewer, obj->GetTitle(),  eventSt.fX*1.f/vp.Width(),  1 - eventSt.fY*1.f/vp.Height(), v);
            }

            fGLViewer->RequestDraw();
         }
         return kTRUE;
      }

      Bool_t grabPointer = kFALSE;
      Bool_t handled     = kFALSE;

      // Record active button for release
      fActiveButtonID = event->fCode;

      if (fGLViewer->fDragAction == TGLViewer::kDragNone && fGLViewer->fCurrentOvlElm)
      {
         if (fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event))
         {
            handled     = kTRUE;
            grabPointer = kTRUE;
            fGLViewer->fDragAction = TGLViewer::kDragOverlay;
            fGLViewer->RequestDraw();
         }
      }
      if ( ! handled)
      {
         switch(event->fCode)
         {
            // LEFT mouse button
            case kButton1:
            {
               if (event->fState & kKeyShiftMask) {
                  if (fGLViewer->RequestSelect(event->fX, event->fY)) {
                     fGLViewer->ApplySelection();
                     handled = kTRUE;
                  } else {
                     fGLViewer->SelectionChanged(); // Just notify clients.
                  }
               } else if ((fSecSelType == TGLViewer::kOnRequest || fSecSelType == TGLViewer::kOnKeyMod1) && (event->fState & kKeyMod1Mask)) {
                  fGLViewer->RequestSelect(event->fX, event->fY);
                  fGLViewer->RequestSecondarySelect(event->fX, event->fY);
                  if (fGLViewer->fSecSelRec.GetPhysShape() != 0)
                  {
                     TGLLogicalShape& lshape = const_cast<TGLLogicalShape&>
                        (*fGLViewer->fSecSelRec.GetPhysShape()->GetLogical());
                     lshape.ProcessSelection(*fGLViewer->fRnrCtx, fGLViewer->fSecSelRec);
                     handled = kTRUE;
                  }
               }
               // Switch to rotate -- thus we avoid Clicked() being sent on release.
               // Admittedly, this is a hack.
               // if ( ! handled)
               {
                  fGLViewer->fDragAction = TGLViewer::kDragCameraRotate;
                  grabPointer = kTRUE;
                  if (fMouseTimer) {
                     fMouseTimer->TurnOff();
                     fMouseTimer->Reset();
                  }
               }
               break;
            }
            // MID mouse button
            case kButton2:
            {
               fGLViewer->fDragAction = TGLViewer::kDragCameraTruck;
               grabPointer = kTRUE;
               break;
            }
            // RIGHT mouse button
            case kButton3:
            {
               // Shift + Right mouse - select+context menu
               if (event->fState & kKeyShiftMask)
               {
                  fGLViewer->RequestSelect(event->fX, event->fY);
                  const TGLPhysicalShape * selected = fGLViewer->fSelRec.GetPhysShape();
                  if (selected) {
                     if (!fGLViewer->fContextMenu) {
                        fGLViewer->fContextMenu = new TContextMenu("glcm", "GL Viewer Context Menu");
                     }
                     Int_t    x, y;
                     Window_t childdum;
                     gVirtualX->TranslateCoordinates(fGLViewer->fGLWidget->GetId(),
                                                     gClient->GetDefaultRoot()->GetId(),
                                                     event->fX, event->fY, x, y, childdum);
                     selected->InvokeContextMenu(*fGLViewer->fContextMenu, x, y);
                  }
               } else {
                  fGLViewer->fDragAction = TGLViewer::kDragCameraDolly;
                  grabPointer = kTRUE;
               }
               break;
            }
         }
      }

      if (grabPointer)
      {
         gVirtualX->GrabPointer(fGLViewer->GetGLWidget()->GetId(),
                                kButtonPressMask | kButtonReleaseMask | kPointerMotionMask,
                                kNone, kNone, kTRUE, kFALSE);
         fInPointerGrab = kTRUE;
      }
   }
   // Button UP
   else if (event->fType == kButtonRelease)
   {
      if (fInPointerGrab)
      {
         gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
         fInPointerGrab = kFALSE;
      }

      if (fGLViewer->GetPushAction() !=  TGLViewer::kPushStd)
      {
         // This should be 'tool' dependant.
         fGLViewer->fPushAction = TGLViewer::kPushStd;
         fGLViewer->RefreshPadEditor(fGLViewer);
         return kTRUE;
      }
      else if (fGLViewer->fDragAction == TGLViewer::kDragOverlay && fGLViewer->fCurrentOvlElm)
      {
         fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event);
         fGLViewer->OverlayDragFinished();
         if (fGLViewer->RequestOverlaySelect(event->fX, event->fY))
            fGLViewer->RequestDraw();
      }
      else if (fGLViewer->fDragAction >= TGLViewer::kDragCameraRotate &&
               fGLViewer->fDragAction <= TGLViewer::kDragCameraDolly)
      {
         fGLViewer->RequestDraw(TGLRnrCtx::kLODHigh);
      }

      // TODO: Check on Linux - on Win32 only see button release events
      // for mouse wheel
      switch(event->fCode) {
         // Buttons 4/5 are mouse wheel
         // Note: Modifiers (ctrl/shift) disabled as fState doesn't seem to
         // have correct modifier flags with mouse wheel under Windows.
         case kButton5: {
            // Zoom out (dolly or adjust camera FOV). TODO : val static const somewhere
            if (fGLViewer->CurrentCamera().Zoom(50, kFALSE, kFALSE))
               fGLViewer->fRedrawTimer->RequestDraw(10, TGLRnrCtx::kLODMed);
            return kTRUE;
            break;
         }
         case kButton4: {
            // Zoom in - adjust camera FOV. TODO : val static const somewhere
            if (fGLViewer->CurrentCamera().Zoom(-50, kFALSE, kFALSE))
               fGLViewer->fRedrawTimer->RequestDraw(10, TGLRnrCtx::kLODMed);
            return kTRUE;
            break;
         }
      }
      fGLViewer->fDragAction = TGLViewer::kDragNone;
      if (fGLViewer->fGLDevice != -1)
      {
         gGLManager->MarkForDirectCopy(fGLViewer->fGLDevice, kFALSE);
      }
      if ((event->fX == eventSt.fX) &&
          (event->fY == eventSt.fY) &&
          (eventSt.fCode == event->fCode))
      {
         TObject *obj = 0;
         fGLViewer->RequestSelect(fLastPos.fX, fLastPos.fY);
         TGLPhysicalShape *phys_shape = fGLViewer->fSelRec.GetPhysShape();

         if (phys_shape) obj = phys_shape->GetLogical()->GetExternal();
      
         // secondary selection
         if (phys_shape && fSecSelType == TGLViewer::kOnRequest
             && phys_shape->GetLogical()->AlwaysSecondarySelect())
         {
            fGLViewer->RequestSecondarySelect(fLastPos.fX, fLastPos.fY);
            fGLViewer->fSecSelRec.SetMultiple(event->fState & kKeyControlMask);

            if (fGLViewer->fSecSelRec.GetPhysShape() != 0)
            {
               TGLLogicalShape& lshape = const_cast<TGLLogicalShape&>
                  (*fGLViewer->fSecSelRec.GetPhysShape()->GetLogical());

               lshape.ProcessSelection(*fGLViewer->fRnrCtx, fGLViewer->fSecSelRec);
               switch (fGLViewer->fSecSelRec.GetSecSelResult())
               {
                  case TGLSelectRecord::kEnteringSelection:
                     fGLViewer->Clicked(obj, event->fCode, event->fState);
                     break;
                  case TGLSelectRecord::kLeavingSelection:
                     fGLViewer->UnClicked(obj, event->fCode, event->fState);
                     break;
                  case TGLSelectRecord::kModifyingInternalSelection:
                     fGLViewer->ReClicked(obj, event->fCode, event->fState);
                     break;
                  default:
                     break;
               };
            }
         }
         else
         {
            fGLViewer->Clicked(obj);
            fGLViewer->Clicked(obj, event->fCode, event->fState);
         }

         eventSt.fX = 0;
         eventSt.fY = 0;
         eventSt.fCode = 0;
         eventSt.fState = 0;
      }
      if (event->fCode == kButton1 && fMouseTimer)
      {
         fMouseTimer->TurnOn();
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleDoubleClick(Event_t *event)
{
   // Handle mouse double click 'event'.

   if (fGLViewer->IsLocked()) {
      if (gDebug>3) {
         Info("TGLEventHandler::HandleDoubleClick", "ignored - viewer is %s",
            fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }

   fGLViewer->MouseIdle(0, 0, 0);
   // Reset interactive camera mode on button double
   // click (unless mouse wheel)
   if (event->fCode != kButton4 && event->fCode != kButton5) {
      if (fGLViewer->fResetCameraOnDoubleClick) {
         fGLViewer->ResetCurrentCamera();
         fGLViewer->RequestDraw();
      }
      fGLViewer->DoubleClicked();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleConfigureNotify(Event_t *event)
{
   // Handle configure notify 'event' - a window resize/movement.

   if (fGLViewer->IsLocked())
   {
      if (gDebug > 0) {
         Info("TGLEventHandler::HandleConfigureNotify", "ignored - viewer is %s",
            fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }
   if (event)
   {
      fGLViewer->SetViewport(event->fX, event->fY, event->fWidth, event->fHeight);
      fGLViewer->fRedrawTimer->RequestDraw(10, TGLRnrCtx::kLODMed);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleExpose(Event_t * event)
{
   // Handle window expose 'event' - show.

   if (event->fCount != 0) return kTRUE;

   if (fGLViewer->IsLocked()) {
      if (gDebug > 0) {
         Info("TGLViewer::HandleExpose", "ignored - viewer is %s",
            fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }

   fGLViewer->fRedrawTimer->RequestDraw(20, TGLRnrCtx::kLODHigh);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleKey(Event_t *event)
{
   // Handle keyboard 'event'.

   fLastEventState = event->fState;

   fGLViewer->MouseIdle(0, 0, 0);
   if (fGLViewer->IsLocked()) {
      if (gDebug>3) {
         Info("TGLEventHandler::HandleKey", "ignored - viewer is %s",
            fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }

   char tmp[10] = {0};
   UInt_t keysym = 0;

   if (fGLViewer->fGLDevice == -1)
      gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);
   else
      keysym = event->fCode;
   fGLViewer->fRnrCtx->SetEventKeySym(keysym);

   Bool_t redraw = kFALSE;
   if (fGLViewer->fCurrentOvlElm &&
       fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event))
   {
      redraw = kTRUE;
   }
   else
   {
      Bool_t mod1 = event->fState & kKeyControlMask;
      Bool_t mod2 = event->fState & kKeyShiftMask;

      switch (keysym)
      {
         case kKey_R:
         case kKey_r:
            fGLViewer->SetStyle(TGLRnrCtx::kFill);
            redraw = kTRUE;
            break;
         case kKey_E:
         case kKey_e:
            fGLViewer->SwitchColorSet();
            redraw = kTRUE;
            break;
         case kKey_W:
         case kKey_w:
            fGLViewer->SetStyle(TGLRnrCtx::kWireFrame);
            redraw = kTRUE;
            break;
         case kKey_T:
         case kKey_t:
            fGLViewer->SetStyle(TGLRnrCtx::kOutline);
            redraw = kTRUE;
            break;

         case kKey_F1:
            fGLViewer->RequestSelect(fLastPos.fX, fLastPos.fY);
            fGLViewer->MouseIdle(fGLViewer->fSelRec.GetPhysShape(), (UInt_t)fLastPos.fX, (UInt_t)fLastPos.fY);
            break;

            // Camera
         case kKey_Plus:
         case kKey_J:
         case kKey_j:
            redraw = fGLViewer->CurrentCamera().Dolly(10, mod1, mod2);
            break;
         case kKey_Minus:
         case kKey_K:
         case kKey_k:
            redraw = fGLViewer->CurrentCamera().Dolly(-10, mod1, mod2);
            break;
         case kKey_Up:
            redraw = fGLViewer->CurrentCamera().Truck(0, 10, mod1, mod2);
            break;
         case kKey_Down:
            redraw = fGLViewer->CurrentCamera().Truck(0, -10, mod1, mod2);
            break;
         case kKey_Left:
            redraw = fGLViewer->CurrentCamera().Truck(-10, 0, mod1, mod2);
            break;
         case kKey_Right:
            redraw = fGLViewer->CurrentCamera().Truck(10, 0, mod1, mod2);
            break;
         case kKey_Home:
            fGLViewer->ResetCurrentCamera();
            redraw = kTRUE;
            break;

            // Toggle debugging mode
         case kKey_D:
         case kKey_d:
            fGLViewer->fDebugMode = !fGLViewer->fDebugMode;
            redraw = kTRUE;
            Info("OpenGL viewer debug mode : ", fGLViewer->fDebugMode ? "ON" : "OFF");
            break;
            // Forced rebuild for debugging mode
         case kKey_Space:
            if (fGLViewer->fDebugMode) {
               Info("OpenGL viewer FORCED rebuild", "");
               fGLViewer->UpdateScene();
            }
         default:;
      } // switch
   }

   if (redraw) {
      if (fGLViewer->fGLDevice != -1)
         gGLManager->MarkForDirectCopy(fGLViewer->fGLDevice, kTRUE);
      fGLViewer->RequestDraw();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleMotion(Event_t * event)
{
   // Handle mouse motion 'event'.

   fGLViewer->MouseIdle(0, 0, 0);
   if (fGLViewer->IsLocked()) {
      if (gDebug>3) {
         Info("TGLEventHandler::HandleMotion", "ignored - viewer is %s",
              fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return kFALSE;
   }

   Bool_t processed = kFALSE, changed = kFALSE;
   Short_t lod = TGLRnrCtx::kLODMed;

   // Camera interface requires GL coords - Y inverted
   Int_t  xDelta = event->fX - fLastPos.fX;
   Int_t  yDelta = event->fY - fLastPos.fY;
   Bool_t mod1   = event->fState & kKeyControlMask;
   Bool_t mod2   = event->fState & kKeyShiftMask;

   if (fMouseTimerRunning) StopMouseTimer();

   if (fTooltipShown &&
       ( TMath::Abs(event->fXRoot - fTooltipPos.fX) > fTooltipPixelTolerance ||
         TMath::Abs(event->fYRoot - fTooltipPos.fY) > fTooltipPixelTolerance ))
   {
      RemoveTooltip();
   }

   if (fGLViewer->fDragAction == TGLViewer::kDragNone)
   {
      if (fGLViewer->fRedrawTimer->IsPending()) {
         if (gDebug > 2)
            Info("TGLEventHandler::HandleMotion", "Redraw pending, ignoring.");
         return kTRUE;
      }
      changed = fGLViewer->RequestOverlaySelect(event->fX, event->fY);
      if (fGLViewer->fCurrentOvlElm)
         processed = fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event);
      lod = TGLRnrCtx::kLODHigh;
      if ( ! processed && ! fMouseTimerRunning)
         StartMouseTimer();
   }
   else if (fGLViewer->fDragAction == TGLViewer::kDragCameraRotate)
   {
      processed = Rotate(xDelta, yDelta, mod1, mod2);
   }
   else if (fGLViewer->fDragAction == TGLViewer::kDragCameraTruck)
   {
      processed = fGLViewer->CurrentCamera().Truck(xDelta, -yDelta, mod1, mod2);
   }
   else if (fGLViewer->fDragAction == TGLViewer::kDragCameraDolly)
   {
      processed = fGLViewer->CurrentCamera().Dolly(xDelta, mod1, mod2);
   }
   else if (fGLViewer->fDragAction == TGLViewer::kDragOverlay)
   {
      if (fGLViewer->fCurrentOvlElm)
         processed = fGLViewer->fCurrentOvlElm->Handle(*fGLViewer->fRnrCtx, fGLViewer->fOvlSelRec, event);
   }

   fLastPos.fX = event->fX;
   fLastPos.fY = event->fY;

   fLastGlobalPos.fX = event->fXRoot;
   fLastGlobalPos.fY = event->fYRoot;

   if (processed || changed) {
      if (fGLViewer->fGLDevice != -1) {
         gGLManager->MarkForDirectCopy(fGLViewer->fGLDevice, kTRUE);
         gVirtualX->SetDrawMode(TVirtualX::kCopy);
      }

      fGLViewer->RequestDraw(lod);
   }

   return processed;
}

//______________________________________________________________________________
Bool_t TGLEventHandler::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Method to handle action TGLViewer::kDragCameraRotate.

   return fGLViewer->CurrentCamera().Rotate(xDelta, -yDelta, mod1, mod2);
}

//______________________________________________________________________________
Bool_t TGLEventHandler::HandleTimer(TTimer *t)
{
   // If mouse delay timer times out emit signal.

   if (t != fMouseTimer) return kFALSE;

   fMouseTimerRunning = kFALSE;

   if (fGLViewer->fRedrawTimer->IsPending()) {
      if (gDebug > 2)
         Info("TGLEventHandler::HandleTimer", "Redraw pending, ignoring.");
      return kTRUE;
   }

   if (fGLViewer->fDragAction == TGLViewer::kDragNone)
   {
      if (fLastMouseOverPos != fLastPos)
      {
         fGLViewer->RequestSelect(fLastPos.fX, fLastPos.fY);
         if (fLastMouseOverShape != fGLViewer->fSelRec.GetPhysShape())
         {
            fLastMouseOverShape = fGLViewer->fSelRec.GetPhysShape();
            fGLViewer->MouseOver(fLastMouseOverShape);
            fGLViewer->MouseOver(fLastMouseOverShape, fLastEventState);
         }
         fLastMouseOverPos = fLastPos;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGLEventHandler::StartMouseTimer()
{
   // Start mouse timer in single-shot mode.

   fMouseTimer->Start(-1, kTRUE);
   fMouseTimerRunning = kTRUE;
}

//______________________________________________________________________________
void TGLEventHandler::StopMouseTimer()
{
   // Make sure mouse timers are not running.

   fMouseTimerRunning = kFALSE;
   fMouseTimer->Stop();
}

//______________________________________________________________________________
void TGLEventHandler::ClearMouseOver()
{
   // Clear mouse-over state and emit mouse-over signals.
   // Current overlay element is also told the mouse has left.

   fLastMouseOverPos.fX = fLastMouseOverPos.fY = -1;
   fLastMouseOverShape = 0;
   fGLViewer->MouseOver(fLastMouseOverShape);
   fGLViewer->MouseOver(fLastMouseOverShape, fLastEventState);

   fGLViewer->ClearCurrentOvlElm();
}

//______________________________________________________________________________
void TGLEventHandler::Repaint()
{
   // Handle window expose 'event' - show.

   if (fGLViewer->IsLocked()) {
      if (gDebug > 0) {
         Info("TGLViewer::HandleExpose", "ignored - viewer is %s",
            fGLViewer->LockName(fGLViewer->CurrentLock()));
      }
      return;
   }
   fGLViewer->fRedrawTimer->RequestDraw(20, TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
void TGLEventHandler::TriggerTooltip(const char* text)
{
   // Trigger display of tooltip.

   static UInt_t screenW = 0, screenH = 0;
   fTooltipPos   = fLastGlobalPos;
   fTooltipShown = kTRUE;
   fTooltip->SetText(text);
   Int_t x = fTooltipPos.fX + 16, y = fTooltipPos.fY + 16;
   if (screenW == 0 || screenH == 0) {
      screenW = gClient->GetDisplayWidth();
      screenH = gClient->GetDisplayHeight();
   }
   if (x + 5 + fTooltip->GetWidth() > screenW) {
      x = screenW - fTooltip->GetWidth() - 5;
      if (y + 5 + fTooltip->GetHeight() > screenH) {
         y -= (25 + fTooltip->GetHeight());
      }
   }
   if (y + 5 + fTooltip->GetHeight() > screenH) {
      y = screenH - fTooltip->GetHeight() - 10;
   }
   fTooltip->SetPosition(x, y);
   fTooltip->Reset();
}

//______________________________________________________________________________
void TGLEventHandler::RemoveTooltip()
{
   // Hide the tooltip.

   fTooltip->Hide();
   fTooltipShown = kFALSE;
}

//______________________________________________________________________________
void TGLEventHandler::SetMouseOverSelectDelay(Int_t ms)
{
   // Set delay of mouse-over probe (highlight).

   fMouseTimer->SetTime(ms);
}

//______________________________________________________________________________
void TGLEventHandler::SetMouseOverTooltipDelay(Int_t ms)
{
   // Set delay of tooltip timer.

   fTooltip->SetDelay(ms);
}
