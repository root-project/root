// @(#)root/gui:$Name:  $:$Id: TGSplitter.cxx,v 1.1 2000/09/07 00:32:14 rdm Exp $
// Author: Fons Rademakers   6/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSplitter, TGVSplitter, TGHSplitter, TGHorizontal3DLine and         //
// TGVertical3DLine                                                     //
//                                                                      //
// A splitter allows the frames left and right or above and below of    //
// it to be resized. The frame to be resized must have the kFixedWidth  //
// or kFixedHeight property set.                                        //
// A horizontal 3D line is a line that typically separates a toolbar    //
// from the menubar.                                                    //
// A vertical 3D line is a line that can be used to separate groups of  //
// widgets.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGSplitter.h"


ClassImp(TGHorizontal3DLine)
ClassImp(TGVertical3DLine)
ClassImp(TGSplitter)
ClassImp(TGVSplitter)
ClassImp(TGHSplitter)


//______________________________________________________________________________
TGSplitter::TGSplitter(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options, ULong_t back) : TGFrame(p, w, h, options, back)
{
   // Create a splitter.

   fDragging = kFALSE;
}


//______________________________________________________________________________
TGVSplitter::TGVSplitter(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options, ULong_t back) : TGSplitter(p, w, h, options, back)
{
   // Create a vertical splitter.

   fSplitCursor = kNone;

   if (!p->InheritsFrom(TGCompositeFrame::Class())) {
      Error("TGVSplitter", "parent must inherit from a TGCompositeFrame");
      return;
   }
   if (!(((TGCompositeFrame*)p)->GetOptions() & kHorizontalFrame)) {
      Error("TGVSplitter", "parent must have a horizontal layout manager");
      return;
   }

   fSplitCursor = gVirtualX->CreateCursor(kArrowHor);
   fFrame = 0;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   gVirtualX->SelectInput(fId, kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
void TGVSplitter::SetFrame(TGFrame *frame, Bool_t left)
{
   // Set frame to be resized. If frame is on the left of the splitter
   // set left to true.

   fFrame = frame;
   fLeft  = left;

   if (!(fFrame->GetOptions() & kFixedWidth))
      Error("SetFrame", "resize frame must have kFixedWidth option set");
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleButton(Event_t *event)
{
   // Handle mouse button event in vertical splitter.

   if (fSplitCursor == kNone) return kTRUE;

   if (!fFrame) {
      Error("HandleButton", "frame to be resized not set");
      return kTRUE;
   }

   if (event->fType == kButtonPress) {
      fStartX   = event->fXRoot;
      fDragging = kTRUE;

      Int_t  x, y;
      gVirtualX->GetWindowSize(fFrame->GetId(), x, y, fWidth, fHeight);

      // get fMin and fMax in root coordinates
      Int_t    xroot, yroot;
      UInt_t   w, h;
      Window_t wdum;
      gVirtualX->GetWindowSize(fParent->GetId(), x, y, w, h);
      gVirtualX->TranslateCoordinates(fParent->GetParent()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      x, y, xroot, yroot, wdum);
      fMin = xroot;
      fMax = xroot + w - 2;

      // last argument kFALSE forces all specified events to this window
      gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                             kPointerMotionMask, kNone, fSplitCursor,
                             kTRUE, kFALSE);
   } else {
      fDragging = kFALSE;
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in vertical splitter.

   if (fDragging) {
      Int_t xr = event->fXRoot;
      if (xr > fMax) xr = fMax;
      if (xr < fMin) xr = fMin;
      Int_t delta = xr - fStartX;
      Int_t w = (Int_t) fWidth;
      if (fLeft)
         w += delta;
      else
         w -= delta;
      if (w < 0) w = 0;
      fStartX = xr;

      if (delta != 0) {
         fWidth = w;
         fFrame->Resize(fWidth, fHeight);

         TGCompositeFrame *p = (TGCompositeFrame *) GetParent();
         p->Layout();
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleCrossing(Event_t *event)
{
   // Handle mouse motion event in vertical splitter.

   if (event->fType == kEnterNotify)
      gVirtualX->SetCursor(fId, fSplitCursor);
   else
      gVirtualX->SetCursor(fId, kNone);

   return kTRUE;
}

//______________________________________________________________________________
void TGVSplitter::DrawBorder()
{
   // Draw vertical splitter.

   // Currently no special graphical representation except for cursor change
   // when crossing a splitter
}


//______________________________________________________________________________
TGHSplitter::TGHSplitter(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options, ULong_t back) : TGSplitter(p, w, h, options, back)
{
   // Create a horizontal splitter.

   fSplitCursor = kNone;

   if (!p->InheritsFrom(TGCompositeFrame::Class())) {
      Error("TGHSplitter", "parent must inherit from a TGCompositeFrame");
      return;
   }
   if (!(((TGCompositeFrame*)p)->GetOptions() & kVerticalFrame)) {
      Error("TGVSplitter", "parent must have a vertical layout manager");
      return;
   }

   fSplitCursor = gVirtualX->CreateCursor(kArrowVer);
   fFrame = 0;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   gVirtualX->SelectInput(fId, kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
void TGHSplitter::SetFrame(TGFrame *frame, Bool_t above)
{
   // Set frame to be resized. If frame is above the splitter
   // set above to true.

   fFrame = frame;
   fAbove = above;

   if (!(fFrame->GetOptions() & kFixedHeight))
      Error("SetFrame", "resize frame must have kFixedHeight option set");
}

//______________________________________________________________________________
Bool_t TGHSplitter::HandleButton(Event_t *event)
{
   // Handle mouse button event in horizontal splitter.

   if (fSplitCursor == kNone) return kTRUE;

   if (!fFrame) {
      Error("HandleButton", "frame to be resized not set");
      return kTRUE;
   }

   if (event->fType == kButtonPress) {
      fStartY   = event->fYRoot;
      fDragging = kTRUE;

      Int_t  x, y;
      gVirtualX->GetWindowSize(fFrame->GetId(), x, y, fWidth, fHeight);

      // get fMin and fMax in root coordinates
      Int_t    xroot, yroot;
      UInt_t   w, h;
      Window_t wdum;
      gVirtualX->GetWindowSize(fParent->GetId(), x, y, w, h);
      gVirtualX->TranslateCoordinates(fParent->GetParent()->GetId(),
                                      fClient->GetRoot()->GetId(),
                                      x, y, xroot, yroot, wdum);
      fMin = yroot;
      fMax = yroot + h - 2;

      // last argument kFALSE forces all specified events to this window
      gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                             kPointerMotionMask, kNone, fSplitCursor,
                             kTRUE, kFALSE);
   } else {
      fDragging = kFALSE;
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHSplitter::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in horizontal splitter.

   if (fDragging) {
      Int_t yr = event->fYRoot;
      if (yr > fMax) yr = fMax;
      if (yr < fMin) yr = fMin;
      Int_t delta = yr - fStartY;
      Int_t h = (Int_t) fHeight;
      if (fAbove)
         h += delta;
      else
         h -= delta;
      if (h < 0) h = 0;
      fStartY = yr;

      if (delta != 0) {
         fHeight = h;
         fFrame->Resize(fWidth, fHeight);

         TGCompositeFrame *p = (TGCompositeFrame *) GetParent();
         p->Layout();
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHSplitter::HandleCrossing(Event_t *event)
{
   // Handle mouse motion event in horizontal splitter.

   if (event->fType == kEnterNotify)
      gVirtualX->SetCursor(fId, fSplitCursor);
   else
      gVirtualX->SetCursor(fId, kNone);

   return kTRUE;
}

//______________________________________________________________________________
void TGHSplitter::DrawBorder()
{
   // Draw horizontal splitter.

   // Currently no special graphical representation except for cursor change
   // when crossing a splitter
}
