// @(#)root/gui:$Name:$:$Id:$
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
// TGSplitter, TGVSplitter, TGHorizontal3DLine and TGVertical3DLine     //
//                                                                      //
// A splitter allows the frames left and right or above and below of    //
// it to be resized.                                                    //
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
   // Create vertical a splitter.

   if (!p->InheritsFrom(TGCompositeFrame::Class())) {
      Error("TGVSplitter", "parent must inherit from a TGCompositeFrame");
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
void TGVSplitter::SetFrame(TGFrame *frame)
{
   // Set frame to be resized.

   fFrame = frame;
}

//______________________________________________________________________________
Bool_t TGVSplitter::HandleButton(Event_t *event)
{
   // Handle mouse button event in vertical splitter.

   if (!fFrame) {
      Error("HandleButton", "frame to be resize not set");
      return kTRUE;
   }

   if (event->fType == kButtonPress) {
      fStartX   = event->fX;
      fDelta    = 0;
      fDragging = kTRUE;

      Int_t  x, y;
      gVirtualX->GetWindowSize(fFrame->GetId(), x, y, fWidth, fHeight);

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
      fDelta = event->fX - fStartX;
      Int_t w = (Int_t) fWidth;
      w += fDelta;
      if (w < 0) w = 0;
      fWidth = w;
      fFrame->Resize(fWidth, fHeight);

      TGCompositeFrame *p = (TGCompositeFrame *) GetParent();
      p->Layout();
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
   //gVirtualX->DrawLine(fId, fgShadowGC,  0, 0, fWidth-2, 0);
   //gVirtualX->DrawLine(fId, fgHilightGC, 0, 1, fWidth-1, 1);
   //gVirtualX->DrawLine(fId, fgHilightGC, fWidth-1, 0, fWidth-1, 1);
}
