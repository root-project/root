// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   10/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGScrollBar and TGScrollBarElement                                   //
//                                                                      //
// The classes in this file implement scrollbars. Scrollbars can be     //
// either placed horizontal or vertical. A scrollbar contains three     //
// TGScrollBarElements: The "head", "tail" and "slider". The head and   //
// tail are fixed at either end and have the typical arrows in them.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGScrollBar.h"
#include "TGPicture.h"
#include "TSystem.h"
#include "TTimer.h"
#include "TMath.h"

ClassImp(TGScrollBarElement)
ClassImp(TGScrollBar)
ClassImp(TGHScrollBar)
ClassImp(TGVScrollBar)


//______________________________________________________________________________
class TSBRepeatTimer : public TTimer {
private:
   TGScrollBar   *fScrollBar;
public:
   TSBRepeatTimer(TGScrollBar *s, Long_t ms) : TTimer(ms, kTRUE) { fScrollBar = s; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TSBRepeatTimer::Notify()
{
   fScrollBar->HandleTimer(this);
   Reset();
   return kFALSE;
}


//______________________________________________________________________________
void TGScrollBarElement::SetState(Int_t state)
{
   // Change state of scrollbar element (either up or down).

   if (state != fState) {
      switch (state) {
         case kButtonDown:
            fOptions &= ~kRaisedFrame;
            fOptions |= kSunkenFrame;
            break;
         case kButtonUp:
            fOptions &= ~kSunkenFrame;
            fOptions |= kRaisedFrame;
            break;
      }
      fState = state;
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
void TGScrollBarElement::DrawBorder()
{
   // Draw border around scollbar element.

   switch (fOptions & (kSunkenFrame | kRaisedFrame)) {
      case kSunkenFrame: // pressed
         gVirtualX->DrawLine(fId, fgBlackGC, 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, fgBlackGC, 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, fgShadowGC,  1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, fgShadowGC,  1, 1, 1, fHeight-3);

         gVirtualX->DrawLine(fId, fgWhiteGC,  0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, fgWhiteGC,  fWidth-1, fHeight-1, fWidth-1, 1);
         gVirtualX->DrawLine(fId, fgBckgndGC,  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, fgBckgndGC,  fWidth-2, fHeight-2, fWidth-2, 2);

         if (fPic) {
            int x = (fWidth - fPic->GetWidth()) >> 1;
            int y = (fHeight - fPic->GetHeight()) >> 1;
            fPic->Draw(fId, fgBckgndGC, x+1, y+1); // 3, 3
         }
         break;

      case kRaisedFrame: // normal
         gVirtualX->DrawLine(fId, fgBckgndGC, 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, fgBckgndGC, 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, fgHilightGC, 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, fgHilightGC, 1, 1, 1, fHeight-3);

         gVirtualX->DrawLine(fId, fgShadowGC,  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, fgShadowGC,  fWidth-2, fHeight-2, fWidth-2, 1);
         gVirtualX->DrawLine(fId, fgBlackGC, 0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, fgBlackGC, fWidth-1, fHeight-1, fWidth-1, 0);

         if (fPic) {
            int x = (fWidth - fPic->GetWidth()) >> 1;
            int y = (fHeight - fPic->GetHeight()) >> 1;
            fPic->Draw(fId, fgBckgndGC, x, y); // 2, 2
         }
         break;

      default:
         break;
   }
}


//______________________________________________________________________________
TGScrollBar::~TGScrollBar()
{
   // Delete a scrollbar (either horizontal or vertical).

   delete fHead;
   delete fTail;
   delete fSlider;
   if (fHeadPic) fClient->FreePicture(fHeadPic);
   if (fTailPic) fClient->FreePicture(fTailPic);
   if (fRepeat) { delete fRepeat; fRepeat = 0; }
}

//______________________________________________________________________________
Bool_t TGScrollBar::HandleTimer(TTimer *t)
{
   // Handle repeat timer for horizontal or vertical scrollbar. Every time
   // timer times out we move slider.

   // shorten time out time to 50 milli seconds
   t->SetTime(50);

   Window_t  dum1, dum2;
   Event_t   ev;

   ev.fType    = kButtonPress;
   ev.fUser[0] = fSubw;

   gVirtualX->QueryPointer(fId, dum1, dum2, ev.fXRoot, ev.fYRoot, ev.fX, ev.fY,
                      ev.fState);

   HandleButton(&ev);

   return kTRUE;
}


//______________________________________________________________________________
TGHScrollBar::TGHScrollBar(const TGWindow *p, UInt_t w, UInt_t h,
                           UInt_t options, ULong_t back) :
    TGScrollBar(p, w, h, options, back)
{
   // Create a horizontal scrollbar widget.

   fHeadPic = fClient->GetPicture("arrow_left.xpm");
   fTailPic = fClient->GetPicture("arrow_right.xpm");

   if (!fHeadPic || !fTailPic)
      Error("TGHScrollBar", "arrow_*.xpm not found");

   fHead   = new TGScrollBarElement(this, fHeadPic, fgScrollBarWidth, fgScrollBarWidth,
                                    kRaisedFrame);
   fTail   = new TGScrollBarElement(this, fTailPic, fgScrollBarWidth, fgScrollBarWidth,
                                    kRaisedFrame);
   fSlider = new TGScrollBarElement(this, 0, fgScrollBarWidth, 50,
                                    kRaisedFrame);

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier, kButtonPressMask |
                    kButtonReleaseMask | kPointerMotionMask, kNone, kNone);

   fDragging = kFALSE;
   fX0 = fY0 = (fgScrollBarWidth = TMath::Max(fgScrollBarWidth, 5));
   fPos = 0;

   fRange = TMath::Max((Int_t) w - (fgScrollBarWidth << 1), 1);
   fPsize = fRange >> 1;

   fSliderSize  = 50;
   fSliderRange = 1;
}

//______________________________________________________________________________
void TGHScrollBar::Layout()
{
   // Layout and move horizontal scrollbar components.

   // Should also recalculate the slider size and range, etc.
   fHead->MoveResize(0, 0, fgScrollBarWidth, fgScrollBarWidth);
   fTail->MoveResize(fWidth-fgScrollBarWidth, 0, fgScrollBarWidth, fgScrollBarWidth);
   fSlider->MoveResize(fX0, 0, 50, fgScrollBarWidth);
}

//______________________________________________________________________________
Bool_t TGHScrollBar::HandleButton(Event_t *event)
{
   // Handle a mouse button event in a horizontal scrolbar.

   if (event->fType == kButtonPress) {

      // fUser[0] contains the subwindow (child) in which the event occured
      // (see GX11Gui.cxx)
      Window_t subw = (Window_t)event->fUser[0];

      if (subw == fSlider->GetId()) {
         fXp = event->fX - fX0;
         fYp = event->fY - fY0;
         fDragging = kTRUE;

      } else {

         if (!fRepeat)
            fRepeat = new TSBRepeatTimer(this, 400); // 500
         fRepeat->Reset();
         gSystem->AddTimer(fRepeat);
         fSubw = subw;

         if (subw == fHead->GetId()) {
            fHead->SetState(kButtonDown);
            fPos--;
         } else if (subw == fTail->GetId()) {
            fTail->SetState(kButtonDown);
            fPos++;
         } else if (event->fX > fgScrollBarWidth && event->fX < fX0)
            fPos -= fPsize;
         else if (event->fX > fX0+fSliderSize && event->fX < (Int_t)fWidth-fgScrollBarWidth)
            fPos += fPsize;

         fPos = TMath::Max(fPos, 0);
         fPos = TMath::Min(fPos, fRange-fPsize);

         fX0 = fgScrollBarWidth + fPos * fSliderRange / TMath::Max(fRange-fPsize, 1);

         fX0 = TMath::Max(fX0, fgScrollBarWidth);
         fX0 = TMath::Min(fX0, fgScrollBarWidth + fSliderRange);

         fSlider->Move(fX0, 0);

         SendMessage(fMsgWindow, MK_MSG(kC_HSCROLL, kSB_SLIDERTRACK), fPos, 0);

      }
   } else {
      fHead->SetState(kButtonUp);
      fTail->SetState(kButtonUp);

      if (fRepeat) {
         fRepeat->Remove();
         fRepeat->SetTime(400);  // might have been shortened in HandleTimer()
      }

      fDragging = kFALSE;

      fPos = TMath::Max(fPos, 0);
      fPos = TMath::Min(fPos, fRange-fPsize);

      SendMessage(fMsgWindow, MK_MSG(kC_HSCROLL, kSB_SLIDERPOS), fPos, 0);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHScrollBar::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in a horizontal scrollbar.

   if (fDragging) {
      fX0 = event->fX - fXp;
      fY0 = event->fY - fYp;

      fX0 = TMath::Max(fX0, fgScrollBarWidth);
      fX0 = TMath::Min(fX0, fgScrollBarWidth + fSliderRange);
      fSlider->Move(fX0, 0);
      fPos = (fX0 - fgScrollBarWidth) * (fRange-fPsize) / fSliderRange;

      fPos = TMath::Max(fPos, 0);
      fPos = TMath::Min(fPos, fRange-fPsize);

      SendMessage(fMsgWindow, MK_MSG(kC_HSCROLL, kSB_SLIDERTRACK), fPos, 0);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGHScrollBar::SetRange(Int_t range, Int_t page_size)
{
   // Set range of horizontal scrollbar.

   fRange = TMath::Max(range, 1);
   fPsize = TMath::Max(page_size, 0);

   fSliderSize = TMath::Max(fPsize * (fWidth - (fgScrollBarWidth << 1)) /
                            fRange, (UInt_t) 6);
   fSliderSize = TMath::Min((UInt_t)fSliderSize, fWidth - (fgScrollBarWidth << 1));

   fSliderRange = TMath::Max(fWidth - (fgScrollBarWidth << 1) - fSliderSize,
                             (UInt_t) 1);

   fX0 = fgScrollBarWidth + fPos * fSliderRange / TMath::Max(fRange-fPsize, 1);
   fX0 = TMath::Max(fX0, fgScrollBarWidth);
   fX0 = TMath::Min(fX0, fgScrollBarWidth + fSliderRange);

   fSlider->MoveResize(fX0, 0, fSliderSize, fgScrollBarWidth);

   //  fPos = (fX0 - fgScrollBarWidth) * (fRange-fPsize) / fSliderRange;

   fPos = TMath::Max(fPos, 0);
   fPos = TMath::Min(fPos, fRange-fPsize);

   SendMessage(fMsgWindow, MK_MSG(kC_HSCROLL, kSB_SLIDERPOS), fPos, 0);
}

//______________________________________________________________________________
void TGHScrollBar::SetPosition(Int_t pos)
{
   // Set logical slider position of horizontal scrollbar.

   fPos = TMath::Max(pos, 0);
   fPos = TMath::Min(pos, fRange-fPsize);

   fX0 = fgScrollBarWidth + fPos * fSliderRange / TMath::Max(fRange-fPsize, 1);
   fX0 = TMath::Max(fX0, fgScrollBarWidth);
   fX0 = TMath::Min(fX0, fgScrollBarWidth + fSliderRange);

   fSlider->MoveResize(fX0, 0, fSliderSize, fgScrollBarWidth);

   SendMessage(fMsgWindow, MK_MSG(kC_HSCROLL, kSB_SLIDERPOS), fPos, 0);
}


//______________________________________________________________________________
TGVScrollBar::TGVScrollBar(const TGWindow *p, UInt_t w, UInt_t h,
                           UInt_t options, ULong_t back) :
    TGScrollBar(p, w, h, options, back)
{
   // Create a vertical scrollbar.

   fHeadPic = fClient->GetPicture("arrow_up.xpm");
   fTailPic = fClient->GetPicture("arrow_down.xpm");

   if (!fHeadPic || !fTailPic)
      Error("TGHScrollBar", "arrow_*.xpm not found");

   fHead   = new TGScrollBarElement(this, fHeadPic, fgScrollBarWidth, fgScrollBarWidth,
                                    kRaisedFrame);
   fTail   = new TGScrollBarElement(this, fTailPic, fgScrollBarWidth, fgScrollBarWidth,
                                    kRaisedFrame);
   fSlider = new TGScrollBarElement(this, 0, fgScrollBarWidth, 50,
                                    kRaisedFrame);

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier, kButtonPressMask |
                    kButtonReleaseMask | kPointerMotionMask, kNone, kNone);

   fDragging = kFALSE;
   fX0 = fY0 = (fgScrollBarWidth = TMath::Max(fgScrollBarWidth, 5));
   fPos = 0;

   fRange = TMath::Max((Int_t) h - (fgScrollBarWidth << 1), 1);
   fPsize = fRange >> 1;

   fSliderSize  = 50;
   fSliderRange = 1;
}

//______________________________________________________________________________
void TGVScrollBar::Layout()
{
   // Layout and move vertical scrollbar components.

   // Should recalculate fSliderSize, fSliderRange, fX0, fY0, etc. too...

   fHead->MoveResize(0, 0, fgScrollBarWidth, fgScrollBarWidth);
   fTail->MoveResize(0, fHeight-fgScrollBarWidth, fgScrollBarWidth, fgScrollBarWidth);
   fSlider->MoveResize(0, fY0, fgScrollBarWidth, 50);
}

//______________________________________________________________________________
Bool_t TGVScrollBar::HandleButton(Event_t *event)
{
   // Handle mouse button event in vertical scrollbar.

   if (event->fType == kButtonPress) {

      // fUser[0] contains the subwindow (child) in which the event occured
      // (see GX11Gui.cxx)
      Window_t subw = (Window_t)event->fUser[0];

      if (subw == fSlider->GetId()) {
         fXp = event->fX - fX0;
         fYp = event->fY - fY0;
         fDragging = kTRUE;

      } else {

         if (!fRepeat)
            fRepeat = new TSBRepeatTimer(this, 400); // 500
         fRepeat->Reset();
         gSystem->AddTimer(fRepeat);
         fSubw = subw;

         if (subw == fHead->GetId()) {
            fHead->SetState(kButtonDown);
            fPos--;
         } else if (subw == fTail->GetId()) {
            fTail->SetState(kButtonDown);
            fPos++;
         } else if (event->fY > fgScrollBarWidth && event->fY < fY0)
            fPos -= fPsize;
         else if (event->fY > fY0+fSliderSize && event->fY < (Int_t)fHeight-fgScrollBarWidth)
            fPos += fPsize;

         fPos = TMath::Max(fPos, 0);
         fPos = TMath::Min(fPos, fRange-fPsize);

         fY0 = fgScrollBarWidth + fPos * fSliderRange / TMath::Max(fRange-fPsize, 1);

         fY0 = TMath::Max(fY0, fgScrollBarWidth);
         fY0 = TMath::Min(fY0, fgScrollBarWidth + fSliderRange);

         fSlider->Move(0, fY0);

         SendMessage(fMsgWindow, MK_MSG(kC_VSCROLL, kSB_SLIDERTRACK), fPos, 0);

      }
   } else {
      fHead->SetState(kButtonUp);
      fTail->SetState(kButtonUp);

      if (fRepeat) {
         fRepeat->Remove();
         fRepeat->SetTime(400);  // might have been shortened in HandleTimer()
      }

      fDragging = kFALSE;

      fPos = TMath::Max(fPos, 0);
      fPos = TMath::Min(fPos, fRange-fPsize);

      SendMessage(fMsgWindow, MK_MSG(kC_VSCROLL, kSB_SLIDERPOS), fPos, 0);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGVScrollBar::HandleMotion(Event_t *event)
{
   // Handle mouse motion in a vertical scrollbar.

   if (fDragging) {
      fX0 = event->fX - fXp;
      fY0 = event->fY - fYp;

      fY0 = TMath::Max(fY0, fgScrollBarWidth);
      fY0 = TMath::Min(fY0, fgScrollBarWidth + fSliderRange);
      fSlider->Move(0, fY0);
      fPos = (fY0 - fgScrollBarWidth) * (fRange-fPsize) / fSliderRange;

      fPos = TMath::Max(fPos, 0);
      fPos = TMath::Min(fPos, fRange-fPsize);

      SendMessage(fMsgWindow, MK_MSG(kC_VSCROLL, kSB_SLIDERTRACK), fPos, 0);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGVScrollBar::SetRange(Int_t range, Int_t page_size)
{
   // Set range of vertical scrollbar.

   fRange = TMath::Max(range, 1);
   fPsize = TMath::Max(page_size, 0);

   fSliderSize = TMath::Max(fPsize * (fHeight - (fgScrollBarWidth << 1)) /
                            fRange, (UInt_t) 6);
   fSliderSize = TMath::Min((UInt_t)fSliderSize, fHeight - (fgScrollBarWidth << 1));

   fSliderRange = TMath::Max(fHeight - (fgScrollBarWidth << 1) - fSliderSize,
                             (UInt_t)1);

   fY0 = fgScrollBarWidth + fPos * fSliderRange / TMath::Max(fRange-fPsize, 1);
   fY0 = TMath::Max(fY0, fgScrollBarWidth);
   fY0 = TMath::Min(fY0, fgScrollBarWidth + fSliderRange);

   fSlider->MoveResize(0, fY0, fgScrollBarWidth, fSliderSize);

   //  fPos = (fY0 - fgScrollBarWidth) * (fRange-fPsize) / fSliderRange;

   fPos = TMath::Max(fPos, 0);
   fPos = TMath::Min(fPos, fRange-fPsize);

   SendMessage(fMsgWindow, MK_MSG(kC_VSCROLL, kSB_SLIDERPOS), fPos, 0);
}

//______________________________________________________________________________
void TGVScrollBar::SetPosition(Int_t pos)
{
   // Set logical slider position of vertical scrollbar.

   fPos = TMath::Max(pos, 0);
   fPos = TMath::Min(pos, fRange-fPsize);

   fY0 = fgScrollBarWidth + fPos * fSliderRange / TMath::Max(fRange-fPsize, 1);
   fY0 = TMath::Max(fY0, fgScrollBarWidth);
   fY0 = TMath::Min(fY0, fgScrollBarWidth + fSliderRange);

   fSlider->MoveResize(0, fY0, fgScrollBarWidth, fSliderSize);

   SendMessage(fMsgWindow, MK_MSG(kC_VSCROLL, kSB_SLIDERPOS), fPos, 0);
}
