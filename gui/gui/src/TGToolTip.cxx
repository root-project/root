// @(#)root/gui:$Id$
// Author: Fons Rademakers   22/02/98

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
// TGToolTip                                                            //
//                                                                      //
// A tooltip can be a one or multiple lines help text that is displayed //
// in a window when the mouse cursor overs a widget, without clicking   //
// it. A small box appears with suplementary information regarding the  //
// item being hovered over.                                             //                               //
//                                                                      //
// A multiline tooltip can be created by inserting a new-line character //
// "\n" in the tooltip string. For example:                             //
//                                                                      //
// fButton->SetToolTipText("Go to the ROOT page\n (http://root.cern.ch) //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGToolTip.h"
#include "TGLabel.h"
#include "TGResourcePool.h"
#include "TTimer.h"
#include "TSystem.h"
#include "TVirtualPad.h"
#include "TBox.h"


ClassImp(TGToolTip)

//______________________________________________________________________________
class TTipDelayTimer : public TTimer {
private:
   TGToolTip   *fTip;  // tooltip
public:
   TTipDelayTimer(TGToolTip *tip, Long_t ms) : TTimer(ms, kTRUE) { fTip = tip; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TTipDelayTimer::Notify()
{
   // Notify when timer times out and reset the timer.

   fTip->HandleTimer(0);
   Reset();
   return kFALSE;
}


//______________________________________________________________________________
TGToolTip::TGToolTip(const TGWindow *p, const TGFrame *f, const char *text,
                     Long_t delayms)
   : TGCompositeFrame(p, 10, 10, kTempFrame | kHorizontalFrame | kRaisedFrame)
{
   // Create a tool tip. P is the tool tips parent window (normally
   // fClient->GetRoot(), f is the frame to which the tool tip is associated,
   // text is the tool tip one-liner and delayms is the delay in ms before
   // the tool tip is shown.

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);
   SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());
   fLabel->SetTextColor(fClient->GetResourcePool()->GetTipFgndColor());

   AddFrame(fLabel, fL1 = new TGLayoutHints(kLHintsLeft | kLHintsTop,
                                            2, 3, 0, 0));
   MapSubwindows();
   Resize(GetDefaultSize());

   fWindow = f;
   fPad    = 0;
   fBox    = 0;
   fX = fY = -1;
   fDelay = new TTipDelayTimer(this, delayms);
}

//______________________________________________________________________________
TGToolTip::TGToolTip(const TGWindow *p, const TBox *box, const char *text,
                     Long_t delayms)
   : TGCompositeFrame(p, 10, 10, kTempFrame | kHorizontalFrame | kRaisedFrame)
{
   // Create a tool tip. P is the tool tips parent window (normally
   // fClient->GetRoot(), box is the area to which the tool tip is associated,
   // text is the tool tip one-liner and delayms is the delay in ms before
   // the tool tip is shown. When using this ctor with the box argument
   // you have to use Reset(const TVirtualPad *parent).

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);
   SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());
   fLabel->SetTextColor(fClient->GetResourcePool()->GetTipFgndColor());

   AddFrame(fLabel, fL1 = new TGLayoutHints(kLHintsLeft | kLHintsTop,
                                            2, 3, 0, 0));
   MapSubwindows();
   Resize(GetDefaultSize());

   fWindow = 0;
   fPad    = 0;
   fBox    = box;
   fDelay = new TTipDelayTimer(this, delayms);
}

//______________________________________________________________________________
TGToolTip::TGToolTip(const TBox *box, const char *text,Long_t delayms)
   : TGCompositeFrame(gClient->GetRoot(), 10, 10, kTempFrame | kHorizontalFrame | kRaisedFrame)
{
   // Create a tool tip in the parent window gClient->GetRoot(),
   // box is the area to which the tool tip is associated,
   // text is the tool tip one-liner and delayms is the delay in ms before
   // the tool tip is shown. When using this ctor with the box argument
   // you have to use Reset(const TVirtualPad *parent).

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);
   SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());
   fLabel->SetTextColor(fClient->GetResourcePool()->GetTipFgndColor());

   AddFrame(fLabel, fL1 = new TGLayoutHints(kLHintsLeft | kLHintsTop,
                                            2, 3, 0, 0));
   MapSubwindows();
   Resize(GetDefaultSize());

   fWindow = 0;
   fPad    = 0;
   fBox    = box;
   fDelay = new TTipDelayTimer(this, delayms);
}

//______________________________________________________________________________
TGToolTip::TGToolTip(Int_t x, Int_t y, const char *text, Long_t delayms)
   : TGCompositeFrame(gClient->GetDefaultRoot(), 10, 10, kTempFrame | kHorizontalFrame | kRaisedFrame)
{
   // Create a tool tip on global coordinates x, y.
   // text is the tool tip one-liner and delayms is the delay in ms before
   // the tool tip is shown.

   SetWindowAttributes_t attr;
   attr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   attr.fOverrideRedirect = kTRUE;
   attr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &attr);
   SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fClient->GetResourcePool()->GetTipBgndColor());
   fLabel->SetTextColor(fClient->GetResourcePool()->GetTipFgndColor());

   AddFrame(fLabel, fL1 = new TGLayoutHints(kLHintsLeft | kLHintsTop,
                                            2, 3, 0, 0));
   MapSubwindows();
   Resize(GetDefaultSize());

   fWindow = 0;
   fPad    = 0;
   fBox    = 0;
   fX      = x;
   fY      = y;
   fDelay = new TTipDelayTimer(this, delayms);
}

//______________________________________________________________________________
TGToolTip::~TGToolTip()
{
   // Delete a tool tip object.

   delete fDelay;
   delete fLabel;
   delete fL1;
}

//______________________________________________________________________________
void TGToolTip::DrawBorder()
{
   // Draw border of tool tip window.

   gVirtualX->DrawLine(fId, GetShadowGC()(), 0, 0, fWidth-2, 0);
   gVirtualX->DrawLine(fId, GetShadowGC()(), 0, 0, 0, fHeight-2);
   gVirtualX->DrawLine(fId, GetBlackGC()(),  0, fHeight-1, fWidth-1, fHeight-1);
   gVirtualX->DrawLine(fId, GetBlackGC()(),  fWidth-1, fHeight-1, fWidth-1, 0);
}

//______________________________________________________________________________
void TGToolTip::Show(Int_t x, Int_t y)
{
   // Show tool tip window.

   Move(x, y);
   MapWindow();
   RaiseWindow();

   Long_t args[2];
   args[0] = x;
   args[1] = y;

   Emit("Show(Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGToolTip::Hide()
{
   // Hide tool tip window. Use this method to hide the tool tip in a client
   // class.

   UnmapWindow();

   fDelay->Remove();

   Emit("Hide()");
}

//______________________________________________________________________________
void TGToolTip::Reset()
{
   // Reset tool tip popup delay timer. Use this method to activate tool tip
   // in a client class.

   fDelay->Reset();
   gSystem->AddTimer(fDelay);

   Emit("Reset()");
}

//______________________________________________________________________________
void TGToolTip::Reset(const TVirtualPad *parent)
{
   // Reset tool tip popup delay timer. Use this method to activate tool tip
   // in a client class. Use this method for graphics objects drawn in a
   // TCanvas, also the tool tip must have been created with the ctor
   // taking the TBox as argument.

   fPad = parent;

   fDelay->Reset();
   gSystem->AddTimer(fDelay);
}

//______________________________________________________________________________
Bool_t TGToolTip::HandleTimer(TTimer *)
{
   // If tool tip delay timer times out show tool tip window.

   Int_t    x = 0, y = 0, px1 = 0, px2 = 0, py1 = 0;
   Window_t wtarget;

   if (fWindow) {
      gVirtualX->TranslateCoordinates(fWindow->GetId(), GetParent()->GetId(),
                                      fX == -1 ? Int_t(fWindow->GetWidth() >> 1) : fX,
                                      fY == -1 ? Int_t(fWindow->GetHeight()) : fY,
                                      x, y, wtarget);
   } else if(fPad) {
      if (fBox) {
         px1 = fPad->XtoAbsPixel(fBox->GetX1());
         px2 = fPad->XtoAbsPixel(fBox->GetX2());
         py1 = fPad->YtoAbsPixel(fBox->GetY1());
         // py2 = fPad->YtoAbsPixel(fBox->GetY2());
      } else {
         px1 = fPad->XtoAbsPixel(fPad->GetX1());
         px2 = fPad->XtoAbsPixel(fPad->GetX2());
         py1 = fPad->YtoAbsPixel(fPad->GetY1());
         // py2 = fPad->YtoAbsPixel(fPad->GetY2());
      }
      gVirtualX->TranslateCoordinates(gVirtualX->GetWindowID(fPad->GetCanvasID()),
                                      GetParent()->GetId(),
                                      px1 + ((px2-px1) >> 1), py1,
                                      x, y, wtarget);
   } else {
      x = fX;
      y = fY;
   }

   Int_t move = 0;
   Window_t dum1, dum2;
   UInt_t mask = 0;
   Int_t mx, my;
   UInt_t screenW = fClient->GetDisplayWidth();
   UInt_t screenH = fClient->GetDisplayHeight();

   gVirtualX->QueryPointer(gVirtualX->GetDefaultRootWindow(),
                           dum1, dum2, mx, my, mx, my, mask);

   fLabel->SetWrapLength(-1);
   Resize(GetDefaultSize());

   // don't allow tooltip text lines longer than half the screen size
   if (fWidth > (screenW/2))
      fLabel->SetWrapLength((screenW/2)-15);
   Resize(GetDefaultSize());

   if (x + fWidth > screenW) {
      x = screenW - fWidth;
      move += 1;
   }

   if (y+4 + GetHeight() > screenH) {
      y = screenH - (fHeight + 25);
      move += 2;
   }

   // check if the mouse is inside the tooltip (may happen after
   // adjusting the position when out of screen) and place the tooltip
   // on the other side of the mouse pointer
   TGRectangle rect(x, y, x+fWidth, y+fHeight);
   if (rect.Contains(mx, my)) {
      if (move == 1) { // left
         if (fWidth+15 < (UInt_t)mx)
            x = mx - fWidth - 15;
         else if (my + fHeight+15 < screenH)
            y = my + 15;
         else if (fHeight+15 < (UInt_t)my)
            y = my - fHeight - 15;
      }
      else if (move == 2) { // up
         if (mx + fWidth+15 < screenW)
            x = mx + 15;
         else if (fHeight+15 < (UInt_t)my)
            y = my - fHeight - 15;
         else if (fWidth+15 < (UInt_t)mx)
            x = mx - fWidth - 15;
      }
      else { // up & left, right, down, ...
         if (my + fHeight+15 < screenH)
            y = my + 15;
         else if (mx + fWidth+15 < screenW)
            x = mx + 15;
         else if (fWidth+15 < (UInt_t)mx)
            x = mx - fWidth - 15;
         else if (fHeight+15 < (UInt_t)my)
            y = my - fHeight - 15;
      }
   }

   Show(x, y+4);

   fDelay->Remove();

   return kTRUE;
}

//______________________________________________________________________________
void TGToolTip::SetText(const char *new_text)
{
   // Set new tool tip text.

   fLabel->SetText(new TGString(new_text));
   Resize(GetDefaultSize());
}

//______________________________________________________________________________
void TGToolTip::SetDelay(Long_t delayms)
{
   // Set delay in milliseconds.

   fDelay->SetTime(delayms);
}

//______________________________________________________________________________
void TGToolTip::SetPosition(Int_t x, Int_t y)
{
   // Set popup position within specified frame (as specified in the ctor).
   // To get back default behaviour (in the middle just below the designated
   // frame) set position to -1,-1.

   fX = x;
   fY = y;

   if (fX < -1)
      fX = 0;
   if (fY < -1)
      fY = 0;

   if (fWindow) {
      if (fX > (Int_t) fWindow->GetWidth())
         fX = fWindow->GetWidth();
      if (fY > (Int_t) fWindow->GetHeight())
         fY = fWindow->GetHeight();
   }
}

//______________________________________________________________________________
const TGString *TGToolTip::GetText() const
{
   // Get the tool tip text.

   return fLabel->GetText();

}
