// @(#)root/gui:$Name:  $:$Id: TGToolTip.cxx,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
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
// A tooltip is a one line help text that is displayed in a window      //
// when the cursor rests over a widget. For an example of usage see     //
// the TGButton class.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGToolTip.h"
#include "TGLabel.h"
#include "TTimer.h"
#include "TSystem.h"
#include "TVirtualPad.h"
#include "TBox.h"


ClassImp(TGToolTip)

//______________________________________________________________________________
class TTipDelayTimer : public TTimer {
private:
   TGToolTip   *fTip;
public:
   TTipDelayTimer(TGToolTip *tip, Long_t ms) : TTimer(ms, kTRUE) { fTip = tip; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TTipDelayTimer::Notify()
{
   fTip->HandleTimer(0);
   Reset();
   return kFALSE;
}


//______________________________________________________________________________
TGToolTip::TGToolTip(const TGWindow *p, const TGFrame *f, const char *text,
   Long_t delayms) : TGCompositeFrame(p, 10, 10, kHorizontalFrame | kRaisedFrame)
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
   SetBackgroundColor(fgLightYellowPixel);

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fgLightYellowPixel);

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
   Long_t delayms) : TGCompositeFrame(p, 10, 10, kHorizontalFrame | kRaisedFrame)
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
   SetBackgroundColor(fgLightYellowPixel);

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fgLightYellowPixel);

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
          : TGCompositeFrame(gClient->GetRoot(), 10, 10, kHorizontalFrame | kRaisedFrame)
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
   SetBackgroundColor(fgLightYellowPixel);

   fLabel = new TGLabel(this, text);
   fLabel->SetBackgroundColor(fgLightYellowPixel);

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

   gVirtualX->DrawLine(fId, fgShadowGC, 0, 0, fWidth-2, 0);
   gVirtualX->DrawLine(fId, fgShadowGC, 0, 0, 0, fHeight-2);
   gVirtualX->DrawLine(fId, fgBlackGC,  0, fHeight-1, fWidth-1, fHeight-1);
   gVirtualX->DrawLine(fId, fgBlackGC,  fWidth-1, fHeight-1, fWidth-1, 0);
}

//______________________________________________________________________________
void TGToolTip::Show(Int_t x, Int_t y)
{
   // Show tool tip window.

   Move(x, y);
   MapWindow();
   RaiseWindow();
}

//______________________________________________________________________________
void TGToolTip::Hide()
{
   // Hide tool tip window. Use this method to hide the tool tip in a client
   // class.

   UnmapWindow();

   fDelay->Remove();
}

//______________________________________________________________________________
void TGToolTip::Reset()
{
   // Reset tool tip popup delay timer. Use this method to activate tool tip
   // in a client class.

   fDelay->Reset();
   gSystem->AddTimer(fDelay);
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

   Int_t    x = 0, y = 0, px1 = 0, px2 = 0, py1 = 0, py2 = 0;
   Window_t wtarget;

   if (fWindow) {
      gVirtualX->TranslateCoordinates(fWindow->GetId(), GetParent()->GetId(),
                                      fX == -1 ? fWindow->GetWidth() >> 1 : fX,
                                      fY == -1 ? fWindow->GetHeight() : fY,
                                      x, y, wtarget);
   } else {

      if (!fPad) {
         Error("HandleTimer", "parent pad not set for tool tip");
         return kTRUE;
      }

      if (fBox) {
         px1 = fPad->XtoAbsPixel(fBox->GetX1());
         px2 = fPad->XtoAbsPixel(fBox->GetX2());
         py1 = fPad->YtoAbsPixel(fBox->GetY1());
         py2 = fPad->YtoAbsPixel(fBox->GetY2());
      } else {
         px1 = fPad->XtoAbsPixel(fPad->GetX1());
         px2 = fPad->XtoAbsPixel(fPad->GetX2());
         py1 = fPad->YtoAbsPixel(fPad->GetY1());
         py2 = fPad->YtoAbsPixel(fPad->GetY2());
      }
      gVirtualX->TranslateCoordinates(gVirtualX->GetWindowID(fPad->GetCanvasID()),
                                 GetParent()->GetId(),
                                 px1 + ((px2-px1) >> 1), py1,
                                 x, y, wtarget);
   }

   Int_t   screenX, screenY;
   UInt_t  screenW, screenH;     // width and height of screen

   gVirtualX->GetWindowSize(gClient->GetRoot()->GetId(), screenX, screenY,
                            screenW, screenH);

   if (x + fWidth > screenW)
      x = screenW - fWidth;

   if (y+4 + GetHeight() > screenH)
      if (fWindow)
         y -= GetHeight() + fWindow->GetHeight() + 2*4;
      else
         y -= GetHeight() + py1-py2 + 2*4;

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
