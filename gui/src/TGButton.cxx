// @(#)root/gui:$Name:  $:$Id: TGButton.cxx,v 1.2 2000/09/29 08:57:05 rdm Exp $
// Author: Fons Rademakers   06/01/98

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
// TGButton, TGTextButton, TGPictureButton, TGCheckButton and           //
// TGRadioButton                                                        //
//                                                                      //
// This header defines all GUI button widgets.                          //
//                                                                      //
// TGButton is a button abstract base class. It defines general button  //
// behaviour.                                                           //
//                                                                      //
// Selecting a text or picture button will generate the event:          //
// kC_COMMAND, kCM_BUTTON, button id, user data.                        //
//                                                                      //
// Selecting a check button will generate the event:                    //
// kC_COMMAND, kCM_CHECKBUTTON, button id, user data.                   //
//                                                                      //
// Selecting a radio button will generate the event:                    //
// kC_COMMAND, kCM_RADIOBUTTON, button id, user data.                   //
//                                                                      //
// If a command string has been specified (via SetCommand()) then this  //
// command string will be executed via the interpreter whenever a       //
// button is selected. A command string can contain the macros:         //
// $MSG   -- kC_COMMAND, kCM[CHECK|RADIO]BUTTON packed message          //
//           (use GET_MSG() and GET_SUBMSG() to unpack)                 //
// $PARM1 -- button id                                                  //
// $PARM2 -- user data pointer                                          //
// Before executing these macros are expanded into the respective       //
// Long_t's                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGButton.h"
#include "TGWidget.h"
#include "TGPicture.h"
#include "TGToolTip.h"


ClassImp(TGButton)
ClassImp(TGTextButton)
ClassImp(TGPictureButton)
ClassImp(TGCheckButton)
ClassImp(TGRadioButton)


//______________________________________________________________________________
TGButton::TGButton(const TGWindow *p, Int_t id, GContext_t norm, UInt_t options)
    : TGFrame(p, 1, 1, options)
{
   // Create button base class part.

   fWidgetId    = id;
   fWidgetFlags = kWidgetWantFocus;
   fMsgWindow   = p;
   fUserData    = 0;
   fTip         = 0;

   fNormGC   = norm;
   fState    = kButtonUp;
   fStayDown = kFALSE;

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask,
                    kNone, kNone);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGButton::~TGButton()
{
   // Delete button.

   delete fTip;
}

//______________________________________________________________________________
void TGButton::SetState(EButtonState state)
{
   // Set button state.

   if (state != fState) {
      switch (state) {
         case kButtonEngaged:
         case kButtonDown:
            fOptions &= ~kRaisedFrame;
            fOptions |= kSunkenFrame;
            break;
         case kButtonDisabled:
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
Bool_t TGButton::HandleButton(Event_t *event)
{
   // Handle mouse button event.

   Bool_t click = kFALSE;

   if (fTip) fTip->Hide();

   if (fState == kButtonDisabled) return kTRUE;

   // We don't need to check the button number as GrabButton will
   // only allow button1 events
   if (event->fType == kButtonPress) {
      if (fState == kButtonEngaged) return kTRUE;
      SetState(kButtonDown);
   } else { // ButtonRelease
      if (fState == kButtonEngaged /*&& !allowRelease*/) {
         click = kTRUE;
      } else {
         click = (fState == kButtonDown);
         if (click && fStayDown)
           SetState(kButtonEngaged);
         else
           SetState(kButtonUp);
      }
   }
   if (click) {
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_BUTTON), fWidgetId,
                  (Long_t) fUserData);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_BUTTON), fWidgetId,
                           (Long_t) fUserData);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGButton::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kEnterNotify)
         fTip->Reset();
      else
         fTip->Hide();
   }

   if (fgDbw != event->fWindow) return kTRUE;

   if (!(event->fState & (kButton1Mask | kButton2Mask | kButton3Mask)))
      return kTRUE;

   if (fState == kButtonEngaged || fState == kButtonDisabled) return kTRUE;

   if (event->fType == kEnterNotify)
      SetState(kButtonDown);
   else
      SetState(kButtonUp);

   return kTRUE;
}

//______________________________________________________________________________
void TGButton::SetToolTipText(const char *text, Long_t delayms)
{
   // Set tool tip text associated with this button. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // text = 0.

   if (fTip) {
      delete fTip;
      fTip = 0;
   }

   if (text && strlen(text))
      fTip = new TGToolTip(fClient->GetRoot(), this, text, delayms);
}

//______________________________________________________________________________
const TGGC &TGButton::GetDefaultGC()
{ return fgDefaultGC; }

//______________________________________________________________________________
const TGGC &TGButton::GetHibckgndGC()
{ return fgHibckgndGC; }


//______________________________________________________________________________
TGTextButton::TGTextButton(const TGWindow *p, TGHotString *s, Int_t id,
                           GContext_t norm, FontStruct_t font,
                           UInt_t options) : TGButton(p, id, norm, options)
{
   // Create a text button widget. The hotstring will be adopted and deleted
   // by the text button.

   fLabel = s;
   fFontStruct = font;

   Init();
}

//______________________________________________________________________________
TGTextButton::TGTextButton(const TGWindow *p, const char *s, Int_t id,
                           GContext_t norm, FontStruct_t font,
                           UInt_t options) : TGButton(p, id, norm, options)
{
   // Create a text button widget.

   fLabel = new TGHotString(s);
   fFontStruct = font;

   Init();
}

//______________________________________________________________________________
TGTextButton::TGTextButton(const TGWindow *p, const char *s, const char *cmd,
                           Int_t id, GContext_t norm, FontStruct_t font,
                           UInt_t options) : TGButton(p, id, norm, options)
{
   // Create a text button widget and set cmd string at same time.

   fLabel = new TGHotString(s);
   fFontStruct = font;
   fCommand = cmd;

   Init();
}

//______________________________________________________________________________
void TGTextButton::Init()
{
   // Common initialization used by the different ctors.

   int hotchar, max_ascent, max_descent;

   fTMode = kTextCenterX | kTextCenterY;
   fHKeycode = 0;

   fTWidth  = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   Resize(fTWidth + 8, fTHeight + 7);

   if ((hotchar = fLabel->GetHotChar()) != 0) {
      if ((fHKeycode = gVirtualX->KeysymToKeycode(hotchar)) != 0) {
         const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
         main->BindKey(this, fHKeycode, kKeyMod1Mask);
      }
   }
}

//______________________________________________________________________________
TGTextButton::~TGTextButton()
{
   // Delete a text button widget.

   if (fHKeycode) {
      const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
      main->RemoveBind(this, fHKeycode, kKeyMod1Mask);
   }
   if (fLabel) delete fLabel;
}

//______________________________________________________________________________
void TGTextButton::SetText(TGHotString *new_label)
{
   // Set new button text.

   if (fLabel) delete fLabel;
   fLabel = new_label;

   int max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGTextButton::DoRedraw()
{
   // Draw the text button.

   int x, y;
   int max_ascent, max_descent;

   TGFrame::DoRedraw();

   if (fTMode & kTextLeft)
      x = 4;
   else if (fTMode & kTextRight)
      x = fWidth - fTWidth - 4;
   else
      x = (fWidth - fTWidth) >> 1;

   if (fTMode & kTextTop)
      y = 3;
   else if (fTMode & kTextBottom)
      y = fHeight - fTHeight - 3;
   else
      y = (fHeight - fTHeight) >> 1;

   if (fState == kButtonDown || fState == kButtonEngaged) { ++x; ++y; }
   if (fState == kButtonEngaged) {
      gVirtualX->FillRectangle(fId, fgHibckgndGC(), 2, 2, fWidth-4, fHeight-4);
      gVirtualX->DrawLine(fId, fgHilightGC(), 2, 2, fWidth-3, 2);
   }
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   if (fState == kButtonDisabled) {
      fLabel->Draw(fId, fgHilightGC(), x+1, y+1 + max_ascent);
      fLabel->Draw(fId, fgShadowGC(), x, y + max_ascent);
   } else {
      fLabel->Draw(fId, fNormGC, x, y + max_ascent);
   }
}

//______________________________________________________________________________
Bool_t TGTextButton::HandleKey(Event_t *event)
{
   // Handle key event. This function will be called when the hotkey is hit.

   Bool_t click = kFALSE;

   if (fTip && event->fType == kGKeyPress) fTip->Hide();

   if (fState == kButtonDisabled) return kTRUE;

   // We don't need to check the key number as GrabKey will
   // only allow fHotchar events
   if (event->fType == kGKeyPress) {
      if (fState == kButtonEngaged) return kTRUE;
      SetState(kButtonDown);
   } else { // KeyRelease
      if (fState == kButtonEngaged /*&& !allowRelease*/) return kTRUE;
      click = (fState == kButtonDown);
      if (click && fStayDown)
         SetState(kButtonEngaged);
      else
         SetState(kButtonUp);
   }
   if (click) {
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_BUTTON), fWidgetId,
                  (Long_t) fUserData);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_BUTTON), fWidgetId,
                           (Long_t) fUserData);
   }

   return kTRUE;
}

//______________________________________________________________________________
FontStruct_t TGTextButton::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }


//______________________________________________________________________________
TGPictureButton::TGPictureButton(const TGWindow *p, const TGPicture *pic,
      Int_t id, GContext_t norm, UInt_t option) : TGButton(p, id, norm, option)
{
   // Create a picture button widget. The picture is not adopted and must
   // later be freed by the user once the picture button is deleted (a single
   // picture reference might be used by other buttons).

   if (!pic) {
      Error("TGPictureButton", "pixmap not found for button %d", id);
      return;
   }

   fPic = pic;

   fTWidth  = fPic->GetWidth();
   fTHeight = fPic->GetHeight();

   Resize(fTWidth  + (fBorderWidth << 1) + fBorderWidth + 1,
          fTHeight + (fBorderWidth << 1) + fBorderWidth); // *3
}

//______________________________________________________________________________
TGPictureButton::TGPictureButton(const TGWindow *p, const TGPicture *pic,
      const char *cmd, Int_t id, GContext_t norm, UInt_t option)
   : TGButton(p, id, norm, option)
{
   // Create a picture button widget and set action command. The picture is
   // not adopted and must later be freed by the user once the picture button
   // is deleted (a single picture reference might be used by other
   // buttons).

   if (!pic) {
      Error("TGPictureButton", "pixmap not found for button\n%s", cmd);
      return;
   }

   fPic     = pic;
   fCommand = cmd;

   fTWidth  = fPic->GetWidth();
   fTHeight = fPic->GetHeight();

   Resize(fTWidth  + (fBorderWidth << 1) + fBorderWidth + 1,
          fTHeight + (fBorderWidth << 1) + fBorderWidth); // *3
}

//______________________________________________________________________________
void TGPictureButton::SetPicture(const TGPicture *new_pic)
{
   // Change a picture in a picture button. The picture is not adopted and
   // must later be freed by the user once the picture button is deleted
   // (a single picture reference might be used by other buttons).

   if (!new_pic) {
      Error("SetPicture", "pixmap not found for button %d\n%s",
            fWidgetId, fCommand.Data());
      return;
   }

   fPic = new_pic;

   fTWidth  = fPic->GetWidth();
   fTHeight = fPic->GetHeight();

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGPictureButton::DoRedraw()
{
   // Redraw picture button.

   if (!fPic) return;

   int x = (fWidth - fTWidth) >> 1;
   int y = (fHeight - fTHeight) >> 1;

   TGFrame::DoRedraw();
   if (fState == kButtonDown || fState == kButtonEngaged) { ++x; ++y; }
   if (fState == kButtonEngaged) {
      gVirtualX->FillRectangle(fId, fgHibckgndGC(), 2, 2, fWidth-4, fHeight-4);
      gVirtualX->DrawLine(fId, fgHilightGC(), 2, 2, fWidth-3, 2);
   }
   fPic->Draw(fId, fNormGC, x, y);
}


//______________________________________________________________________________
TGCheckButton::TGCheckButton(const TGWindow *p, TGHotString *s, Int_t id,
                             GContext_t norm, FontStruct_t font, UInt_t option)
   : TGButton(p, id, norm, option)
{
   // Create a check button widget. The hotstring will be adopted and deleted
   // by the check button.

   fLabel = s;
   fFontStruct = font;

   Init();
}

//______________________________________________________________________________
TGCheckButton::TGCheckButton(const TGWindow *p, const char *s, Int_t id,
                             GContext_t norm, FontStruct_t font, UInt_t option)
   : TGButton(p, id, norm, option)
{
   // Create a check button widget.

   fLabel = new TGHotString(s);
   fFontStruct = font;

   Init();
}

//______________________________________________________________________________
TGCheckButton::TGCheckButton(const TGWindow *p, const char *s, const char *cmd,
                             Int_t id, GContext_t norm, FontStruct_t font,
                             UInt_t option) : TGButton(p, id, norm, option)
{
   // Create a check button widget.

   fLabel = new TGHotString(s);
   fFontStruct = font;
   fCommand = cmd;

   Init();
}

//______________________________________________________________________________
void TGCheckButton::Init()
{
   // Common initialization.

   fPrevState =
   fState     = kButtonUp;

   int hotchar, max_ascent, max_descent;
   fTWidth  = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   Resize(fTWidth + 22, fTHeight + 2);

   if ((hotchar = fLabel->GetHotChar()) != 0) {
      if ((fHKeycode = gVirtualX->KeysymToKeycode(hotchar)) != 0) {
         const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
         main->BindKey(this, fHKeycode, kKeyMod1Mask);
      }
   }
}

//______________________________________________________________________________
TGCheckButton::~TGCheckButton()
{
   // Delete a check button.

   if (fHKeycode) {
      const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
      main->RemoveBind(this, fHKeycode, kKeyMod1Mask);
   }
   if (fLabel) delete fLabel;
}

//______________________________________________________________________________
void TGCheckButton::PSetState(EButtonState state)
{
   // Set check button state.

   if (state != fState) {
      fState = state;
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
Bool_t TGCheckButton::HandleButton(Event_t *event)
{
   // Handle mouse button event.

   Bool_t click = kFALSE;

   if (fTip) fTip->Hide();

   if (fState == kButtonDisabled) return kTRUE;

   // We don't need to check the button number as GrabButton will
   // only allow button1 events
   if (event->fType == kButtonPress) {
      PSetState((fPrevState == kButtonUp) ? kButtonDown : kButtonUp);
   } else { // ButtonRelease
      click = (fState != fPrevState);
      fPrevState = fState;
   }
   if (click) {
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_CHECKBUTTON),
                  fWidgetId, (Long_t) fUserData);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_CHECKBUTTON),
                           fWidgetId, (Long_t) fUserData);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGCheckButton::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kEnterNotify)
         fTip->Reset();
      else
         fTip->Hide();
   }

   if (fgDbw != event->fWindow) return kTRUE;

   if (!(event->fState & (kButton1Mask | kButton2Mask | kButton3Mask)))
      return kTRUE;

   if (fState == kButtonDisabled) return kTRUE;

   if (event->fType == kEnterNotify) {
      PSetState((fPrevState == kButtonUp) ? kButtonDown : kButtonUp);
   } else {
      PSetState(fPrevState);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGCheckButton::HandleKey(Event_t *event)
{
   // Handle key event. This function will be called when the hotkey is hit.

   Bool_t click = kFALSE;

   if (fTip && event->fType == kGKeyPress) fTip->Hide();

   if (fState == kButtonDisabled) return kTRUE;

   // We don't need to check the key number as GrabKey will
   // only allow fHotchar events
   if (event->fType == kGKeyPress) {
      PSetState((fPrevState == kButtonUp) ? kButtonDown : kButtonUp);
   } else { // KeyRelease
      click = (fState != fPrevState);
      fPrevState = fState;
   }
   if (click) {
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_CHECKBUTTON), fWidgetId,
                  (Long_t) fUserData);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_CHECKBUTTON), fWidgetId,
                           (Long_t) fUserData);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGCheckButton::DoRedraw()
{
   // Draw the check button widget.

   int x, y, y0, cw;

   TGFrame::DoRedraw();

   cw = 13;
   y0 = (fHeight - cw) >> 1;

   gVirtualX->DrawLine(fId, fgShadowGC(), 0, y0, cw-2, y0);
   gVirtualX->DrawLine(fId, fgShadowGC(), 0, y0, 0, y0+cw-2);
   gVirtualX->DrawLine(fId, fgBlackGC(), 1, y0+1, cw-3, y0+1);
   gVirtualX->DrawLine(fId, fgBlackGC(), 1, y0+1, 1, y0+cw-3);

   gVirtualX->DrawLine(fId, fgHilightGC(), 0, y0+cw-1, cw-1, y0+cw-1);
   gVirtualX->DrawLine(fId, fgHilightGC(), cw-1, y0+cw-1, cw-1, y0);
   gVirtualX->DrawLine(fId, fgBckgndGC(),  2, y0+cw-2, cw-2, y0+cw-2);
   gVirtualX->DrawLine(fId, fgBckgndGC(),  cw-2, y0+2, cw-2, y0+cw-2);

   gVirtualX->FillRectangle(fId, fgWhiteGC(), 2, y0+2, cw-4, cw-4);

   if (fState == kButtonDown) {
      Segment_t seg[6];

      int l = 2;
      int t = y0+2;

      seg[0].fX1 = 1+l; seg[0].fY1 = 3+t; seg[0].fX2 = 3+l; seg[0].fY2 = 5+t;
      seg[1].fX1 = 1+l; seg[1].fY1 = 4+t; seg[1].fX2 = 3+l; seg[1].fY2 = 6+t;
      seg[2].fX1 = 1+l; seg[2].fY1 = 5+t; seg[2].fX2 = 3+l; seg[2].fY2 = 7+t;
      seg[3].fX1 = 3+l; seg[3].fY1 = 5+t; seg[3].fX2 = 7+l; seg[3].fY2 = 1+t;
      seg[4].fX1 = 3+l; seg[4].fY1 = 6+t; seg[4].fX2 = 7+l; seg[4].fY2 = 2+t;
      seg[5].fX1 = 3+l; seg[5].fY1 = 7+t; seg[5].fX2 = 7+l; seg[5].fY2 = 3+t;

      gVirtualX->DrawSegments(fId, fgBlackGC(), seg, 6);
   }

   x = 20;
   y = 1;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   if (fState == kButtonDisabled) {
     fLabel->Draw(fId, fgHilightGC(), x+1, y+1 + max_ascent);
     fLabel->Draw(fId, fgShadowGC(), x, y + max_ascent);
   } else {
     fLabel->Draw(fId, fNormGC, x, y + max_ascent);
   }
}

//______________________________________________________________________________
FontStruct_t TGCheckButton::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGCheckButton::GetDefaultGC()
{ return fgDefaultGC; }


//______________________________________________________________________________
TGRadioButton::TGRadioButton(const TGWindow *p, TGHotString *s, Int_t id,
                             GContext_t norm, FontStruct_t font, UInt_t option)
   : TGButton(p, id, norm, option)
{
   // Create a radio button widget. The hotstring will be adopted and deleted
   // by the radio button.

   fLabel = s;
   fFontStruct = font;

   Init();
}

//______________________________________________________________________________
TGRadioButton::TGRadioButton(const TGWindow *p, const char *s, Int_t id,
                             GContext_t norm, FontStruct_t font, UInt_t option)
   : TGButton(p, id, norm, option)
{
   // Create a radio button widget.

   fLabel = new TGHotString(s);
   fFontStruct = font;

   Init();
}
//______________________________________________________________________________
TGRadioButton::TGRadioButton(const TGWindow *p, const char *s, const char *cmd,
                             Int_t id, GContext_t norm,
                             FontStruct_t font, UInt_t option)
    : TGButton(p, id, norm, option)
{
   // Create a radio button widget.

   fLabel = new TGHotString(s);
   fFontStruct = font;
   fCommand = cmd;

   Init();
}

//______________________________________________________________________________
void TGRadioButton::Init()
{
   // Common radio button initialization.

   fPrevState =
   fState     = kButtonUp;

   fOn  = fClient->GetPicture("rbutton_on.xpm");
   fOff = fClient->GetPicture("rbutton_off.xpm");

   if (!fOn || !fOff)
      Error("TGRadioButton", "rbutton_*.xpm not found");

   int hotchar, max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   Resize(fTWidth + 22, fTHeight + 2);

   if ((hotchar = fLabel->GetHotChar()) != 0) {
      if ((fHKeycode = gVirtualX->KeysymToKeycode(hotchar)) != 0) {
         const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
         main->BindKey(this, fHKeycode, kKeyMod1Mask);
      }
   }
}

//______________________________________________________________________________
TGRadioButton::~TGRadioButton()
{
   // Delete a radio button.

   if (fHKeycode) {
      const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
      main->RemoveBind(this, fHKeycode, kKeyMod1Mask);
   }
   if (fOn)  fClient->FreePicture(fOn);
   if (fOff) fClient->FreePicture(fOff);
   if (fLabel) delete fLabel;
}

//______________________________________________________________________________
void TGRadioButton::PSetState(EButtonState state)
{
   // Set radio button state.

   if (state != fState) {
      fPrevState = fState = state;
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
Bool_t TGRadioButton::HandleButton(Event_t *event)
{
   // Handle mouse button event.

   if (fTip) fTip->Hide();

   if (fState == kButtonDisabled) return kTRUE;

   // We don't need to check the button number as GrabButton will
   // only allow button1 events
   if (event->fType == kButtonPress) {
      PSetState(kButtonDown);
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_RADIOBUTTON),
                  fWidgetId, (Long_t) fUserData);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_RADIOBUTTON),
                           fWidgetId, (Long_t) fUserData);
   } else { // ButtonRelease
      fPrevState = fState;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGRadioButton::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kEnterNotify)
         fTip->Reset();
      else
         fTip->Hide();
   }

   if (fgDbw != event->fWindow) return kTRUE;

   if (!(event->fState & (kButton1Mask | kButton2Mask | kButton3Mask)))
      return kTRUE;

   if (fState == kButtonDisabled) return kTRUE;

   if (event->fType == kEnterNotify) {
      PSetState(kButtonDown);
   } else {
      PSetState(fPrevState);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGRadioButton::HandleKey(Event_t *event)
{
   // Handle key event. This function will be called when the hotkey is hit.

   if (fTip && event->fType == kGKeyPress) fTip->Hide();

   if (fState == kButtonDisabled) return kTRUE;

   // We don't need to check the key number as GrabKey will
   // only allow fHotchar events
   if (event->fType == kGKeyPress) {
      PSetState(kButtonDown);
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_RADIOBUTTON),
                  fWidgetId, (Long_t) fUserData);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_RADIOBUTTON),
                           fWidgetId, (Long_t) fUserData);
   } else { // KeyRelease
      fPrevState = fState;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGRadioButton::DoRedraw()
{
   // Draw a radio button.

   int nlines, tx, ty, y0, pw;

   TGFrame::DoRedraw();

   tx = 20;
   nlines = fLabel->GetLines(fFontStruct, fWidth-tx-1);
   ty = (fHeight - fTHeight*nlines) >> 1;

   pw = 12;
   y0 = ty + ((fTHeight - pw) >> 1);

   if (fState == kButtonDown) {
      if (fOn) fOn->Draw(fId, fNormGC, 0, y0);
   } else {
      if (fOff) fOff->Draw(fId, fNormGC, 0, y0);
   }

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   // ty = max_ascent + 1;
   ty += max_ascent;

   //  fLabel->Draw(fId, fNormGC, tx, ty);
   if (fState == kButtonDisabled) {
      fLabel->DrawWrapped(fId, fgHilightGC(), tx+1, ty+1, fWidth-tx-1, fFontStruct);
      fLabel->DrawWrapped(fId, fgShadowGC(), tx, ty, fWidth-tx-1, fFontStruct);
   } else {
      fLabel->DrawWrapped(fId, fNormGC, tx, ty, fWidth-tx-1, fFontStruct);
   }
}

//______________________________________________________________________________
FontStruct_t TGRadioButton::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGRadioButton::GetDefaultGC()
{ return fgDefaultGC; }
