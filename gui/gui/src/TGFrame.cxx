// @(#)root/gui:$Id$
// Author: Fons Rademakers   03/01/98

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
// TGFrame, TGCompositeFrame, TGVerticalFrame, TGHorizontalFrame,       //
// TGMainFrame, TGTransientFrame and TGGroupFrame                       //
//                                                                      //
// The frame classes describe the different "dressed" GUI windows.      //
//                                                                      //
// The TGFrame class is a subclasses of TGWindow, and is used as base   //
// class for some simple widgets (buttons, labels, etc.).               //
// It provides:                                                         //
//  - position & dimension fields                                       //
//  - an 'options' attribute (see constant above)                       //
//  - a generic event handler                                           //
//  - a generic layout mechanism                                        //
//  - a generic border                                                  //
//                                                                      //
// The TGCompositeFrame class is the base class for composite widgets   //
// (menu bars, list boxes, etc.).                                       //
// It provides:                                                         //
//  - a layout manager                                                  //
//  - a frame container (TList *)                                       //
//                                                                      //
// The TGVerticalFrame and TGHorizontalFrame are composite frame that   //
// layout their children in vertical or horizontal way.                 //
//                                                                      //
// The TGMainFrame class defines top level windows that interact with   //
// the system Window Manager.                                           //
//                                                                      //
// The TGTransientFrame class defines transient windows that typically  //
// are used for dialogs windows.                                        //
//                                                                      //
// The TGGroupFrame is a composite frame with a border and a title.     //
// It is typically used to group a number of logically related widgets  //
// visually together.                                                   //
//                                                                      //
//Begin_Html
/*
<img src="gif/tgcompositeframe_classtree.gif">
*/
//End_Html
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TGFrame.h"
#include "TGResourcePool.h"
#include "TGPicture.h"
#include "TList.h"
#include "TApplication.h"
#include "TTimer.h"
#include "Riostream.h"
#include "TClass.h"

#include "TObjString.h"
#include "TObjArray.h"
#include "TBits.h"
#include "TColor.h"
#include "TROOT.h"
#include "KeySymbols.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TSystem.h"
#include "TVirtualDragManager.h"
#include "TGuiBuilder.h"
#include "TQConnection.h"
#include "TGButton.h"
#include "TGSplitter.h"
#include "TGDNDManager.h"
#include "TImage.h"
#include "TObjectSpy.h"


Bool_t      TGFrame::fgInit = kFALSE;
Pixel_t     TGFrame::fgDefaultFrameBackground = 0;
Pixel_t     TGFrame::fgDefaultSelectedBackground = 0;
Pixel_t     TGFrame::fgWhitePixel = 0;
Pixel_t     TGFrame::fgBlackPixel = 0;
const TGGC *TGFrame::fgBlackGC = 0;
const TGGC *TGFrame::fgWhiteGC = 0;
const TGGC *TGFrame::fgHilightGC = 0;
const TGGC *TGFrame::fgShadowGC = 0;
const TGGC *TGFrame::fgBckgndGC = 0;
Time_t      TGFrame::fgLastClick = 0;
UInt_t      TGFrame::fgLastButton = 0;
Int_t       TGFrame::fgDbx = 0;
Int_t       TGFrame::fgDby = 0;
Window_t    TGFrame::fgDbw = 0;
UInt_t      TGFrame::fgUserColor = 0;

const TGFont *TGGroupFrame::fgDefaultFont = 0;
const TGGC   *TGGroupFrame::fgDefaultGC = 0;

TGLayoutHints *TGCompositeFrame::fgDefaultHints = 0;

static const char *gSaveMacroTypes[] = {
   "ROOT macros", "*.C",
   "GIF",         "*.gif",
   "PNG",         "*.png",
   "JPEG",        "*.jpg",
   "TIFF",        "*.tiff",
   "XPM",         "*.xpm",
   "All files",   "*",
   0,             0
};

TList *gListOfHiddenFrames = new TList();

ClassImp(TGFrame)
ClassImp(TGCompositeFrame)
ClassImp(TGVerticalFrame)
ClassImp(TGHorizontalFrame)
ClassImp(TGMainFrame)
ClassImp(TGTransientFrame)
ClassImp(TGGroupFrame)
ClassImp(TGHeaderFrame)


//______________________________________________________________________________
TGFrame::TGFrame(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options, Pixel_t back)
   : TGWindow(p, 0, 0, w, h, 0, 0, 0, 0, 0, options)
{
   // Create a TGFrame object. Options is an OR of the EFrameTypes.

   if (!fgInit && gClient) {
      TGFrame::GetDefaultFrameBackground();
      TGFrame::GetDefaultSelectedBackground();
      TGFrame::GetWhitePixel();
      TGFrame::GetBlackPixel();
      TGFrame::GetBlackGC();
      TGFrame::GetWhiteGC();
      TGFrame::GetHilightGC();
      TGFrame::GetShadowGC();
      TGFrame::GetBckgndGC();
      fgInit = kTRUE;
   }

   SetWindowAttributes_t wattr;

   fDNDState   = 0;
   fBackground = back;
   fOptions    = options;
   fWidth = w; fHeight = h; fX = fY = fBorderWidth = 0;
   fMinWidth    = 0;
   fMinHeight   = 0;
   fMaxWidth    = kMaxUInt;
   fMaxHeight   = kMaxUInt;
   fFE          = 0;

   if (fOptions & (kSunkenFrame | kRaisedFrame))
      fBorderWidth = (fOptions & kDoubleBorder) ? 2 : 1;

   wattr.fMask = kWABackPixel | kWAEventMask;
   wattr.fBackgroundPixel = back;
   wattr.fEventMask = kExposureMask;
   if (fOptions & kMainFrame) {
      wattr.fEventMask |= kStructureNotifyMask;
      gVirtualX->ChangeWindowAttributes(fId, &wattr);
      //if (fgDefaultBackgroundPicture)
      //   SetBackgroundPixmap(fgDefaultBackgroundPicture->GetPicture());
   } else {
      gVirtualX->ChangeWindowAttributes(fId, &wattr);
      //if (!(fOptions & kOwnBackground))
      //   SetBackgroundPixmap(kParentRelative);
   }
   fEventMask = (UInt_t) wattr.fEventMask;

   SetWindowName();
}

//______________________________________________________________________________
TGFrame::TGFrame(TGClient *c, Window_t id, const TGWindow *parent)
   : TGWindow(c, id, parent)
{
   // Create a frame using an externally created window. For example
   // to register the root window (called by TGClient), or a window
   // created via TVirtualX::InitWindow() (id is obtained with
   // TVirtualX::GetWindowID()).

   if (!fgInit && gClient) {
      TGFrame::GetDefaultFrameBackground();
      TGFrame::GetDefaultSelectedBackground();
      TGFrame::GetWhitePixel();
      TGFrame::GetBlackPixel();
      TGFrame::GetBlackGC();
      TGFrame::GetWhiteGC();
      TGFrame::GetHilightGC();
      TGFrame::GetShadowGC();
      TGFrame::GetBckgndGC();
      fgInit = kTRUE;
   }

   WindowAttributes_t attributes;
   // Initialize some values - needed for batch mode!
   attributes.fX      = 0;
   attributes.fY      = 0;
   attributes.fWidth  = 100;
   attributes.fHeight = 100;
   attributes.fBorderWidth = 4;
   attributes.fYourEventMask = 0;
   gVirtualX->GetWindowAttributes(id, attributes);

   fDNDState    = 0;
   fX           = attributes.fX;
   fY           = attributes.fY;
   fWidth       = attributes.fWidth;
   fHeight      = attributes.fHeight;
   fBorderWidth = attributes.fBorderWidth;
   fEventMask   = (UInt_t) attributes.fYourEventMask;
   fBackground  = 0;
   fOptions     = 0;
   fMinWidth    = 0;
   fMinHeight   = 0;
   fMaxWidth    = kMaxUInt;
   fMaxHeight   = kMaxUInt;
   fFE          = 0;

   SetWindowName();
}

//______________________________________________________________________________
TGFrame::~TGFrame()
{
   // Destructor.

}

//______________________________________________________________________________
void TGFrame::DeleteWindow()
{
   // Delete window. Use single shot timer to call final delete method.
   // We use this indirect way since deleting the window in its own
   // execution "thread" can cause side effects because frame methods
   // can still be called while the window object has already been deleted.

   if (gDNDManager) {
      if (gDNDManager->GetMainFrame() == this)
         gDNDManager->SetMainFrame(0);
   }
   if (!TestBit(kDeleteWindowCalled))
      TTimer::SingleShot(150, IsA()->GetName(), this, "ReallyDelete()");
   SetBit(kDeleteWindowCalled);
}

//______________________________________________________________________________
void TGFrame::ChangeBackground(Pixel_t back)
{
   // Change frame background color.

   fBackground = back;
   gVirtualX->SetWindowBackground(fId, back);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
Pixel_t TGFrame::GetForeground() const
{
   // Return frame foreground color.

   return fgBlackPixel;
}

//______________________________________________________________________________
void TGFrame::SetBackgroundColor(Pixel_t back)
{
   // Set background color (override from TGWindow base class).
   // Same effect as ChangeBackground().

   fBackground = back;
   TGWindow::SetBackgroundColor(back);
}

//______________________________________________________________________________
void TGFrame::ChangeOptions(UInt_t options)
{
   // Change frame options. Options is an OR of the EFrameTypes.

   if ((options & (kDoubleBorder | kSunkenFrame | kRaisedFrame)) !=
      (fOptions & (kDoubleBorder | kSunkenFrame | kRaisedFrame))) {
      if (!InheritsFrom(TGGroupFrame::Class())) {
         if (options & (kSunkenFrame | kRaisedFrame))
            fBorderWidth = (options & kDoubleBorder) ? 2 : 1;
         else
            fBorderWidth = 0;
      }
   }

   fOptions = options;
}

//______________________________________________________________________________
void TGFrame::AddInput(UInt_t emask)
{
   // Add events specified in the emask to the events the frame should handle.

   fEventMask |= emask;
   gVirtualX->SelectInput(fId, fEventMask);
}

//______________________________________________________________________________
void TGFrame::RemoveInput(UInt_t emask)
{
   // Remove events specified in emask from the events the frame should handle.

   fEventMask &= ~emask;
   gVirtualX->SelectInput(fId, fEventMask);
}

//________________________________________________________________________________
void TGFrame::Draw3dRectangle(UInt_t type, Int_t x, Int_t y,
                              UInt_t w, UInt_t h)
{
   // Draw 3D rectangle on the frame border.

   switch (type) {
      case kSunkenFrame:
         gVirtualX->DrawLine(fId, GetShadowGC()(),  x,     y,     x+w-2, y);
         gVirtualX->DrawLine(fId, GetShadowGC()(),  x,     y,     x,     y+h-2);
         gVirtualX->DrawLine(fId, GetHilightGC()(), x,     y+h-1, x+w-1, y+h-1);
         gVirtualX->DrawLine(fId, GetHilightGC()(), x+w-1, y+h-1, x+w-1, y);
         break;

      case kSunkenFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, GetShadowGC()(), x,     y,     x+w-2, y);
         gVirtualX->DrawLine(fId, GetShadowGC()(), x,     y,     x,     y+h-2);
         gVirtualX->DrawLine(fId, GetBlackGC()(),  x+1,   y+1,   x+w-3, y+1);
         gVirtualX->DrawLine(fId, GetBlackGC()(),  x+1,   y+1,   x+1,   y+h-3);

         gVirtualX->DrawLine(fId, GetHilightGC()(), x,     y+h-1, x+w-1, y+h-1);
         gVirtualX->DrawLine(fId, GetHilightGC()(), x+w-1, y+h-1, x+w-1, y);
         gVirtualX->DrawLine(fId, GetBckgndGC()(),  x+1,   y+h-2, x+w-2, y+h-2);
         gVirtualX->DrawLine(fId, GetBckgndGC()(),  x+w-2, y+1,   x+w-2, y+h-2);
         break;

      case kRaisedFrame:
         gVirtualX->DrawLine(fId, GetHilightGC()(), x,     y,     x+w-2, y);
         gVirtualX->DrawLine(fId, GetHilightGC()(), x,     y,     x,     y+h-2);
         gVirtualX->DrawLine(fId, GetShadowGC()(),  x,     y+h-1, x+w-1, y+h-1);
         gVirtualX->DrawLine(fId, GetShadowGC()(),  x+w-1, y+h-1, x+w-1, y);
         break;

      case kRaisedFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, GetHilightGC()(), x,     y,     x+w-2, y);
         gVirtualX->DrawLine(fId, GetHilightGC()(), x,     y,     x,     y+h-2);
         gVirtualX->DrawLine(fId, GetBckgndGC()(),  x+1,   y+1,   x+w-3, y+1);
         gVirtualX->DrawLine(fId, GetBckgndGC()(),  x+1,   y+1,   x+1,   y+h-3);

         gVirtualX->DrawLine(fId, GetShadowGC()(),  x+1,   y+h-2, x+w-2, y+h-2);
         gVirtualX->DrawLine(fId, GetShadowGC()(),  x+w-2, y+h-2, x+w-2, y+1);
         gVirtualX->DrawLine(fId, GetBlackGC()(),   x,     y+h-1, x+w-1, y+h-1);
         gVirtualX->DrawLine(fId, GetBlackGC()(),   x+w-1, y+h-1, x+w-1, y);
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void TGFrame::DrawBorder()
{
   // Draw frame border.

   Draw3dRectangle(fOptions & (kSunkenFrame | kRaisedFrame | kDoubleBorder),
                   0, 0, fWidth, fHeight);
}

//______________________________________________________________________________
void TGFrame::DoRedraw()
{
   // Redraw the frame.

   gVirtualX->ClearArea(fId, fBorderWidth, fBorderWidth,
                   fWidth - (fBorderWidth << 1), fHeight - (fBorderWidth << 1));

   // border will only be drawn if we have a 3D option hint
   // (kRaisedFrame or kSunkenFrame)
   DrawBorder();
}

//______________________________________________________________________________
Bool_t TGFrame::HandleConfigureNotify(Event_t *event)
{
   // This event is generated when the frame is resized.

   if ((event->fWidth != fWidth) || (event->fHeight != fHeight)) {
      fWidth  = event->fWidth;
      fHeight = event->fHeight;
      Layout();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGFrame::HandleEvent(Event_t *event)
{
   // Handle all frame events. Events are dispatched to the specific
   // event handlers.

   if (gDragManager && !fClient->IsEditDisabled() &&
       gDragManager->HandleEvent(event)) return kTRUE;

   TObjectSpy deleteCheck(this);

   switch (event->fType) {

      case kExpose:
         HandleExpose(event);
         break;

      case kConfigureNotify:
         while (gVirtualX->CheckEvent(fId, kConfigureNotify, *event))
            ;
         // protection
         if ((event->fWidth < 32768) && (event->fHeight  < 32768)){
            ProcessedConfigure(event);  // emit signal
            HandleConfigureNotify(event);
         }
         break;

      case kGKeyPress:
      case kKeyRelease:
         HandleKey(event);
         break;

      case kFocusIn:
      case kFocusOut:
         HandleFocusChange(event);
         break;

      case kButtonPress:
         {
            Int_t dbl_clk = kFALSE;

            if ((event->fTime - fgLastClick < 350) &&
                (event->fCode == fgLastButton) &&
                (TMath::Abs(event->fXRoot - fgDbx) < 6) &&
                (TMath::Abs(event->fYRoot - fgDby) < 6) &&
                (event->fWindow == fgDbw))
               dbl_clk = kTRUE;

            fgLastClick  = event->fTime;
            fgLastButton = event->fCode;
            fgDbx = event->fXRoot;
            fgDby = event->fYRoot;
            fgDbw = event->fWindow;

            if (dbl_clk) {
               if ((event->fState & kKeyControlMask) &&
                    !GetEditDisabled() && gGuiBuilder) {
                  StartGuiBuilding(!IsEditable());
                  return kTRUE;
               }

               if (!HandleDoubleClick(event)) {
                  HandleButton(event);
               }
            } else {
               HandleButton(event);
            }
         }
         break;

      case kButtonDoubleClick:
         {
            fgLastClick  = event->fTime;
            fgLastButton = event->fCode;
            fgDbx = event->fXRoot;
            fgDby = event->fYRoot;
            fgDbw = event->fWindow;

            HandleDoubleClick(event);
         }
         break;

      case kButtonRelease:
         HandleButton(event);
         break;

      case kEnterNotify:
      case kLeaveNotify:
         HandleCrossing(event);
         break;

      case kMotionNotify:
         while (gVirtualX->CheckEvent(fId, kMotionNotify, *event))
            ;
         HandleMotion(event);
         break;

      case kClientMessage:
         HandleClientMessage(event);
         break;

      case kSelectionNotify:
         HandleSelection(event);
         break;

      case kSelectionRequest:
         HandleSelectionRequest(event);
         break;

      case kSelectionClear:
         HandleSelectionClear(event);
         break;

      case kColormapNotify:
         HandleColormapChange(event);
         break;

      default:
         //Warning("HandleEvent", "unknown event (%#x) for (%#x)", event->fType, fId);
         break;
   }

   if (deleteCheck.GetObject())
      ProcessedEvent(event);  // emit signal

   return kTRUE;
}

//______________________________________________________________________________
TGDimension TGFrame::GetDefaultSize() const
{
   //   std::cout << fWidth << "x" << fHeight << std::endl;
   return TGDimension(fWidth, fHeight);
}


//______________________________________________________________________________
void TGFrame::Move(Int_t x, Int_t y)
{
   // Move frame.

   if (x != fX || y != fY) {
      TGWindow::Move(x, y);
      fX = x; fY = y;
   }
}

//______________________________________________________________________________
void TGFrame::Resize(UInt_t w, UInt_t h)
{
   // Resize the frame.
   // If w=0 && h=0 - Resize to deafult size

   if (w != fWidth || h != fHeight) {
      TGDimension siz(0,0);
      siz = GetDefaultSize();
      fWidth = w ? w : siz.fWidth;
      fHeight = h ? h : siz.fHeight;
      TGWindow::Resize(fWidth, fHeight);
      Layout();
   }
}

//______________________________________________________________________________
void TGFrame::Resize(TGDimension size)
{
   // Resize the frame.

   Resize(size.fWidth, size.fHeight);
}

//______________________________________________________________________________
void TGFrame::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and/or resize the frame.
   // If w=0 && h=0 - Resize to deafult size

   // we do it anyway as we don't know if it's only a move or only a resize
   TGDimension siz(0,0);
   siz = GetDefaultSize();
   fWidth = w ? w : siz.fWidth;
   fHeight = h ? h : siz.fHeight;
   fX = x; fY = y;
   TGWindow::MoveResize(x, y, fWidth, fHeight);
   Layout();
}

//______________________________________________________________________________
void TGFrame::SendMessage(const TGWindow *w, Long_t msg, Long_t parm1, Long_t parm2)
{
   // Send message (i.e. event) to window w. Message is encoded in one long
   // as message type and up to two long parameters.

   Event_t event;

   if (w) {
      event.fType   = kClientMessage;
      event.fFormat = 32;
      event.fHandle = gROOT_MESSAGE;

      event.fWindow  = w->GetId();
      event.fUser[0] = msg;
      event.fUser[1] = parm1;
      event.fUser[2] = parm2;
      event.fUser[3] = 0;
      event.fUser[4] = 0;

      gVirtualX->SendEvent(w->GetId(), &event);
   }
}

//______________________________________________________________________________
Bool_t TGFrame::HandleClientMessage(Event_t *event)
{
   // Handle a client message. Client messages are the ones sent via
   // TGFrame::SendMessage (typically by widgets).

   if (gDNDManager) {
      gDNDManager->HandleClientMessage(event);
   }
   if (event->fHandle == gROOT_MESSAGE) {
      ProcessMessage(event->fUser[0], event->fUser[1], event->fUser[2]);
   }
   return kTRUE;
}

//______________________________________________________________________________
ULong_t TGFrame::GetDefaultFrameBackground()
{
   // Get default frame background.

   static Bool_t init = kFALSE;
   if (!init && gClient) {
      fgDefaultFrameBackground = gClient->GetResourcePool()->GetFrameBgndColor();
      init = kTRUE;
   }
   return fgDefaultFrameBackground;
}

//______________________________________________________________________________
ULong_t TGFrame::GetDefaultSelectedBackground()
{
   // Get default selected frame background.

   static Bool_t init = kFALSE;
   if (!init && gClient) {
      fgDefaultSelectedBackground = gClient->GetResourcePool()->GetSelectedBgndColor();
      init = kTRUE;
   }
   return fgDefaultSelectedBackground;
}

//______________________________________________________________________________
ULong_t TGFrame::GetWhitePixel()
{
   // Get white pixel value.

   static Bool_t init = kFALSE;
   if (!init && gClient) {
      fgWhitePixel = gClient->GetResourcePool()->GetWhiteColor();
      init  = kTRUE;
   }
   return fgWhitePixel;
}

//______________________________________________________________________________
ULong_t TGFrame::GetBlackPixel()
{
   // Get black pixel value.

   static Bool_t init = kFALSE;
   if (!init && gClient) {
      fgBlackPixel = gClient->GetResourcePool()->GetBlackColor();
      init = kTRUE;
   }
   return fgBlackPixel;
}

//______________________________________________________________________________
const TGGC &TGFrame::GetBlackGC()
{
   // Get black graphics context.

   if (!fgBlackGC && gClient)
      fgBlackGC = gClient->GetResourcePool()->GetBlackGC();
   return *fgBlackGC;
}

//______________________________________________________________________________
const TGGC &TGFrame::GetWhiteGC()
{
   // Get white graphics context.

   if (!fgWhiteGC && gClient)
      fgWhiteGC = gClient->GetResourcePool()->GetWhiteGC();
   return *fgWhiteGC;
}

//______________________________________________________________________________
const TGGC &TGFrame::GetHilightGC()
{
   // Get highlight color graphics context.

   if (!fgHilightGC && gClient)
      fgHilightGC = gClient->GetResourcePool()->GetFrameHiliteGC();
   return *fgHilightGC;
}

//______________________________________________________________________________
const TGGC &TGFrame::GetShadowGC()
{
   // Get shadow color graphics context.

   if (!fgShadowGC && gClient)
      fgShadowGC = gClient->GetResourcePool()->GetFrameShadowGC();
   return *fgShadowGC;
}

//______________________________________________________________________________
const TGGC &TGFrame::GetBckgndGC()
{
   // Get background color graphics context.

   if (!fgBckgndGC && gClient)
      fgBckgndGC = gClient->GetResourcePool()->GetFrameBckgndGC();
   return *fgBckgndGC;
}

//______________________________________________________________________________
Time_t TGFrame::GetLastClick()
{
   // Get time of last mouse click.

   return fgLastClick;
}

//______________________________________________________________________________
void TGFrame::Print(Option_t *option) const
{
   // Print window id.

   TString opt = option;
   if (opt.Contains("tree")) {
      TGWindow::Print(option);
      return;
   }

   cout <<  option << ClassName() << ":\tid=" << fId << " parent=" << fParent->GetId();
   cout << " x=" << fX << " y=" << fY;
   cout << " w=" << fWidth << " h=" << fHeight << endl;
}

//______________________________________________________________________________
void TGFrame::SetDragType(Int_t)
{
   // SetDragType
}

//______________________________________________________________________________
void TGFrame::SetDropType(Int_t)
{
   // SetDropType
}

//______________________________________________________________________________
Int_t TGFrame::GetDragType() const
{
   // Returns drag source type.
   // If frame is not "draggable" - return zero

   return fClient->IsEditable();
}

//______________________________________________________________________________
Int_t TGFrame::GetDropType() const
{
   // Returns drop target type.
   // If frame cannot accept drop - return zero

   return 0;
}

//______________________________________________________________________________
void TGFrame::StartGuiBuilding(Bool_t on)
{
   // Go into GUI building mode.

   if (GetEditDisabled()) return;
   if (!gDragManager) gDragManager = TVirtualDragManager::Instance();
   if (!gDragManager) return;

   TGCompositeFrame *comp = 0;

   if (InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame *)this;
   } else if (fParent->InheritsFrom(TGCompositeFrame::Class())) {
      comp = (TGCompositeFrame*)fParent;
   }
   if (comp) comp->SetEditable(on);
}

//______________________________________________________________________________
TGCompositeFrame::TGCompositeFrame(const TGWindow *p, UInt_t w, UInt_t h,
         UInt_t options, Pixel_t back) : TGFrame(p, w, h, options, back)
{
   // Create a composite frame. A composite frame has in addition to a TGFrame
   // also a layout manager and a list of child frames.

   fLayoutManager = 0;
   fList          = new TList;
   fLayoutBroken  = kFALSE;
   fMustCleanup   = kNoCleanup;
   fMapSubwindows = fParent->IsMapSubwindows();
   if (!fgDefaultHints)
      fgDefaultHints = new TGLayoutHints;

   if (fOptions & kHorizontalFrame)
      SetLayoutManager(new TGHorizontalLayout(this));
   else
      SetLayoutManager(new TGVerticalLayout(this));

   SetWindowName();
}

//______________________________________________________________________________
TGCompositeFrame::TGCompositeFrame(TGClient *c, Window_t id, const TGWindow *parent)
   : TGFrame(c, id, parent)
{
   // Create a frame using an externally created window. For example
   // to register the root window (called by TGClient), or a window
   // created via TVirtualX::InitWindow() (id is obtained with TVirtualX::GetWindowID()).

   fLayoutManager = 0;
   fList          = new TList;
   fLayoutBroken  = kFALSE;
   fMustCleanup   = kNoCleanup;
   fMapSubwindows = fParent->IsMapSubwindows();
   if (!fgDefaultHints)
      fgDefaultHints = new TGLayoutHints;

   SetLayoutManager(new TGVerticalLayout(this));

   SetWindowName();
}

//______________________________________________________________________________
TGCompositeFrame::~TGCompositeFrame()
{
   // Delete a composite frame.

   if (fMustCleanup != kNoCleanup) {
      Cleanup();
   } else {
      TGFrameElement *el = 0;
      TIter next(fList);

      while ((el = (TGFrameElement *) next())) {
         fList->Remove(el);
         delete el;
      }
   }

   delete fList;
   delete fLayoutManager;
   fList = 0;
   fLayoutManager = 0;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::IsEditable() const
{
   // Return kTRUE if frame is being edited.

   return (fClient->GetRoot() == (TGWindow*)this);
}

//______________________________________________________________________________
void TGCompositeFrame::SetEditable(Bool_t on)
{
   // Switch ON/OFF edit mode.
   // If edit mode is ON it is possible:
   //
   //  1. embed other ROOT GUI application (a la ActiveX)
   //
   //  For example:
   //    TGMainFrame *m = new TGMainFrame(gClient->GetRoot(), 500, 500);
   //    m->SetEditable();
   //    gSystem->Load("$ROOTSYS/test/Aclock"); // load Aclock demo
   //    Aclock a;
   //    gROOT->Macro("$ROOTSYS/tutorials/gui/guitest.C");
   //    m->SetEditable(0);
   //    m->MapWindow();
   //

   if (on && ((fEditDisabled & kEditDisable) ||
              (fEditDisabled & kEditDisableLayout))) return;

   if (on) {
      fClient->SetRoot(this);
   } else {
      fClient->SetRoot(0);
   }
   if (gDragManager) gDragManager->SetEditable(on);
}

//______________________________________________________________________________
void TGCompositeFrame::Cleanup()
{
   // Cleanup and delete all objects contained in this composite frame.
   // This will delete all objects added via AddFrame().
   // CAUTION: all objects (frames and layout hints) must be unique, i.e.
   // cannot be shared.

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if (el->fFrame) {
         el->fFrame->SetFrameElement(0);
         if (!gVirtualX->InheritsFrom("TGX11"))
            el->fFrame->DestroyWindow();
         delete el->fFrame;
      }

      if (el->fLayout && (el->fLayout != fgDefaultHints) &&
          (el->fLayout->References() > 0)) {
         el->fLayout->RemoveReference();
         if (!el->fLayout->References()) {
            el->fLayout->fFE = 0;
            delete el->fLayout;
         }
      }
      fList->Remove(el);
      delete el;
   }
}

//______________________________________________________________________________
void TGCompositeFrame::SetLayoutManager(TGLayoutManager *l)
{
   // Set the layout manager for the composite frame.
   // The layout manager is adopted by the frame and will be deleted
   // by the frame.

   if (l) {
      delete fLayoutManager;
      fLayoutManager = l;
   } else
      Error("SetLayoutManager", "no layout manager specified");
}

//______________________________________________________________________________
void TGCompositeFrame::SetLayoutBroken(Bool_t on)
{
   // Set broken layout. No Layout method is called.

   fLayoutBroken = on;
}

//______________________________________________________________________________
void TGCompositeFrame::SetEditDisabled(UInt_t on)
{
   // Set edit disable flag for this frame and subframes
   //
   //  - if (on & kEditDisable) - disable edit for this frame and all subframes.

   fEditDisabled = on;
   UInt_t set = on & kEditDisable;

   // propagate only kEditDisable
   if (set == kEditDisable) {

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         if (el->fFrame) {
            el->fFrame->SetEditDisabled(set);
         }
      }
   }
}

//______________________________________________________________________________
void TGCompositeFrame::ChangeOptions(UInt_t options)
{
   // Change composite frame options. Options is an OR of the EFrameTypes.

   TGFrame::ChangeOptions(options);

   if (options & kHorizontalFrame)
      SetLayoutManager(new TGHorizontalLayout(this));
   else if (options & kVerticalFrame)
      SetLayoutManager(new TGVerticalLayout(this));
}

//______________________________________________________________________________
void TGCompositeFrame::SetCleanup(Int_t mode)
{
   // Turn on automatic cleanup of child frames in dtor.
   //
   // if mode = kNoCleanup    - no automatic cleanup
   // if mode = kLocalCleanup - automatic cleanup in this composite frame only
   // if mode = kDeepCleanup  - automatic deep cleanup in this composite frame
   //                           and all child composite frames (hierarchical)
   //
   // Attention!
   //    Hierarchical cleaning is dangerous and must be used with caution.
   //    There are many GUI components (in ROOT and in user code) which do not
   //    use Clean method in destructor ("custom deallocation").
   //    Adding such component to GUI container which is using hierarchical
   //    cleaning will produce seg. violation when container is deleted.
   //    The reason is double deletion: first whem Clean method is invoked,
   //    then at "custom deallocation".
   //    We are going to correct all ROOT code to make it to be
   //    consitent with hierarchical cleaning scheeme.

   if (mode == fMustCleanup)
      return;

   fMustCleanup = mode;

   if (fMustCleanup == kDeepCleanup) {
      TGFrameElement *el;
      TIter next(fList);

      while ((el = (TGFrameElement *) next())) {
         if (el->fFrame->InheritsFrom(TGCompositeFrame::Class())) {
            el->fFrame->SetCleanup(kDeepCleanup);
         }
      }
   }
}

//______________________________________________________________________________
TGFrameElement* TGCompositeFrame::FindFrameElement(TGFrame *f) const
{
   // Find frame-element holding frame f.

   if (!fList) return 0;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f)
         return el;

   return 0;
}

//______________________________________________________________________________
void TGCompositeFrame::AddFrame(TGFrame *f, TGLayoutHints *l)
{
   // Add frame to the composite frame using the specified layout hints.
   // If no hints are specified default hints TGLayoutHints(kLHintsNormal,0,0,0,0)
   // will be used. Most of the time, however, you will want to provide
   // specific hints. User specified hints can be reused many times
   // and need to be destroyed by the user. The added frames cannot not be
   // added to different composite frames but still need to be deleted by
   // the user.

   TGFrameElement *nw = new TGFrameElement(f, l ? l : fgDefaultHints);
   fList->Add(nw);

   // in case of recusive cleanup, propagate cleanup setting to all
   // child composite frames
   if (fMustCleanup == kDeepCleanup)
      f->SetCleanup(kDeepCleanup);
}

//______________________________________________________________________________
void TGCompositeFrame::RemoveAll()
{
   // Remove all frames from composite frame.

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      fList->Remove(el);
      if (el->fLayout) el->fLayout->RemoveReference();
//       el->fFrame->SetFrameElement(0);
      delete el;
   }
}

//______________________________________________________________________________
void TGCompositeFrame::RemoveFrame(TGFrame *f)
{
   // Remove frame from composite frame.

   TGFrameElement *el = FindFrameElement(f);

   if (el) {
      fList->Remove(el);
      if (el->fLayout) el->fLayout->RemoveReference();
      f->SetFrameElement(0);
      delete el;
   }
}

//______________________________________________________________________________
void TGCompositeFrame::MapSubwindows()
{
   // Map all sub windows that are part of the composite frame.

   if (!fMapSubwindows) {
      //MapWindow();
      return;
   }

   TGWindow::MapSubwindows();

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if (el->fFrame) {
         el->fFrame->MapSubwindows();
         TGFrameElement *fe = el->fFrame->GetFrameElement();
         if (fe) fe->fState |= kIsVisible;
      }
   }
}

//______________________________________________________________________________
void TGCompositeFrame::HideFrame(TGFrame *f)
{
   // Hide sub frame.

   TGFrameElement *el = FindFrameElement(f);

   if (el) {
      el->fState = 0;
      el->fFrame->UnmapWindow();
      Layout();
   }
}

//______________________________________________________________________________
void TGCompositeFrame::ShowFrame(TGFrame *f)
{
   // Show sub frame.

   TGFrameElement *el = FindFrameElement(f);

   if (el) {
      el->fState = 1;
      el->fFrame->MapWindow();
      Layout();
   }
}

//______________________________________________________________________________
Int_t TGCompositeFrame::GetState(TGFrame *f) const
{
   // Get state of sub frame.

   TGFrameElement *el = FindFrameElement(f);

   if (el)
      return el->fState;
   else
      return 0;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::IsVisible(TGFrame *f) const
{
   // Get state of sub frame.

   TGFrameElement *el = FindFrameElement(f);

   if (el)
      return (el->fState & kIsVisible);
   else
      return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::IsArranged(TGFrame *f) const
{
   // Get state of sub frame.

   TGFrameElement *el = FindFrameElement(f);

   if (el)
      return (el->fState & kIsArranged);
   else
      return kFALSE;
}

//______________________________________________________________________________
void TGCompositeFrame::Layout()
{
   // Layout the elements of the composite frame.

   if (IsLayoutBroken()) return;
   fLayoutManager->Layout();
}

//______________________________________________________________________________
void TGCompositeFrame::Print(Option_t *option) const
{
   // Print all frames in this composite frame.

   TString opt = option;
   if (opt.Contains("tree")) {
      TGWindow::Print(option);
      return;
   }

   TGFrameElement *el;
   TIter next(fList);
   TString tab = option;

   TGFrame::Print(tab.Data());
   tab += "   ";
   while ((el = (TGFrameElement*)next())) {
      el->fFrame->Print(tab.Data());
   }
}

//______________________________________________________________________________
void TGCompositeFrame::ChangeSubframesBackground(Pixel_t back)
{
   // Change background color for this frame and all subframes.

   TGFrame::ChangeBackground(back);
   TGFrameElement *el;

   TIter next(fList);

   while ((el = (TGFrameElement*)next())) {
      el->fFrame->SetBackgroundColor(back);
      if (el->fFrame->InheritsFrom(TGCompositeFrame::Class())) {
         ((TGCompositeFrame*)el->fFrame)->ChangeSubframesBackground(back);
      }
      fClient->NeedRedraw(el->fFrame);
   }
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
TGFrame *TGCompositeFrame::GetFrameFromPoint(Int_t x, Int_t y)
{
   // Get frame located at specified point.

   if (!Contains(x, y)) return 0;

   if (!fList) return this;

   TGFrame *f;
   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      //if (el->fFrame->IsVisible()) { //for this need to move IsVisible to TGFrame
      if (el->fState & kIsVisible) {
         f = el->fFrame->GetFrameFromPoint(x - el->fFrame->GetX(),
                                           y - el->fFrame->GetY());
         if (f) return f;
      }
   }
   return this;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::TranslateCoordinates(TGFrame *child, Int_t x, Int_t y,
                                              Int_t &fx, Int_t &fy)
{
   // Translate coordinates to child frame.

   if (child == this) {
      fx = x;
      fy = y;
      return kTRUE;
   }

   if (!Contains(x, y)) return kFALSE;

   if (!fList) return kFALSE;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if (el->fFrame == child) {
         fx = x - el->fFrame->GetX();
         fy = y - el->fFrame->GetY();
         return kTRUE;
      } else if (el->fFrame->IsComposite()) {
         if (((TGCompositeFrame *)el->fFrame)->TranslateCoordinates(child,
              x - el->fFrame->GetX(), y - el->fFrame->GetY(), fx, fy))
            return kTRUE;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::HandleDragEnter(TGFrame *)
{
   // Handle drag enter event.

   if (fClient && fClient->IsEditable() &&
       (fId != fClient->GetRoot()->GetId())) {

      // the dragged frame cannot be droppped
      if (fEditDisabled & (kEditDisable | kEditDisableLayout)) return kFALSE;

      //
      if (IsEditable()) {
         return kTRUE;
      }

      Float_t r, g, b;
      TColor::Pixel2RGB(fBackground, r, g, b);
      r *= 1.12;
      g *= 1.13;
      b *= 1.12;
      Pixel_t back = TColor::RGB2Pixel(r, g, b);
      gVirtualX->SetWindowBackground(fId, back);
      DoRedraw();
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::HandleDragLeave(TGFrame *)
{
   // Handle drag leave event.

   if (fClient && fClient->IsEditable() &&
       (fId != fClient->GetRoot()->GetId())) {

      if (fEditDisabled & (kEditDisable | kEditDisableLayout)) return kFALSE;

      gVirtualX->SetWindowBackground(fId, fBackground);
      DoRedraw();
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::HandleDragMotion(TGFrame *)
{
   // Handle drag motion event.

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::HandleDragDrop(TGFrame *frame, Int_t x, Int_t y,
                                        TGLayoutHints *lo)
{
   // Handle drop event.

   if (fClient && fClient->IsEditable() && frame && (x >= 0) && (y >= 0) &&
       (x + frame->GetWidth() <= fWidth) && (y + frame->GetHeight() <= fHeight)) {

      if (fEditDisabled & (kEditDisable | kEditDisableLayout)) return kFALSE;

      frame->ReparentWindow(this, x, y);
      AddFrame(frame, lo);
      frame->MapWindow();
      SetEditable(kTRUE);
      return kTRUE;
   }

   return kFALSE;
}


//______________________________________________________________________________
TGMainFrame::TGMainFrame(const TGWindow *p, UInt_t w, UInt_t h,
        UInt_t options) : TGCompositeFrame(p, w, h, options | kMainFrame)
{
   // Create a top level main frame. A main frame interacts
   // with the window manager.

   // WMDeleteNotify causes the system to send a kClientMessage to the
   // window with fFormat=32 and fUser[0]=gWM_DELETE_WINDOW when window
   // closed via WM

   gVirtualX->WMDeleteNotify(fId);

   fBindList = new TList;

   fMWMValue    = 0;
   fMWMFuncs    = 0;
   fMWMInput    = 0;
   fWMX         = -1;
   fWMY         = -1;
   fWMWidth     = (UInt_t) -1;
   fWMHeight    = (UInt_t) -1;
   fWMMinWidth  = (UInt_t) -1;
   fWMMinHeight = (UInt_t) -1;
   fWMMaxWidth  = (UInt_t) -1;
   fWMMaxHeight = (UInt_t) -1;
   fWMWidthInc  = (UInt_t) -1;
   fWMHeightInc = (UInt_t) -1;
   fWMInitState = (EInitialState) 0;

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_s),
                      kKeyControlMask, kTRUE);
   if (p == fClient->GetDefaultRoot()) {
      fMWMValue    = kMWMDecorAll;
      fMWMFuncs    = kMWMFuncAll;
      fMWMInput    = kMWMInputModeless;
      gVirtualX->SetMWMHints(fId, fMWMValue, fMWMFuncs, fMWMInput);
   }
   // if parent is editing/embedable add this frame to the parent
   if (fClient->IsEditable() && (p == fClient->GetRoot())) {
      TGCompositeFrame *frame;
      if (p && p->InheritsFrom(TGCompositeFrame::Class())) {
         frame = (TGCompositeFrame*)p;
         frame->AddFrame(this, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

         // used during paste operation
         if (gDragManager && gDragManager->IsPasting()) {
            gDragManager->SetPasteFrame(this);
         }
      }
   }
   //AddInput(kButtonPressMask); // to allow Drag and Drop
   // Create Drag&Drop Manager and define a few DND types
   fDNDTypeList = new Atom_t[3];
   fDNDTypeList[0] = gVirtualX->InternAtom("application/root", kFALSE);
   fDNDTypeList[1] = gVirtualX->InternAtom("text/uri-list", kFALSE);
   fDNDTypeList[2] = 0;
   if (!gDNDManager)
      new TGDNDManager(this, fDNDTypeList);
   SetWindowName();
}

//______________________________________________________________________________
TGMainFrame::~TGMainFrame()
{
   // TGMainFrame destructor.

   delete [] fDNDTypeList;
   if (fBindList) {
      fBindList->Delete();
      delete fBindList;
   }
}

//______________________________________________________________________________
Bool_t TGMainFrame::SaveFrameAsCodeOrImage()
{
   // Opens dialog window allowing user to save the frame contents
   // as a ROOT macro or as an image.
   // Returns kTRUE if something was saved.
   //
   // This is bound to Ctrl-S by default.

   static TString dir(".");
   static Bool_t overwr = kFALSE;

   Bool_t repeat_save;
   do {
      repeat_save = kFALSE;

      TGFileInfo fi;
      TGMainFrame *main = (TGMainFrame*)GetMainFrame();
      fi.fFileTypes = gSaveMacroTypes;
      fi.fIniDir    = StrDup(dir);
      fi.fOverwrite = overwr;
      new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);
      if (!fi.fFilename) return kFALSE;
      dir = fi.fIniDir;
      overwr = fi.fOverwrite;
      TString fname = gSystem->UnixPathName(fi.fFilename);
      if (fname.EndsWith(".C"))
         main->SaveSource(fname.Data(), "");
      else {
         TImage::EImageFileTypes gtype = TImage::kUnknown;
         if (fname.EndsWith("gif")) {
            gtype = TImage::kGif;
         } else if (fname.EndsWith(".png")) {
            gtype = TImage::kPng;
         } else if (fname.EndsWith(".jpg")) {
            gtype = TImage::kJpeg;
         } else if (fname.EndsWith(".tiff")) {
            gtype = TImage::kTiff;
         } else if (fname.EndsWith(".xpm")) {
            gtype = TImage::kXpm;
         }
         if (gtype != TImage::kUnknown) {
            Int_t saver = gErrorIgnoreLevel;
            gErrorIgnoreLevel = kFatal;
            TImage *img = TImage::Create();
            RaiseWindow();
            img->FromWindow(GetId());
            img->WriteImage(fname, gtype);
            gErrorIgnoreLevel = saver;
            delete img;
         }
         else {
            Int_t retval;
            new TGMsgBox(fClient->GetDefaultRoot(), this, "Error...",
                         TString::Format("file (%s) cannot be saved with this extension",
                                         fname.Data()), kMBIconExclamation,
                         kMBRetry | kMBCancel, &retval);
            repeat_save = (retval == kMBRetry);
         }
      }
   } while (repeat_save);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGMainFrame::HandleKey(Event_t *event)
{
   // Handle keyboard events.

   if (fBindList) {

      TIter next(fBindList);
      TGMapKey *m;
      TGFrame  *w = 0;

      while ((m = (TGMapKey *) next())) {
         if (m->fKeyCode == event->fCode) {
            w = (TGFrame *) m->fWindow;
            if (w->HandleKey(event)) return kTRUE;
         }
      }
   }

   if ((event->fType == kGKeyPress) && (event->fState & kKeyControlMask)) {
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);

      if ((keysym & ~0x20) == kKey_S) { // case insensitive ctrl-s
         return SaveFrameAsCodeOrImage();
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGMainFrame::BindKey(const TGWindow *w, Int_t keycode, Int_t modifier) const
{
   // Bind key to a window.

   TList *list = fBindList;
   Handle_t id = fId;

   if (fClient->IsEditable()) {
      TGMainFrame *main = (TGMainFrame*)GetMainFrame();
      list = main->GetBindList();
      id = main->GetId();
   }

   if (list) {
      TGMapKey *m = new TGMapKey(keycode, (TGWindow *)w);
      list->Add(m);
      gVirtualX->GrabKey(id, keycode, modifier, kTRUE);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGMainFrame::RemoveBind(const TGWindow *, Int_t keycode, Int_t modifier) const
{
   // Remove key binding.

   if (fBindList) {
      TIter next(fBindList);
      TGMapKey *m;
      while ((m = (TGMapKey *) next())) {
         if (m->fKeyCode == (UInt_t) keycode) {
            fBindList->Remove(m);
            delete m;
            gVirtualX->GrabKey(fId, keycode, modifier, kFALSE);
            return;
         }
      }
   }
}

//______________________________________________________________________________
Bool_t TGMainFrame::HandleButton(Event_t *event)
{
   // Handle mouse button events.

   if (event->fType == kButtonRelease) {
      if (gDNDManager->IsDragging()) gDNDManager->Drop();
   }
   return TGCompositeFrame::HandleButton(event);
}


//______________________________________________________________________________
Bool_t TGMainFrame::HandleMotion(Event_t *event)
{
   // Handle mouse motion events.

   if (gDNDManager && gDNDManager->IsDragging()) {
      gDNDManager->Drag(event->fXRoot, event->fYRoot,
                        TGDNDManager::GetDNDActionCopy(), event->fTime);
   }
   return TGCompositeFrame::HandleMotion(event);
}

//______________________________________________________________________________
Bool_t TGMainFrame::HandleSelection(Event_t *event)
{
   // Handle primary selection event.

   if ((Atom_t)event->fUser[1] == TGDNDManager::GetDNDSelection()) {
      if (gDNDManager)
         return gDNDManager->HandleSelection(event);
   }
   return TGCompositeFrame::HandleSelection(event);
}

//______________________________________________________________________________
Bool_t TGMainFrame::HandleSelectionRequest(Event_t *event)
{
   // Handle selection request event.

   if ((Atom_t)event->fUser[1] == TGDNDManager::GetDNDSelection()) {
      if (gDNDManager)
         return gDNDManager->HandleSelectionRequest(event);
   }
   return TGCompositeFrame::HandleSelectionRequest(event);
}

//______________________________________________________________________________
Bool_t TGMainFrame::HandleClientMessage(Event_t *event)
{
   // Handle client messages sent to this frame.

   TGCompositeFrame::HandleClientMessage(event);

   if ((event->fFormat == 32) && ((Atom_t)event->fUser[0] == gWM_DELETE_WINDOW) &&
       (event->fHandle != gROOT_MESSAGE)) {
      Emit("CloseWindow()");
      if (TestBit(kNotDeleted) && !TestBit(kDontCallClose))
         CloseWindow();
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGMainFrame::SendCloseMessage()
{
   // Send close message to self. This method should be called from
   // a button to close this window.

   Event_t event;

   event.fType   = kClientMessage;
   event.fFormat = 32;
   event.fHandle = gWM_DELETE_WINDOW;

   event.fWindow  = GetId();
   event.fUser[0] = (Long_t) gWM_DELETE_WINDOW;
   event.fUser[1] = 0;
   event.fUser[2] = 0;
   event.fUser[3] = 0;
   event.fUser[4] = 0;

   gVirtualX->SendEvent(GetId(), &event);
}

//______________________________________________________________________________
void TGMainFrame::CloseWindow()
{
   // Close and delete main frame. We get here in response to ALT+F4 or
   // a window manager close command. To terminate the application when this
   // happens override this method and call gApplication->Terminate(0) or
   // make a connection to this signal (if after the slot this method
   // should not be called call DontCallClose() in the slot).
   // By default the window will be deleted.

   DeleteWindow();
}

//______________________________________________________________________________
void TGMainFrame::DontCallClose()
{
   // Typically call this method in the slot connected to the CloseWindow()
   // signal to prevent the calling of the default or any derived CloseWindow()
   // methods to prevent premature or double deletion of this window.

   SetBit(kDontCallClose);
}

//______________________________________________________________________________
void TGMainFrame::SetWindowName(const char *name)
{
   // Set window name. This is typically done via the window manager.

   if (!name) {
      TGWindow::SetWindowName();
   } else {
      fWindowName = name;
      gVirtualX->SetWindowName(fId, (char *)name);
   }
}

//______________________________________________________________________________
void TGMainFrame::SetIconName(const char *name)
{
   // Set window icon name. This is typically done via the window manager.

   fIconName = name;
   gVirtualX->SetIconName(fId, (char *)name);
}

//______________________________________________________________________________
const TGPicture *TGMainFrame::SetIconPixmap(const char *iconName)
{
   // Set window icon pixmap by name. This is typically done via the window
   // manager. Icon can be in any image format supported by TImage, e.g.
   // GIF, XPM, PNG, JPG .. or even PS, PDF (see EImageFileTypes in TImage.h
   // for the full list of supported formats).
   //
   // For example,
   //    main_frame->SetIconPixmap("/home/root/icons/bld_rgb.png");

   fIconPixmap = iconName;
   const TGPicture *iconPic = fClient->GetPicture(iconName);
   if (iconPic) {
      Pixmap_t pic = iconPic->GetPicture();
      gVirtualX->SetIconPixmap(fId, pic);
      return iconPic;
   } else
      return 0;
}

//______________________________________________________________________________
void TGMainFrame::SetIconPixmap(char **xpm_array)
{
   // Set window icon by xpm array. That allows to have icons
   // builtin to the source code.
   //
   // For example,
   //    #include "/home/root/icons/bld_rgb.xpm"
   //    //bld_rgb.xpm contains char *bld_rgb[] array
   //    main_frame->SetIconPixmap(bld_rgb);

   TImage *img = TImage::Create();
   img->SetImageBuffer(xpm_array, TImage::kXpm);
   Pixmap_t pic = img->GetPixmap();
   if (pic) {
      gVirtualX->SetIconPixmap(fId, pic);
   } else {
      Warning("SetIconPixmap", "Failed to set window icon from xpm array.");
   }
   delete img;
}

//______________________________________________________________________________
void TGMainFrame::SetClassHints(const char *className, const char *resourceName)
{
   // Set the windows class and resource name. Used to get the right
   // resources from the resource database. However, ROOT applications
   // will typically use the .rootrc file for this.

   fClassName    = className;
   fResourceName = resourceName;
   gVirtualX->SetClassHints(fId, (char *)className, (char *)resourceName);
}

//______________________________________________________________________________
void TGMainFrame::SetMWMHints(UInt_t value, UInt_t funcs, UInt_t input)
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).

   if (fClient->IsEditable() && (fParent == fClient->GetRoot())) return;

   fMWMValue = value;
   fMWMFuncs = funcs;
   fMWMInput = input;
   gVirtualX->SetMWMHints(fId, value, funcs, input);
}

//______________________________________________________________________________
void TGMainFrame::SetWMPosition(Int_t x, Int_t y)
{
   // Give the window manager a window position hint.

   if (fClient->IsEditable() && (fParent == fClient->GetRoot())) return;

   fWMX = x;
   fWMY = y;
   gVirtualX->SetWMPosition(fId, x, y);
}

//______________________________________________________________________________
void TGMainFrame::SetWMSize(UInt_t w, UInt_t h)
{
   // Give the window manager a window size hint.

   if (fClient->IsEditable() && (fParent == fClient->GetRoot())) return;

   fWMWidth  = w;
   fWMHeight = h;
   gVirtualX->SetWMSize(fId, w, h);
}

//______________________________________________________________________________
void TGMainFrame::SetWMSizeHints(UInt_t wmin, UInt_t hmin,
                                 UInt_t wmax, UInt_t hmax,
                                 UInt_t winc, UInt_t hinc)
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.

   if (fClient->IsEditable() && (fParent == fClient->GetRoot())) return;

   fMinWidth    = fWMMinWidth  = wmin;
   fMinHeight   = fWMMinHeight = hmin;
   fMaxWidth    = fWMMaxWidth  = wmax;
   fMaxHeight   = fWMMaxHeight = hmax;
   fWMWidthInc  = winc;
   fWMHeightInc = hinc;
   gVirtualX->SetWMSizeHints(fId, wmin, hmin, wmax, hmax, winc, hinc);
}

//______________________________________________________________________________
void TGMainFrame::SetWMState(EInitialState state)
{
   // Set the initial state of the window. Either kNormalState or kIconicState.

   if (fClient->IsEditable() && (fParent == fClient->GetRoot())) return;

   fWMInitState = state;
   gVirtualX->SetWMState(fId, state);
}


//______________________________________________________________________________
TGTransientFrame::TGTransientFrame(const TGWindow *p, const TGWindow *main,
                                   UInt_t w, UInt_t h, UInt_t options)
   : TGMainFrame(p, w, h, options | kTransientFrame)
{
   // Create a transient window. A transient window is typically used for
   // dialog boxes.

   fMain = main;
   if (!fMain && gClient)
      fMain = gClient->GetRoot();

   if (fMain) {
      gVirtualX->SetWMTransientHint(fId, fMain->GetId());
   }
}

//______________________________________________________________________________
void TGTransientFrame::CenterOnParent(Bool_t croot, EPlacement pos)
{
   // Position transient frame centered relative to the parent frame.
   // If fMain is 0 (i.e. TGTransientFrame is acting just like a
   // TGMainFrame) and croot is true, the window will be centered on
   // the root window, otherwise no action is taken and the default
   // wm placement will be used.

   Int_t x=0, y=0, ax, ay;
   Window_t wdummy;

   UInt_t dw = fClient->GetDisplayWidth();
   UInt_t dh = fClient->GetDisplayHeight();

   if (fMain) {

      switch (pos) {
         case kCenter:
            x = (Int_t)(((TGFrame *) fMain)->GetWidth() - fWidth) >> 1;
            y = (Int_t)(((TGFrame *) fMain)->GetHeight() - fHeight) >> 1;
            break;
         case kRight:
            x = (Int_t)(((TGFrame *) fMain)->GetWidth() - (fWidth >> 1));
            y = (Int_t)(((TGFrame *) fMain)->GetHeight() - fHeight) >> 1;
            break;
         case kLeft:
            x = (Int_t)(-1 * (Int_t)(fWidth >> 1));
            y = (Int_t)(((TGFrame *) fMain)->GetHeight() - fHeight) >> 1;
            break;
         case kTop:
            x = (Int_t)(((TGFrame *) fMain)->GetWidth() - fWidth) >> 1;
            y = (Int_t)(-1 * (Int_t)(fHeight >> 1));
            break;
         case kBottom:
            x = (Int_t)(((TGFrame *) fMain)->GetWidth() - fWidth) >> 1;
            y = (Int_t)(((TGFrame *) fMain)->GetHeight() - (fHeight >> 1));
            break;
         case kTopLeft:
            x = (Int_t)(-1 * (Int_t)(fWidth >> 1));
            y = (Int_t)(-1 * (Int_t)(fHeight >> 1));
            break;
         case kTopRight:
            x = (Int_t)(((TGFrame *) fMain)->GetWidth() - (fWidth >> 1));
            y = (Int_t)(-1 * (Int_t)(fHeight >> 1));
            break;
         case kBottomLeft:
            x = (Int_t)(-1 * (Int_t)(fWidth >> 1));
            y = (Int_t)(((TGFrame *) fMain)->GetHeight() - (fHeight >> 1));
            break;
         case kBottomRight:
            x = (Int_t)(((TGFrame *) fMain)->GetWidth() - (fWidth >> 1));
            y = (Int_t)(((TGFrame *) fMain)->GetHeight() - (fHeight >> 1));
            break;
      }

      gVirtualX->TranslateCoordinates(fMain->GetId(), GetParent()->GetId(),
                                      x, y, ax, ay, wdummy);
      if (ax < 10)
         ax = 10;
      else if (ax + fWidth + 10 > dw)
         ax = dw - fWidth - 10;

      if (ay < 20)
         ay = 20;
      else if (ay + fHeight + 50 > dh)
         ay = dh - fHeight - 50;

   } else if (croot) {

      switch (pos) {
         case kCenter:
            x = (dw - fWidth) >> 1;
            y = (dh - fHeight) >> 1;
            break;
         case kRight:
            x = dw - (fWidth >> 1);
            y = (dh - fHeight) >> 1;
            break;
         case kLeft:
            x = -1 * (Int_t)(fWidth >> 1);
            y = (dh - fHeight) >> 1;
            break;
         case kTop:
            x = (dw - fWidth) >> 1;
            y = -1 * (Int_t)(fHeight >> 1);
            break;
         case kBottom:
            x = (dw - fWidth) >> 1;
            y = dh - (fHeight >> 1);
            break;
         case kTopLeft:
            x = -1 * (Int_t)(fWidth >> 1);
            y = -1 * (Int_t)(fHeight >> 1);
            break;
         case kTopRight:
            x = dw - (fWidth >> 1);
            y = -1 * (Int_t)(fHeight >> 1);
            break;
         case kBottomLeft:
            x = -1 * (Int_t)(fWidth >> 1);
            y = dh - (fHeight >> 1);
            break;
         case kBottomRight:
            x = dw - (fWidth >> 1);
            y = dh - (fHeight >> 1);
            break;
      }

      ax = x;
      ay = y;

   } else {

      return;

   }

   Move(ax, ay);
   SetWMPosition(ax, ay);
}

//______________________________________________________________________________
TGGroupFrame::TGGroupFrame(const TGWindow *p, TGString *title,
                           UInt_t options, GContext_t norm,
                           FontStruct_t font, Pixel_t back) :
   TGCompositeFrame(p, 1, 1, options, back)
{
   // Create a group frame. The title will be adopted and deleted by the
   // group frame.

   fText       = title;
   fFontStruct = font;
   fNormGC     = norm;
   fTitlePos   = kLeft;
   fHasOwnFont = kFALSE;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fBorderWidth = max_ascent + max_descent + 1;
}

//______________________________________________________________________________
TGGroupFrame::TGGroupFrame(const TGWindow *p, const char *title,
                           UInt_t options, GContext_t norm,
                           FontStruct_t font, Pixel_t back) :
   TGCompositeFrame(p, 1, 1, options, back)
{
   // Create a group frame.

   fText       = new TGString(!p && !title ? GetName() : title);
   fFontStruct = font;
   fNormGC     = norm;
   fTitlePos   = kLeft;
   fHasOwnFont = kFALSE;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fBorderWidth = max_ascent + max_descent + 1;

   SetWindowName();
}

//______________________________________________________________________________
TGGroupFrame::~TGGroupFrame()
{
   // Delete a group frame.

   if (fHasOwnFont) {
      TGGCPool *pool = fClient->GetGCPool();
      TGGC *gc = pool->FindGC(fNormGC);
      pool->FreeGC(gc);
   }
   delete fText;
}

//______________________________________________________________________________
TGDimension TGGroupFrame::GetDefaultSize() const
{
   // Returns default size.

   UInt_t tw = gVirtualX->TextWidth(fFontStruct, fText->GetString(),
                                    fText->GetLength()) + 24;

   TGDimension dim = TGCompositeFrame::GetDefaultSize();

   return  tw>dim.fWidth ? TGDimension(tw, dim.fHeight) : dim;
}

//______________________________________________________________________________
void TGGroupFrame::DoRedraw()
{
   // Redraw the group frame. Need special DoRedraw() since we need to
   // redraw with fBorderWidth=0.

   gVirtualX->ClearArea(fId, 0, 0, fWidth, fHeight);

   DrawBorder();
}


//______________________________________________________________________________
void TGGroupFrame::SetTextColor(Pixel_t color, Bool_t local)
{
   // Changes text color.
   // If local is true color is changed locally, otherwise - globally.

   TGGCPool *pool =  fClient->GetResourcePool()->GetGCPool();
   TGGC *gc = pool->FindGC(fNormGC);

   if (local) {
      gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy
      fHasOwnFont = kTRUE;
   }

   gc->SetForeground(color);
   fNormGC = gc->GetGC();
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGGroupFrame::SetTextFont(FontStruct_t font, Bool_t local)
{
   // Changes text font.
   // If local is true font is changed locally - otherwise globally.

   FontH_t v = gVirtualX->GetFontHandle(font);
   if (!v) return;

   fFontStruct = font;

   TGGCPool *pool =  fClient->GetResourcePool()->GetGCPool();
   TGGC *gc = pool->FindGC(fNormGC);

   if (local) {
      gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy
      fHasOwnFont = kTRUE;
   }

   gc->SetFont(v);
   fNormGC = gc->GetGC();

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGGroupFrame::SetTextFont(const char *fontName, Bool_t local)
{
   // Changes text font specified by name.
   // If local is true font is changed locally - otherwise globally.

   TGFont *font = fClient->GetFont(fontName);

   if (font) {
      SetTextFont(font->GetFontStruct(), local);
   }
}

//______________________________________________________________________________
Bool_t TGGroupFrame::HasOwnFont() const
{
   // Returns kTRUE if text attributes are unique,
   // returns kFALSE if text attributes are shared (global).

   return fHasOwnFont;
}

//______________________________________________________________________________
void TGGroupFrame::DrawBorder()
{
   // Draw border of around the group frame.
   //
   // if frame is kRaisedFrame  - a frame border is of "wall style",
   // otherwise of "groove style".

   Int_t x, y, l, t, r, b, gl, gr, sep, max_ascent, max_descent;

   UInt_t tw = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   l = 0;
   t = (max_ascent + max_descent + 2) >> 1;
   r = fWidth - 1;
   // next three lines are for backward compatibility in case of horizontal layout
   TGLayoutManager * lm = GetLayoutManager();
   if ((lm->InheritsFrom(TGHorizontalLayout::Class())) ||
       (lm->InheritsFrom(TGMatrixLayout::Class())))
      b = fHeight - 1;
   else
      b = fHeight - t;

   sep = 3;
   UInt_t rr = 5 + (sep << 1) + tw;

   switch (fTitlePos) {
      case kRight:
         gl = fWidth>rr ? Int_t(fWidth - rr) : 5 + sep;
         break;
      case kCenter:
         gl = fWidth>tw ? Int_t((fWidth - tw)>>1) - sep : 5 + sep;
         break;
      case kLeft:
      default:
         gl = 5 + sep;
   }
   gr = gl + tw + (sep << 1);

   switch (fOptions & (kSunkenFrame | kRaisedFrame)) {
      case kRaisedFrame:
         gVirtualX->DrawLine(fId, GetHilightGC()(),  l,   t,   gl,  t);
         gVirtualX->DrawLine(fId, GetShadowGC()(), l+1, t+1, gl,  t+1);

         gVirtualX->DrawLine(fId, GetHilightGC()(),  gr,  t,   r-1, t);
         gVirtualX->DrawLine(fId, GetShadowGC()(), gr,  t+1, r-2, t+1);

         gVirtualX->DrawLine(fId, GetHilightGC()(),  r-1, t,   r-1, b-1);
         gVirtualX->DrawLine(fId, GetShadowGC()(), r,   t,   r,   b);

         gVirtualX->DrawLine(fId, GetHilightGC()(),  r-1, b-1, l,   b-1);
         gVirtualX->DrawLine(fId, GetShadowGC()(), r,   b,   l,   b);

         gVirtualX->DrawLine(fId, GetHilightGC()(),  l,   b-1, l,   t);
         gVirtualX->DrawLine(fId, GetShadowGC()(), l+1, b-2, l+1, t+1);
         break;
      case kSunkenFrame:
      default:
         gVirtualX->DrawLine(fId, GetShadowGC()(),  l,   t,   gl,  t);
         gVirtualX->DrawLine(fId, GetHilightGC()(), l+1, t+1, gl,  t+1);

         gVirtualX->DrawLine(fId, GetShadowGC()(),  gr,  t,   r-1, t);
         gVirtualX->DrawLine(fId, GetHilightGC()(), gr,  t+1, r-2, t+1);

         gVirtualX->DrawLine(fId, GetShadowGC()(),  r-1, t,   r-1, b-1);
         gVirtualX->DrawLine(fId, GetHilightGC()(), r,   t,   r,   b);

         gVirtualX->DrawLine(fId, GetShadowGC()(),  r-1, b-1, l,   b-1);
         gVirtualX->DrawLine(fId, GetHilightGC()(), r,   b,   l,   b);

         gVirtualX->DrawLine(fId, GetShadowGC()(),  l,   b-1, l,   t);
         gVirtualX->DrawLine(fId, GetHilightGC()(), l+1, b-2, l+1, t+1);
         break;
   }

   x = gl + sep;
   y = 1;

   fText->Draw(fId, fNormGC, x, y + max_ascent);
}

//______________________________________________________________________________
void TGGroupFrame::SetTitle(TGString *title)
{
   // Set or change title of the group frame. Titlte TGString is adopted
   // by the TGGroupFrame.

   if (!title) {
      Warning("SetTitle", "title cannot be 0, try \"\"");
      title = new TGString("");
   }

   delete fText;

   fText = title;
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGGroupFrame::SetTitle(const char *title)
{
   // Set or change title of the group frame.

   if (!title) {
      Error("SetTitle", "title cannot be 0, try \"\"");
      return;
   }

   SetTitle(new TGString(title));
}

//______________________________________________________________________________
FontStruct_t TGGroupFrame::GetDefaultFontStruct()
{
   // Return default font structure in use.

   if (!fgDefaultFont && gClient)
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGGroupFrame::GetDefaultGC()
{
   // Return default graphics context in use.

   if (!fgDefaultGC && gClient)
      fgDefaultGC = gClient->GetResourcePool()->GetFrameGC();
   return *fgDefaultGC;
}

//______________________________________________________________________________
TGHeaderFrame::TGHeaderFrame(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options, Pixel_t back) :
  TGHorizontalFrame(p, w, h, options | kVerticalFrame, back)
{
   // Header Frame constructor.

   fSplitCursor = kNone;
   fSplitCursor = gVirtualX->CreateCursor(kArrowHor);
   fOverSplitter = false;
   fOverButton = -1;
   fLastButton = -1;
   fNColumns   = 1;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone);
   AddInput(kPointerMotionMask);
}

//______________________________________________________________________________
void TGHeaderFrame::SetColumnsInfo(Int_t nColumns, TGTextButton  **colHeader,
               TGVFileSplitter  **splitHeader)
{
   // Set columns informations in the header frame.

   fNColumns = nColumns;
   fColHeader = colHeader;
   fSplitHeader = splitHeader;
}

//______________________________________________________________________________
Bool_t TGHeaderFrame::HandleButton(Event_t* event)
{
   // Handle mouse button event in header frame.

   if ( event->fY > 0 &&
        event->fY <= (Int_t) this->GetHeight() ) {
      for (Int_t i = 1; i < fNColumns; ++i ) {
         if ( event->fX < fColHeader[i]->GetX() &&
            event->fX >= fColHeader[i-1]->GetX() ) {
            if ( fOverSplitter ) {
               if ( event->fX <= fColHeader[i-1]->GetX() + 5 )
                  fSplitHeader[i-2]->HandleButton(event);
               else
                  fSplitHeader[i-1]->HandleButton(event);
            } else {
               if ( event->fType == kButtonPress ) {
                  fLastButton = i - 1;
               } else {
                  fLastButton = -1;
               }
               event->fX -= fColHeader[i-1]->GetX();
               fColHeader[i-1]->HandleButton(event);
            }
            break;
         }
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHeaderFrame::HandleDoubleClick(Event_t *event)
{
   // Handle double click mouse event in header frame.

   if ( event->fY > 0 &&
        event->fY <= (Int_t) this->GetHeight() ) {
      for (Int_t i = 1; i < fNColumns; ++i ) {
         if ( event->fX < fColHeader[i]->GetX() &&
            event->fX >= fColHeader[i-1]->GetX() ) {
            if ( fOverSplitter ) {
               if ( event->fX <= fColHeader[i-1]->GetX() + 5 )
                  fSplitHeader[i-2]->HandleDoubleClick(event);
               else
                  fSplitHeader[i-1]->HandleDoubleClick(event);
            } else {
               event->fX -= fColHeader[i-1]->GetX();
               fColHeader[i-1]->HandleDoubleClick(event);
            }
            break;
         }
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHeaderFrame::HandleMotion(Event_t* event)
{
   // Handle mouse motion events in header frame.

   if ( event->fY > 0 &&
        event->fY <= (Int_t) this->GetHeight() ) {
      Bool_t inMiddle = false;

      for (Int_t i = 1; i < fNColumns; ++i ) {
         if ( event->fX > fColHeader[i]->GetX() - 5 &&
            event->fX < fColHeader[i]->GetX() + 5 ) {
            inMiddle = true;
         }
         if ( event->fX < fColHeader[i]->GetX() &&
            event->fX >= fColHeader[i-1]->GetX() ) {
            fOverButton = i - 1;
         }
      }
      fOverSplitter = inMiddle;
      if ( fOverSplitter ) {
         gVirtualX->SetCursor(fId, fSplitCursor);
      }
      else {
         gVirtualX->SetCursor(fId, kNone);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGFrame::SaveUserColor(ostream &out, Option_t *option)
{
   // Save a user color in a C++ macro file - used in SavePrimitive().

   char quote = '"';

   if (gROOT->ClassSaved(TGFrame::Class())) {
      out << endl;
   } else {
      //  declare a color variable to reflect required user changes
      out << endl;
      out << "   ULong_t ucolor;        // will reflect user color changes" << endl;
   }
   ULong_t ucolor;
   if (option && !strcmp(option, "slider"))
      ucolor = GetDefaultFrameBackground();
   else
      ucolor = GetBackground();
   if ((ucolor != fgUserColor) || (ucolor == GetWhitePixel())) {
      const char *ucolorname = TColor::PixelAsHexString(ucolor);
      out << "   gClient->GetColorByName(" << quote << ucolorname << quote
          << ",ucolor);" << endl;
      fgUserColor = ucolor;
   }
}

//______________________________________________________________________________
TString TGFrame::GetOptionString() const
{
   // Returns a frame option string - used in SavePrimitive().

   TString options;

   if (!GetOptions()) {
      options = "kChildFrame";
   } else {
      if (fOptions & kMainFrame) {
         if (options.Length() == 0) options  = "kMainFrame";
         else                       options += " | kMainFrame";
      }
      if (fOptions & kVerticalFrame) {
         if (options.Length() == 0) options  = "kVerticalFrame";
         else                       options += " | kVerticalFrame";
      }
      if (fOptions & kHorizontalFrame) {
         if (options.Length() == 0) options  = "kHorizontalFrame";
         else                       options += " | kHorizontalFrame";
      }
      if (fOptions & kSunkenFrame) {
         if (options.Length() == 0) options  = "kSunkenFrame";
         else                       options += " | kSunkenFrame";
      }
      if (fOptions & kRaisedFrame) {
         if (options.Length() == 0) options  = "kRaisedFrame";
         else                       options += " | kRaisedFrame";
      }
      if (fOptions & kDoubleBorder) {
         if (options.Length() == 0) options  = "kDoubleBorder";
         else                       options += " | kDoubleBorder";
      }
      if (fOptions & kFitWidth) {
         if (options.Length() == 0) options  = "kFitWidth";
         else                       options += " | kFitWidth";
      }
      if (fOptions & kFixedWidth) {
         if (options.Length() == 0) options  = "kFixedWidth";
         else                       options += " | kFixedWidth";
      }
      if (fOptions & kFitHeight) {
         if (options.Length() == 0) options  = "kFitHeight";
         else                       options += " | kFitHeight";
      }
      if (fOptions & kFixedHeight) {
         if (options.Length() == 0) options  = "kFixedHeight";
         else                       options += " | kFixedHeight";
      }
      if (fOptions & kOwnBackground) {
         if (options.Length() == 0) options  = "kOwnBackground";
         else                       options += " | kOwnBackground";
      }
      if (fOptions & kTransientFrame) {
         if (options.Length() == 0) options  = "kTransientFrame";
         else                       options += " | kTransientFrame";
      }
      if (fOptions & kTempFrame) {
         if (options.Length() == 0) options  = "kTempFrame";
         else                       options += " | kTempFrame";
      }
   }
   return options;
}

//______________________________________________________________________________
TString TGMainFrame::GetMWMvalueString() const
{
   // Returns MWM decoration hints as a string - used in SavePrimitive().

   TString hints;

   if (fMWMValue) {
      if (fMWMValue & kMWMDecorAll) {
         if (hints.Length() == 0) hints  = "kMWMDecorAll";
         else                     hints += " | kMWMDecorAll";
      }
      if (fMWMValue & kMWMDecorBorder) {
         if (hints.Length() == 0) hints  = "kMWMDecorBorder";
         else                     hints += " | kMWMDecorBorder";
      }
      if (fMWMValue & kMWMDecorResizeH) {
         if (hints.Length() == 0) hints  = "kMWMDecorResizeH";
         else                     hints += " | kMWMDecorResizeH";
      }
      if (fMWMValue & kMWMDecorTitle) {
         if (hints.Length() == 0) hints  = "kMWMDecorTitle";
         else                     hints += " | kMWMDecorTitle";
      }
      if (fMWMValue & kMWMDecorMenu) {
         if (hints.Length() == 0) hints  = "kMWMDecorMenu";
         else                     hints += " | kMWMDecorMenu";
      }
      if (fMWMValue & kMWMDecorMinimize) {
         if (hints.Length() == 0) hints  = "kMWMDecorMinimize";
         else                     hints += " | kMWMDecorMinimize";
      }
      if (fMWMValue & kMWMDecorMaximize) {
         if (hints.Length() == 0) hints  = "kMWMDecorMaximize";
         else                     hints += " | kMWMDecorMaximize";
      }
   }
   return hints;
}

//______________________________________________________________________________
TString TGMainFrame::GetMWMfuncString() const
{
   // Returns MWM function hints as a string - used in SavePrimitive().

   TString hints;

   if (fMWMFuncs) {

      if (fMWMFuncs & kMWMFuncAll) {
         if (hints.Length() == 0) hints  = "kMWMFuncAll";
         else                     hints += " | kMWMFuncAll";
      }
      if (fMWMFuncs & kMWMFuncResize) {
         if (hints.Length() == 0) hints  = "kMWMFuncResize";
         else                     hints += " | kMWMFuncResize";
      }
      if (fMWMFuncs & kMWMFuncMove) {
         if (hints.Length() == 0) hints  = "kMWMFuncMove";
         else                     hints += " | kMWMFuncMove";
      }
      if (fMWMFuncs & kMWMFuncMinimize) {
         if (hints.Length() == 0) hints  = "kMWMFuncMinimize";
         else                     hints += " | kMWMFuncMinimize";
      }
      if (fMWMFuncs & kMWMFuncMaximize) {
         if (hints.Length() == 0) hints  = "kMWMFuncMaximize";
         else                     hints += " | kMWMFuncMaximize";
      }
      if (fMWMFuncs & kMWMFuncClose) {
         if (hints.Length() == 0) hints  = "kMWMFuncClose";
         else                     hints += " | kMWMFuncClose";
      }
   }
   return hints;
}

//______________________________________________________________________________
TString TGMainFrame::GetMWMinpString() const
{
   // Returns MWM input mode hints as a string - used in SavePrimitive().

   TString hints;

   if (fMWMInput == 0) hints = "kMWMInputModeless";

   if (fMWMInput == 1) hints = "kMWMInputPrimaryApplicationModal";

   if (fMWMInput == 2) hints = "kMWMInputSystemModal";

   if (fMWMInput == 3) hints = "kMWMInputFullApplicationModal";

   return hints;
}

//______________________________________________________________________________
void TGCompositeFrame::SavePrimitiveSubframes(ostream &out, Option_t *option /*= ""*/)
{
   // Auxilary protected method  used to save subframes.

   if (fLayoutBroken)
      out << "   " << GetName() << "->SetLayoutBroken(kTRUE);" << endl;

   if (!fList) return;

   char quote = '"';

   TGFrameElement *el;
   static TGHSplitter *hsplit = 0;
   static TGVSplitter *vsplit = 0;
   TList *signalslist;
   TList *connlist;
   TQConnection *conn;
   TString signal_name, slot_name;

   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {

      // Don't save hidden (unmapped) frames having a parent different
      // than this frame. Solves a problem with shared frames
      // (e.g. shared menus in the new Browser)
      if ((!el->fState & kIsVisible) && (el->fFrame->GetParent() != this))
         continue;

      // Remember if the frame to be saved is a TG(H,V)Splitter
      // See comments below and in TG[H/V]Splitter::SavePrimitive()
      if (el->fFrame->InheritsFrom("TGVSplitter")) {
         vsplit = (TGVSplitter *)el->fFrame;
         if (vsplit->GetLeft())
            vsplit = 0;
      }
      else if (el->fFrame->InheritsFrom("TGHSplitter")) {
         hsplit = (TGHSplitter *)el->fFrame;
         if (hsplit->GetAbove())
            hsplit = 0;
      }
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");"<< endl;
      if (IsLayoutBroken()) {
         out << "   " << el->fFrame->GetName() << "->MoveResize(";
         out << el->fFrame->GetX() << "," << el->fFrame->GetY() << ",";
         out << el->fFrame->GetWidth() << ","  << el->fFrame->GetHeight();
         out << ");" << endl;
      }
      // TG(H,V)Splitter->SetFrame(theframe) can only be saved _AFTER_
      // having saved "theframe", when "theframe" is either at right
      // or below the splitter (that means after the splitter in the
      // list of frames), otherwise "theframe" would be undefined
      // (aka used before to be created)...
      if (vsplit && el->fFrame == vsplit->GetFrame()) {
         out << "   " << vsplit->GetName() << "->SetFrame(" << vsplit->GetFrame()->GetName();
         if (vsplit->GetLeft()) out << ",kTRUE);" << endl;
         else                 out << ",kFALSE);"<< endl;
         vsplit = 0;
      }
      if (hsplit && el->fFrame == hsplit->GetFrame()) {
         out << "   " << hsplit->GetName() << "->SetFrame(" << hsplit->GetFrame()->GetName();
         if (hsplit->GetAbove()) out << ",kTRUE);" << endl;
         else                  out << ",kFALSE);"<< endl;
         hsplit = 0;
      }

      if (!el->fState & kIsVisible) {
         gListOfHiddenFrames->Add(el->fFrame);
      }

      // saving signals/slots
      signalslist = (TList*)el->fFrame->GetListOfSignals();
      if (!signalslist)  continue;
      connlist = (TList*)signalslist->Last();
      if (connlist) {
         conn = (TQConnection*)connlist->Last();
         signal_name = connlist->GetName();
         slot_name = conn->GetName();
         Int_t eq = slot_name.First('=');
         Int_t rb = slot_name.First(')');
         if (eq != -1)
            slot_name.Remove(eq, rb-eq);
         out << "   " << el->fFrame->GetName() << "->Connect(" << quote << signal_name
             << quote << ", 0, 0, " << quote << slot_name << quote << ");" << endl;

         TList *lsl = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfSlots");
         if (lsl) {
            TObjString *slotel = (TObjString *)lsl->FindObject(slot_name);
            if (!slotel)
               lsl->Add(new TObjString(slot_name));
         }
      }
   }
   out << endl;
}

//______________________________________________________________________________
void TGCompositeFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a composite frame widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   if (!strcmp(GetName(),"")) {
      SetName(Form("fCompositeframe%d",fgCounter));
      fgCounter++;
   }

   out << endl << "   // composite frame" << endl;
   out << "   TGCompositeFrame *";
   out << GetName() << " = new TGCompositeFrame(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out << ");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   // setting layout manager if it differs from the composite frame type
   TGLayoutManager *lm = GetLayoutManager();
   if ((GetOptions() & kHorizontalFrame) &&
       (lm->InheritsFrom(TGHorizontalLayout::Class()))) {
      ;
   } else if ((GetOptions() & kVerticalFrame) &&
              (lm->InheritsFrom(TGVerticalLayout::Class()))) {
      ;
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      lm->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   SavePrimitiveSubframes(out, option);
}

//______________________________________________________________________________
void TGMainFrame::SaveSource(const char *filename, Option_t *option)
{
   // Save the GUI main frame widget in a C++ macro file.

   // iteration over all active classes to exclude the base ones
   TString opt = option;
   TBits *bc = new TBits();
   TClass *c1, *c2, *c3;
   UInt_t k = 0;      // will mark k-bit of TBits if the class is a base class

   TIter nextc1(gROOT->GetListOfClasses());
   //gROOT->GetListOfClasses()->ls();    // valid. test
   while((c1 = (TClass *)nextc1())) {

      //   resets bit TClass::kClassSaved for all classes
      c1->ResetBit(TClass::kClassSaved);

      TIter nextc2(gROOT->GetListOfClasses());
      while ((c2 = (TClass *)nextc2())) {
         if (c1==c2) continue;
         else {
            c3 = c2->GetBaseClass(c1);
            if (c3 != 0) {
               bc->SetBitNumber(k, kTRUE);
               break;
            }
         }
      }
      k++;
   }

   TList *ilist = new TList();   // will contain include file names without '.h'
   ilist->SetName("ListOfIncludes");
   gROOT->GetListOfSpecials()->Add(ilist);
   k=0;

   //   completes list of include file names
   TIter nextdo(gROOT->GetListOfClasses());
   while ((c2 = (TClass *)nextdo())) {
      // for used GUI header files
      if (bc->TestBitNumber(k) == 0 && c2->InheritsFrom(TGObject::Class()) == 1) {
         // for any used ROOT header files activate the line below, comment the line above
         //if (bc->TestBitNumber(k) == 0) {
         const char *iname;
         iname = c2->GetDeclFileName();
         if (strlen(iname) != 0 && strstr(iname,".h")) {
            const char *lastsl = strrchr(iname,'/');
            if (lastsl) iname = lastsl + 1;
            char *tname = new char[strlen(iname)+1];
            Int_t i=0;
            while (*iname != '.') {
               tname[i] = *iname;
               i++; iname++;
            }
            tname[i] = 0;    //tname = include file name without '.h'

            TObjString *iel = (TObjString *)ilist->FindObject(tname);
            if (!iel) {
               ilist->Add(new TObjString(tname));
            }
            // Weird, but when saving a canvas, the following two classes
            // may be missing if the toolbar has not been displayed...
            if (strstr(tname, "TRootCanvas")) {
               if (!ilist->FindObject("TGDockableFrame"))
                  ilist->Add(new TObjString("TGDockableFrame"));
               if (!ilist->FindObject("TG3DLine"))
                  ilist->Add(new TObjString("TG3DLine"));
            }
            delete [] tname;
         }
         k++;  continue;
      }
      k++;
   }

   char quote = '"';
   ofstream out;

   TString ff = filename && strlen(filename) ? filename : "Rootappl.C";

   // Computes the main method name.
   const char *fname = gSystem->BaseName(ff.Data());
   Int_t lenfname = strlen(fname);
   char *sname = new char[lenfname+1];

   Int_t i = 0;
   while ((*fname != '.') && (i < lenfname)) {
      sname[i] = *fname;
      i++; fname++;
   }
   if (i == lenfname)
      ff += ".C";
   sname[i] = 0;

   out.open(ff.Data(), ios::out);
   if (!out.good()) {
      Error("SaveSource", "cannot open file: %s", ff.Data());
      delete [] sname;
      return;
   }

   // writes include files in C++ macro
   TObjString *inc;
   ilist = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfIncludes");

   if (!ilist) {
      delete [] sname;
      return;
   }

   // write macro header, date/time stamp as string, and the used Root version
   TDatime t;
   out <<"// Mainframe macro generated from application: "<< gApplication->Argv(0) << endl;
   out <<"// By ROOT version "<< gROOT->GetVersion() <<" on "<<t.AsSQLString()<< endl;
   out << endl;

   TIter nexti(ilist);
   while((inc = (TObjString *)nexti())) {
         out << "#ifndef ROOT_" << inc->GetString() << endl;
         out << "#include " << quote << inc->GetString() << ".h" << quote << endl;
         out << "#endif" << endl;
         if (strstr(inc->GetString(),"TRootEmbeddedCanvas")) {
            out << "#ifndef ROOT_TCanvas" << endl;
            out << "#include " << quote << "TCanvas.h" << quote << endl;
            out << "#endif" << endl;
         }
   }
   out << endl << "#include " << quote << "Riostream.h" << quote << endl;
   // deletes created ListOfIncludes
   gROOT->GetListOfSpecials()->Remove(ilist);
   ilist->Delete();
   delete ilist;
   delete bc;

   // writes the macro entry point equal to the fname
   out << endl;
   out << "void " << sname << "()" << endl;
   out <<"{"<< endl;
   delete [] sname;

   gListOfHiddenFrames->Clear();

   // saivng slots
   TList *lSlots = new TList;
   lSlots->SetName("ListOfSlots");
   gROOT->GetListOfSpecials()->Add(lSlots);

   TGMainFrame::SavePrimitive(out, option);

   GetClassHints((const char *&)fClassName, (const char *&)fResourceName);
   if (strlen(fClassName) || strlen(fResourceName)) {
      out << "   " << GetName() << "->SetClassHints(" << quote << fClassName
          << quote << "," << quote << fResourceName << quote << ");" << endl;
   }

   GetMWMHints(fMWMValue, fMWMFuncs, fMWMInput);
   if (fMWMValue || fMWMFuncs || fMWMInput) {
      out << "   " << GetName() << "->SetMWMHints(";
      out << GetMWMvalueString() << "," << endl;
      out << "                        ";
      out << GetMWMfuncString() << "," << endl;
      out << "                        ";
      out << GetMWMinpString() << ");"<< endl;
   }

///   GetWMPosition(fWMX, fWMY);
///   if ((fWMX != -1) || (fWMY != -1)) {
///      out <<"   "<<GetName()<<"->SetWMPosition("<<fWMX<<","<<fWMY<<");"<<endl;
///   }   // does not work - fixed via Move() below...

   GetWMSize(fWMWidth, fWMHeight);
   if (fWMWidth != UInt_t(-1) || fWMHeight != UInt_t(-1)) {
      out <<"   "<<GetName()<<"->SetWMSize("<<fWMWidth<<","<<fWMHeight<<");"<<endl;
   }

   GetWMSizeHints(fWMMinWidth, fWMMinHeight, fWMMaxWidth, fWMMaxHeight, fWMWidthInc, fWMHeightInc);
   if (fWMMinWidth != UInt_t(-1) || fWMMinHeight != UInt_t(-1) ||
      fWMMaxWidth != UInt_t(-1) || fWMMaxHeight != UInt_t(-1) ||
      fWMWidthInc != UInt_t(-1) || fWMHeightInc != UInt_t(-1)) {
      out <<"   "<<GetName()<<"->SetWMSizeHints("<<fWMMinWidth<<","<<fWMMinHeight
          <<","<<fWMMaxWidth<<","<<fWMMaxHeight
          <<","<<fWMWidthInc<<","<<fWMHeightInc <<");"<<endl;
   }

   out << "   " <<GetName()<< "->MapSubwindows();" << endl;

   TIter nexth(gListOfHiddenFrames);
   TGFrame *fhidden;
   while ((fhidden = (TGFrame*)nexth())) {
      out << "   " <<fhidden->GetName()<< "->UnmapWindow();" << endl;
   }

   out << endl;
   gListOfHiddenFrames->Clear();

   Bool_t usexy = kFALSE;
   TGLayoutManager * lm = GetLayoutManager();
   if (lm->InheritsFrom("TGXYLayout"))
      usexy = kTRUE;

   if (!usexy)
      out << "   " <<GetName()<< "->Resize("<< GetName()<< "->GetDefaultSize());" << endl;
   else
      out << "   " <<GetName()<< "->Resize("<< GetWidth()<<","<<GetHeight()<<");"<<endl;

   out << "   " <<GetName()<< "->MapWindow();" <<endl;

   GetWMPosition(fWMX, fWMY);
   if ((fWMX != -1) || (fWMY != -1)) {
      out <<"   "<<GetName()<<"->Move("<<fWMX<<","<<fWMY<<");"<<endl;
   }

   // needed in case the frame was resized
   // otherwhice the frame became bigger showing all hidden widgets (layout algorithm)
   if (!usexy) out << "   " <<GetName()<< "->Resize("<< GetWidth()<<","<<GetHeight()<<");"<<endl;
   out << "}  " << endl;

   // writing slots
   TList *sl = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfSlots");
   if (sl) {
      TIter nextsl(sl);
      TObjString *slobj;
      Int_t pnumber = 1;

      while ((slobj = (TObjString*) nextsl())) {
         TString s = slobj->GetString();
         TString p = "";
         Int_t lb, rb, eq;
         lb = s.First('(');
         rb = s.First(')');
         eq = s.First('=');
         out << endl;

         if (rb - lb > 1 && eq == -1) {
            p = TString::Format(" par%d", pnumber);
            s.Insert(rb, p);
            pnumber++;
            out << "void " << s << endl;
            out << "{" << endl;
            s = slobj->GetString();
            s[rb] = ' ';
            out << "   cout << " << quote << "Slot " << s  << quote
                << " <<" << p << " << " << quote << ")" << quote
                << " << endl; " << endl;
            } else {
               if (eq != -1) {
                  s.Remove(eq, rb-eq);
                  out << "void " << s << endl;
                  out << "{" << endl;
                  out << "   cout << " << quote << "Slot " << s
                      << quote << " << endl; " << endl;
               } else {
                  out << "void " << slobj->GetString() << endl;
                  out << "{" << endl;
                  out << "   cout << " << quote << "Slot " << slobj->GetString()
                      << quote << " << endl; " << endl;
               }
            }
         out << "}" << endl;
      }
      gROOT->GetListOfSpecials()->Remove(sl);
      sl->Delete();
      delete sl;
   }
   out.close();

   if (!opt.Contains("quiet"))
      printf(" C++ macro file %s has been generated\n", gSystem->BaseName(ff.Data()));

   // reset bit TClass::kClassSaved for all classes
   nextc1.Reset();
   while((c1=(TClass*)nextc1())) {
      c1->ResetBit(TClass::kClassSaved);
   }
}

//______________________________________________________________________________
void TGMainFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a main frame widget as a C++ statement(s) on output stream out.

   if (fParent != gClient->GetDefaultRoot()) { // frame is embedded
      fOptions &= ~kMainFrame;
      TGCompositeFrame::SavePrimitive(out, option);
      fOptions |= kMainFrame;
      return;
   }

   char quote = '"';

   out << endl << "   // main frame" << endl;
   out << "   TGMainFrame *";
   out << GetName() << " = new TGMainFrame(gClient->GetRoot(),10,10,"   // layout alg.
       << GetOptionString() << ");" <<endl;
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   // setting layout manager if it differs from the main frame type
   TGLayoutManager * lm = GetLayoutManager();
   if ((GetOptions() & kHorizontalFrame) &&
       (lm->InheritsFrom(TGHorizontalLayout::Class()))) {
      ;
   } else if ((GetOptions() & kVerticalFrame) &&
              (lm->InheritsFrom(TGVerticalLayout::Class()))) {
      ;
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      lm->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   SavePrimitiveSubframes(out, option);

   if (strlen(fWindowName)) {
      out << "   " << GetName() << "->SetWindowName(" << quote << GetWindowName()
          << quote << ");" << endl;
   }
   if (strlen(fIconName)) {
      out <<"   "<<GetName()<< "->SetIconName("<<quote<<GetIconName()<<quote<<");"<<endl;
   }
   if (strlen(fIconPixmap)) {
      out << "   " << GetName() << "->SetIconPixmap(" << quote << GetIconPixmap()
          << quote << ");" << endl;
   }
}

//______________________________________________________________________________
void TGHorizontalFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a horizontal frame widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // horizontal frame" << endl;
   out << "   TGHorizontalFrame *";
   out << GetName() << " = new TGHorizontalFrame(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out << ");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   // setting layout manager if it differs from the main frame type
   TGLayoutManager * lm = GetLayoutManager();
   if ((GetOptions() & kHorizontalFrame) &&
       (lm->InheritsFrom(TGHorizontalLayout::Class()))) {
      ;
   } else if ((GetOptions() & kVerticalFrame) &&
              (lm->InheritsFrom(TGVerticalLayout::Class()))) {
      ;
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      lm->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   SavePrimitiveSubframes(out, option);
}

//______________________________________________________________________________
void TGVerticalFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save a vertical frame widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // vertical frame" << endl;
   out << "   TGVerticalFrame *";
   out << GetName() << " = new TGVerticalFrame(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   // setting layout manager if it differs from the main frame type
   TGLayoutManager * lm = GetLayoutManager();
   if ((GetOptions() & kHorizontalFrame) &&
       (lm->InheritsFrom(TGHorizontalLayout::Class()))) {
      ;
   } else if ((GetOptions() & kVerticalFrame) &&
              (lm->InheritsFrom(TGVerticalLayout::Class()))) {
      ;
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      lm->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   SavePrimitiveSubframes(out, option);
}

//______________________________________________________________________________
void TGFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a frame widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << "   TGFrame *";
   out << GetName() << " = new TGFrame("<< fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;
}

//______________________________________________________________________________
void TGGroupFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a group frame widget as a C++ statement(s) on output stream out.

   char quote = '"';

   // font + GC
   option = GetName()+5;         // unique digit id of the name
   TString parGC, parFont;
   parFont.Form("%s::GetDefaultFontStruct()",IsA()->GetName());
   parGC.Form("%s::GetDefaultGC()()",IsA()->GetName());

   if ((GetDefaultFontStruct() != fFontStruct) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = gClient->GetResourcePool()->GetFontPool()->FindFont(fFontStruct);
      if (ufont) {
         ufont->SavePrimitive(out, option);
         parFont.Form("ufont->GetFontStruct()");
      }

      TGGC *userGC = gClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         parGC.Form("uGC->GetGC()");
      }
   }

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // " << quote << GetTitle() << quote << " group frame" << endl;
   out << "   TGGroupFrame *";
   out << GetName() <<" = new TGGroupFrame("<<fParent->GetName()
       << "," << quote << GetTitle() << quote;

   if (fBackground == GetDefaultFrameBackground()) {
      if (fFontStruct == GetDefaultFontStruct()) {
         if (fNormGC == GetDefaultGC()()) {
            if (GetOptions() & kVerticalFrame) {
               out <<");" << endl;
            } else {
               out << "," << GetOptionString() <<");" << endl;
            }
         } else {
            out << "," << GetOptionString() << "," << parGC.Data() <<");" << endl;
         }
      } else {
         out << "," << GetOptionString() << "," << parGC.Data() << "," << parFont.Data() << ");" << endl;
      }
   } else {
      out << "," << GetOptionString() << "," << parGC.Data() << "," << parFont.Data() << ",ucolor);"  << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   if (GetTitlePos() != -1)
      out << "   " << GetName() <<"->SetTitlePos(";
   if (GetTitlePos() == 0)
      out << "TGGroupFrame::kCenter);" << endl;
   if (GetTitlePos() == 1)
      out << "TGGroupFrame::kRight);" << endl;

   SavePrimitiveSubframes(out, option);

   // setting layout manager
   out << "   " << GetName() <<"->SetLayoutManager(";
   GetLayoutManager()->SavePrimitive(out, option);
   out << ");"<< endl;

   out << "   " << GetName() <<"->Resize(" << GetWidth() << ","
       << GetHeight() << ");" << endl;
}


//______________________________________________________________________________
void TGTransientFrame::SaveSource(const char *filename, Option_t *option)
{
   // Save the GUI tranzient frame widget in a C++ macro file.

   // iterate over all active classes to exclude the base ones

   TString opt = option;
   TBits *bc = new TBits();
   TClass *c1, *c2, *c3;
   UInt_t k = 0;      // will mark k-bit of TBits if the class is a base class

   TIter nextc1(gROOT->GetListOfClasses());
   while((c1 = (TClass *)nextc1())) {

      //   resets bit TClass::kClassSaved for all classes
      c1->ResetBit(TClass::kClassSaved);

      TIter nextc2(gROOT->GetListOfClasses());
      while ((c2 = (TClass *)nextc2())) {
         if (c1==c2) continue;
         else {
            c3 = c2->GetBaseClass(c1);
            if (c3 != 0) {
               bc->SetBitNumber(k, kTRUE);
               break;
            }
         }
      }
      k++;
   }

   TList *ilist = new TList();   // will contain include file names without '.h'
   ilist->SetName("ListOfIncludes");
   gROOT->GetListOfSpecials()->Add(ilist);
   k=0;

   // completes list of include file names
   TIter nextdo(gROOT->GetListOfClasses());
   while ((c2 = (TClass *)nextdo())) {
      // to have only used GUI header files
      if (bc->TestBitNumber(k) == 0 && c2->InheritsFrom(TGObject::Class()) == 1) {
         // for any used ROOT header files activate the line below, comment the line above
         //if (bc->TestBitNumber(k) == 0) {
         const char *iname;
         iname = c2->GetDeclFileName();
         if (strlen(iname) != 0 && strstr(iname,".h")) {
            const char *lastsl = strrchr(iname,'/');
            if (lastsl) iname = lastsl + 1;
            char *tname = new char[strlen(iname)+1];
            Int_t i=0;
            while (*iname != '.') {
               tname[i] = *iname;
               i++; iname++;
            }
            tname[i] = 0;    //tname = include file name without '.h'

            TObjString *iel = (TObjString *)ilist->FindObject(tname);
            if (!iel) {
               ilist->Add(new TObjString(tname));
            }
            delete [] tname;
         }
         k++;  continue;
      }
      k++;
   }

   char quote = '"';
   ofstream out;

   TString ff = filename && strlen(filename) ? filename : "Rootdlog.C";

   // Computes the main method name.
   const char *fname = gSystem->BaseName(ff.Data());
   Int_t lenfname = strlen(fname);
   char *sname = new char[lenfname+1];

   Int_t i = 0;
   while ((*fname != '.') && (i < lenfname)) {
      sname[i] = *fname;
      i++; fname++;
   }
   if (i == lenfname)
      ff += ".C";
   sname[i] = 0;

   out.open(ff.Data(), ios::out);
   if (!out.good()) {
      Error("SaveSource", "cannot open file: %s", ff.Data());
      delete [] sname;
      return;
   }

   // writes include files in C++ macro
   TObjString *inc;
   ilist = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfIncludes");

   if (!ilist) {
      delete [] sname;
      return;
   }

   // write macro header, date/time stamp as string, and the used Root version
   TDatime t;
   out <<"// Dialog macro generated from application: "<< gApplication->Argv(0) << endl;
   out <<"// By ROOT version "<< gROOT->GetVersion() <<" on "<<t.AsSQLString()<< endl;
   out << endl;

   out << "#if !defined( __CINT__) || defined (__MAKECINT__)" << endl << endl;

   TIter nexti(ilist);
   while((inc = (TObjString *)nexti())) {
      out <<"#ifndef ROOT_"<< inc->GetString() << endl;
      out <<"#include "<< quote << inc->GetString() <<".h"<< quote << endl;
      out <<"#endif" << endl;
      if (strstr(inc->GetString(),"TRootEmbeddedCanvas")) {
         out <<"#ifndef ROOT_TCanvas"<< endl;
         out <<"#include "<< quote <<"TCanvas.h"<< quote << endl;
         out <<"#endif" << endl;
      }
   }
   out << endl << "#include " << quote << "Riostream.h" << quote << endl;
   out << endl << "#endif" << endl;
   // deletes created ListOfIncludes
   gROOT->GetListOfSpecials()->Remove(ilist);
   ilist->Delete();
   delete ilist;
   delete bc;

   // writes the macro entry point equal to the fname
   out << endl;
   out << "void " << sname << "()" << endl;
   delete [] sname;

   //  Save GUI widgets as a C++ macro in a file
   out <<"{"<< endl;

   gListOfHiddenFrames->Clear();

   // saivng slots
   TList *lSlots = new TList;
   lSlots->SetName("ListOfSlots");
   gROOT->GetListOfSpecials()->Add(lSlots);

   TGTransientFrame::SavePrimitive(out, option);

   GetClassHints((const char *&)fClassName, (const char *&)fResourceName);
   if (strlen(fClassName) || strlen(fResourceName)) {
      out<<"   "<<GetName()<< "->SetClassHints("<<quote<<fClassName<<quote
                                            <<"," <<quote<<fResourceName<<quote
                                            <<");"<<endl;
   }

   GetMWMHints(fMWMValue, fMWMFuncs, fMWMInput);
   if (fMWMValue || fMWMFuncs || fMWMInput) {
      out << "   " << GetName() << "->SetMWMHints(";
      out << GetMWMvalueString() << "," << endl;
      out << "                        ";
      out << GetMWMfuncString() << "," << endl;
      out << "                        ";
      out << GetMWMinpString() << ");"<< endl;
   }

   GetWMPosition(fWMX, fWMY);
   if ((fWMX != -1) || (fWMY != -1)) {
      out <<"   "<<GetName()<<"->SetWMPosition("<<fWMX<<","<<fWMY<<");"<<endl;
   }

   GetWMSize(fWMWidth, fWMHeight);
   if (fWMWidth != UInt_t(-1) || fWMHeight != UInt_t(-1)) {
      out <<"   "<<GetName()<<"->SetWMSize("<<fWMWidth<<","<<fWMHeight<<");"<<endl;
   }

   GetWMSizeHints(fWMMinWidth,fWMMinHeight,fWMMaxWidth,fWMMaxHeight,fWMWidthInc,fWMHeightInc);
   if (fWMMinWidth != UInt_t(-1) || fWMMinHeight != UInt_t(-1) ||
       fWMMaxWidth != UInt_t(-1) || fWMMaxHeight != UInt_t(-1) ||
       fWMWidthInc != UInt_t(-1) || fWMHeightInc != UInt_t(-1)) {

      out <<"   "<<GetName()<<"->SetWMSizeHints("<<fWMMinWidth<<","<<fWMMinHeight
          <<","<<fWMMaxWidth<<","<<fWMMaxHeight <<","<<fWMWidthInc<<","<<fWMHeightInc
          <<");"<<endl;
   }

   GetWMPosition(fWMX, fWMY);
   if ((fWMX != -1) || (fWMY != -1)) {
      out <<"   "<<GetName()<<"->Move("<<fWMX<<","<<fWMY<<");"<<endl;
   }

   out << "   " <<GetName()<< "->MapSubwindows();" << endl;

   TIter nexth(gListOfHiddenFrames);
   TGFrame *fhidden;
   while ((fhidden = (TGFrame*)nexth())) {
      out << "   " <<fhidden->GetName()<< "->UnmapWindow();" << endl;
   }
   out << endl;
   gListOfHiddenFrames->Clear();

   Bool_t usexy = kFALSE;
   TGLayoutManager * lm = GetLayoutManager();
   if (lm->InheritsFrom("TGXYLayout"))
      usexy = kTRUE;

   if (!usexy)
      out << "   " <<GetName()<< "->Resize("<< GetName()<< "->GetDefaultSize());" << endl;
   else
      out << "   " <<GetName()<< "->Resize("<< GetWidth()<<","<<GetHeight()<<");"<<endl;

   out << "   " <<GetName()<< "->MapWindow();" <<endl;
   if (!usexy) out << "   " <<GetName()<< "->Resize();" << endl;
   out << "}  " << endl;

   // writing slots
   TList *sl = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfSlots");
   if (sl) {
      TIter nextsl(sl);
      TObjString *slobj;
      Int_t pnumber = 1;

      while ((slobj = (TObjString*) nextsl())) {
         TString s = slobj->GetString();
         TString p = "";
         Int_t lb, rb, eq;
         lb = s.First('(');
         rb = s.First(')');
         eq = s.First('=');
         out << endl;

         if (rb - lb > 1 && eq == -1) {
            p = TString::Format(" par%d", pnumber);
            s.Insert(rb, p);
            pnumber++;
            out << "void " << s << endl;
            out << "{" << endl;
            s = slobj->GetString();
            s[rb] = ' ';
            out << "   cout << " << quote << "Slot " << s  << quote
                << " <<" << p << " << " << quote << ")" << quote
                << " << endl; " << endl;
            } else {
               if (eq != -1) {
                  s.Remove(eq, rb-eq);
                  out << "void " << s << endl;
                  out << "{" << endl;
                  out << "   cout << " << quote << "Slot " << s
                      << quote << " << endl; " << endl;
               } else {
                  out << "void " << slobj->GetString() << endl;
                  out << "{" << endl;
                  out << "   cout << " << quote << "Slot " << slobj->GetString()
                      << quote << " << endl; " << endl;
               }
            }
         out << "}" << endl;
      }
      gROOT->GetListOfSpecials()->Remove(sl);
      sl->Delete();
      delete sl;
   }

   out.close();

   if (!opt.Contains("quiet"))
      printf(" C++ macro file %s has been generated\n", gSystem->BaseName(ff.Data()));

   // reset bit TClass::kClassSaved for all classes
   nextc1.Reset();
   while((c1=(TClass*)nextc1())) {
      c1->ResetBit(TClass::kClassSaved);
   }
}

//______________________________________________________________________________
void TGTransientFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a transient frame widget as a C++ statement(s) on output stream out.

   char quote = '"';

   out << endl << "   // transient frame" << endl;
   out << "   TGTransientFrame *";
   out << GetName()<<" = new TGTransientFrame(gClient->GetRoot(),0"
       << "," << GetWidth() << "," << GetHeight() << "," << GetOptionString() <<");" << endl;

   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   // setting layout manager if it differs from transient frame type
   TGLayoutManager * lm = GetLayoutManager();
   if ((GetOptions() & kHorizontalFrame) &&
       (lm->InheritsFrom(TGHorizontalLayout::Class()))) {
      ;
   } else if ((GetOptions() & kVerticalFrame) &&
              (lm->InheritsFrom(TGVerticalLayout::Class()))) {
      ;
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      lm->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   SavePrimitiveSubframes(out, option);

   if (strlen(fWindowName)) {
      out << "   " << GetName() << "->SetWindowName(" << quote << GetWindowName()
          << quote << ");" << endl;
   }
   if (strlen(fIconName)) {
      out <<"   "<<GetName()<< "->SetIconName("<<quote<<GetIconName()<<quote<<");"<<endl;
   }
   if (strlen(fIconPixmap)) {
      out << "   " << GetName() << "->SetIconPixmap(" << quote << GetIconPixmap()
          << quote << ");" << endl;
   }
}
