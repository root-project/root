// @(#)root/gui:$Name:  $:$Id: TGFrame.cxx,v 1.44 2004/01/28 19:06:05 brun Exp $
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

#include "TGFrame.h"
#include "TGResourcePool.h"
#include "TGPicture.h"
#include "TList.h"
#include "TApplication.h"
#include "TTimer.h"
#include "Riostream.h"

#include "TObjString.h"
#include "TObjArray.h"
#include "TBits.h"
#include "TColor.h"
#include "TROOT.h"
#include "KeySymbols.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TSystem.h"
#include "TContextMenu.h"


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

TContextMenu *TGCompositeFrame::fgContextMenu = 0;
TGLayoutHints *TGCompositeFrame::fgDefaultHints = new TGLayoutHints;

static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                         "All files",   "*",
                                          0,             0 };

ClassImp(TGFrame)
ClassImp(TGCompositeFrame)
ClassImp(TGVerticalFrame)
ClassImp(TGHorizontalFrame)
ClassImp(TGMainFrame)
ClassImp(TGTransientFrame)
ClassImp(TGGroupFrame)


//______________________________________________________________________________
TGFrame::TGFrame(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options, ULong_t back)
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

   fBackground = back;
   fOptions    = options;
   fWidth = w; fHeight = h; fX = fY = fBorderWidth = 0;

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
   gVirtualX->GetWindowAttributes(id, attributes);

   fX           = attributes.fX;
   fY           = attributes.fY;
   fWidth       = attributes.fWidth;
   fHeight      = attributes.fHeight;
   fBorderWidth = attributes.fBorderWidth;
   fEventMask   = (UInt_t) attributes.fYourEventMask;
   fBackground  = 0;
   fOptions     = 0;
}

//______________________________________________________________________________
void TGFrame::DeleteWindow()
{
   // Delete window. Use single shot timer to call final delete method.
   // We use this inderect way since deleting the window in its own
   // execution "thread" can cause side effects because frame methods
   // can still be called while the window object has already been deleted.

   TTimer::SingleShot(50, IsA()->GetName(), this, "ReallyDelete()");
}

//______________________________________________________________________________
void TGFrame::ChangeBackground(ULong_t back)
{
   // Change frame background color.

   fBackground = back;
   gVirtualX->SetWindowBackground(fId, back);
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
      if (options & (kSunkenFrame | kRaisedFrame))
         fBorderWidth = (options & kDoubleBorder) ? 2 : 1;
      else
         fBorderWidth = 0;
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

   if (IsEditEvent(event)) return HandleEditEvent(event);

   switch (event->fType) {

      case kExpose:
         HandleExpose(event);
         break;

      case kConfigureNotify:
         while (gVirtualX->CheckEvent(fId, kConfigureNotify, *event))
            ;
         HandleConfigureNotify(event);
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
                (event->fWindow == fgDbw)) dbl_clk = kTRUE;

             fgLastClick = event->fTime;
             fgLastButton = event->fCode;
             fgDbx = event->fXRoot;
             fgDby = event->fYRoot;
             fgDbw = event->fWindow;

            if (!dbl_clk && IsEditEvent(event) && 
                (event->fCode == kButton3)) {
               OnContextMenu(event);
            }

            if (dbl_clk) {
               if (!HandleDoubleClick(event)) {
                  HandleButton(event);
               }
            } else {
               HandleButton(event);
            }
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

   if (TestBit(kNotDeleted))
      ProcessedEvent(event);  // emit signal

   return kTRUE;
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
      TGDimension siz = GetDefaultSize();
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
   TGDimension siz = GetDefaultSize();
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

   cout <<  option << ClassName() << ":\tid=" << fId << " parent=" << fParent->GetId();
   cout << " x=" << fX << " y=" << fY;
   cout << " w=" << fWidth << " h=" << fHeight << endl;
}

//______________________________________________________________________________
TGCompositeFrame::TGCompositeFrame(const TGWindow *p, UInt_t w, UInt_t h,
         UInt_t options, ULong_t back) : TGFrame(p, w, h, options, back)
{
   // Create a composite frame. A composite frame has in addition to a TGFrame
   // also a layout manager and a list of child frames.

   fLayoutManager = 0;
   fList          = new TList;
   fLayoutBroken  = kFALSE;

   if (fOptions & kHorizontalFrame)
      SetLayoutManager(new TGHorizontalLayout(this));
   else
      SetLayoutManager(new TGVerticalLayout(this));
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

   SetLayoutManager(new TGVerticalLayout(this));
}

//______________________________________________________________________________
TGCompositeFrame::~TGCompositeFrame()
{
   // Delete a composite frame.

   if (fList) fList->Delete();
   delete fList;
   delete fLayoutManager;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::IsEditable() const
{
   // return kTRUE if frame is being eddited

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
   //    gROOT->Macro("$ROOTSYS/tutorials/guitest.C");
   //    m->SetEditable(0);
   //    m->MapWindow();
   //
   //  2. ... to be continued   

   if (on) {
      fClient->SetRoot(this);
   } else {
      fClient->SetRoot(0);
      Resize();
   }
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::HandleEditEvent(Event_t *)
{
   // Edit events allow to move, resize, remove frames
   // from the composite frame.

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::OnContextMenu(Event_t *event)
{
   // Handle context menu

   if (!fgContextMenu) fgContextMenu = new TContextMenu("GUI context menu");

   TGFrame *f = GetFrameFromPoint(event->fX, event->fY);

   if (f) fgContextMenu->Popup(event->fXRoot, event->fYRoot, f, (TBrowser*)NULL);
   else return kFALSE;

   return kTRUE;
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
      delete el->fFrame;
      if (el->fLayout != fgDefaultHints)
         delete el->fLayout;
   }
   fList->Delete();
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
   //  set broken layout. No Layout method is called

   fLayoutBroken = on;
   if (!fLayoutBroken) Resize();
}

//______________________________________________________________________________
void TGCompositeFrame::ChangeOptions(UInt_t options)
{
   // Change composite frame options. Options is an OR of the EFrameTypes.

  TGFrame::ChangeOptions(options);

  if (options & kHorizontalFrame)
     SetLayoutManager(new TGHorizontalLayout(this));
  else
     SetLayoutManager(new TGVerticalLayout(this));
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

   TGFrameElement *nw;

   nw = new TGFrameElement;
   nw->fFrame  = f;
   nw->fLayout = l ? l : fgDefaultHints;
   nw->fState  = 1;
   fList->Add(nw);
}

//______________________________________________________________________________
void TGCompositeFrame::RemoveFrame(TGFrame *f)
{
   // Remove frame from composite frame.

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f) {
         fList->Remove(el);
         delete el;
         break;
      }
}

//______________________________________________________________________________
void TGCompositeFrame::MapSubwindows()
{
   // Map all sub windows that are part of the composite frame.

   TGWindow::MapSubwindows();

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      el->fFrame->MapSubwindows();
}

//______________________________________________________________________________
void TGCompositeFrame::HideFrame(TGFrame *f)
{
   // Hide sub frame.

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f) {
         el->fState = 0;
         el->fFrame->UnmapWindow();
         Layout();
         break;
      }
}

//______________________________________________________________________________
void TGCompositeFrame::ShowFrame(TGFrame *f)
{
   // Show sub frame.

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f) {
         el->fState = 1;
         el->fFrame->MapWindow();
         Layout();
         break;
      }
}

//______________________________________________________________________________
Int_t TGCompositeFrame::GetState(TGFrame *f) const
{
   // Get state of sub frame.

   if (!fList) return 0;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f)
         return el->fState;

   return 0;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::IsVisible(TGFrame *f) const
{
   // Get state of sub frame.

   if (!fList) return kFALSE;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f)
         return (el->fState & kIsVisible);

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGCompositeFrame::IsArranged(TGFrame *f) const
{
   // Get state of sub frame.

   if (!fList) return kFALSE;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next()))
      if (el->fFrame == f)
         return (el->fState & kIsArranged);

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

   gVirtualX->GrabKey(fId, kKey_s, kKeyControlMask, kTRUE);

   AddInput(kKeyPressMask | kKeyReleaseMask);

   // if parent is editting/embedable add this frame to the parent
   if (fClient->IsEditable() && (p==fClient->GetRoot())) {
      TGCompositeFrame *frame;
      if (p->InheritsFrom(TGCompositeFrame::Class())) {
         frame = (TGCompositeFrame*)p;
         frame->AddFrame(this);
      }
   }
}

//______________________________________________________________________________
TGMainFrame::~TGMainFrame()
{
   // TGMainFrame destructor.

   if (fBindList) {
      fBindList->Delete();
      delete fBindList;
   }
   DestroyWindow();
}

//______________________________________________________________________________
Bool_t TGMainFrame::HandleKey(Event_t *event)
{
   // Handle keyboard events.

   if ((event->fType == kGKeyPress) && (event->fState & kKeyControlMask)) {
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);
      if (str[0] == 19) {  // ctrl-s
         static TString dir(".");
         TGFileInfo fi;
         fi.fFileTypes = gSaveMacroTypes;
         fi.fIniDir    = StrDup(dir);
         new TGFileDialog(fClient->GetRoot(), this, kFDSave, &fi);
         if (!fi.fFilename) return kTRUE;
         dir = fi.fIniDir;
         const char *fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));
         if (strstr(fname, ".C"))
            SaveSource(fname, "");
         else {
            Int_t retval;
            new TGMsgBox(fClient->GetRoot(), this, "Error...",
                        Form("file (%s) must have extension .C", fname),
                        kMBIconExclamation, kMBRetry | kMBCancel, &retval);
            if (retval == kMBRetry)
               HandleKey(event);
         }
         return kTRUE;
      }
   }

   if (!fBindList) return kFALSE;

   TIter next(fBindList);
   TGMapKey *m;
   TGFrame  *w;

   while ((m = (TGMapKey *) next())) {
      if (m->fKeyCode == event->fCode) {
         w = (TGFrame *) m->fWindow;
         return w->HandleKey(event);
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGMainFrame::BindKey(const TGWindow *w, Int_t keycode, Int_t modifier) const
{
   // Bind key to a window.

   if (fBindList) {
      TGMapKey *m = new TGMapKey(keycode, (TGWindow *)w);
      fBindList->Add(m);
      gVirtualX->GrabKey(fId, keycode, modifier, kTRUE);
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
   // Close main frame. We get here in response to ALT+F4 or a window
   // manager close command. To terminate the application when this
   // happens override this method and call gApplication->Terminate(0) or
   // make a connection to this signal. If not the window will be just
   // destroyed and can not be used anymore.

   DestroyWindow();
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

   fWindowName = name;
   gVirtualX->SetWindowName(fId, (char *)name);
}

//______________________________________________________________________________
void TGMainFrame::SetIconName(const char *name)
{
   // Set window icon name. This is typically done via the window manager.

   fIconName = name;
   gVirtualX->SetIconName(fId, (char *)name);
}

//______________________________________________________________________________
void TGMainFrame::SetIconPixmap(const char *iconName)
{
   // Set window icon pixmap by name. This is typically done via the window
   // manager.

   fIconPixmap = iconName;
   const TGPicture *iconPic = fClient->GetPicture(iconName);
   if (iconPic) {
      Pixmap_t pic = iconPic->GetPicture();
      gVirtualX->SetIconPixmap(fId, pic);
   }
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

   if (fClient->IsEditable() && (fParent==fClient->GetRoot())) return;

   fMWMValue = value;
   fMWMFuncs = funcs;
   fMWMInput = input;
   gVirtualX->SetMWMHints(fId, value, funcs, input);
}

//______________________________________________________________________________
void TGMainFrame::SetWMPosition(Int_t x, Int_t y)
{
   // Give the window manager a window position hint.

   if (fClient->IsEditable() && (fParent==fClient->GetRoot())) return;

   fWMX = x;
   fWMY = y;
   gVirtualX->SetWMPosition(fId, x, y);
}

//______________________________________________________________________________
void TGMainFrame::SetWMSize(UInt_t w, UInt_t h)
{
   // Give the window manager a window size hint.

   if (fClient->IsEditable() && (fParent==fClient->GetRoot())) return;

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

   if (fClient->IsEditable() && (fParent==fClient->GetRoot())) return;

   fWMMinWidth  = wmin;
   fWMMinHeight = hmin;
   fWMMaxWidth  = wmax;
   fWMMaxHeight = hmax;
   fWMWidthInc  = winc;
   fWMHeightInc = hinc;
   gVirtualX->SetWMSizeHints(fId, wmin, hmin, wmax, hmax, winc, hinc);
}

//______________________________________________________________________________
void TGMainFrame::SetWMState(EInitialState state)
{
   // Set the initial state of the window. Either kNormalState or kIconicState.

   if (fClient->IsEditable() && (fParent==fClient->GetRoot())) return;

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

   if (fMain) {
      gVirtualX->SetWMTransientHint(fId, fMain->GetId());
   }
}

//______________________________________________________________________________
TGGroupFrame::TGGroupFrame(const TGWindow *p, TGString *title,
                           UInt_t options, GContext_t norm,
                           FontStruct_t font, ULong_t back) :
   TGCompositeFrame(p, 1, 1, options, back)
{
   // Create a group frame. The title will be adopted and deleted by the
   // group frame.

   fText       = title;
   fFontStruct = font;
   fNormGC     = norm;
   fTitlePos   = kLeft;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fBorderWidth = max_ascent + max_descent + 1;
}

//______________________________________________________________________________
TGGroupFrame::TGGroupFrame(const TGWindow *p, const char *title,
                           UInt_t options, GContext_t norm,
                           FontStruct_t font, ULong_t back) :
   TGCompositeFrame(p, 1, 1, options, back)
{
   // Create a group frame.

   fText       = new TGString(title);
   fFontStruct = font;
   fNormGC     = norm;
   fTitlePos   = kLeft;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fBorderWidth = max_ascent + max_descent + 1;
}

//______________________________________________________________________________
TGGroupFrame::~TGGroupFrame()
{
   // Delete a group frame.

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
   b = fHeight - 1;

   sep = 3;
   UInt_t rr = 5 + (sep << 1) + tw;

   switch (fTitlePos) {
      case kRight:
         gl = fWidth>rr ? fWidth - rr : 5 + sep;
         break;
      case kCenter:
         gl = fWidth>tw ? ((fWidth - tw)>>1) - sep : 5 + sep;
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
   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGGroupFrame::GetDefaultGC()
{
   if (!fgDefaultGC)
      fgDefaultGC = gClient->GetResourcePool()->GetFrameGC();
   return *fgDefaultGC;
}

//______________________________________________________________________________
void TGFrame::SaveUserColor(ofstream &out, Option_t *)
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
   ULong_t ucolor = GetBackground();
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
void TGCompositeFrame::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a composite frame widget as a C++ statement(s) on output stream out

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

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

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");"<< endl;
   }

   TGLayoutManager *lm = GetLayoutManager();
   if (GetOptions() & kHorizontalFrame) {
      if (lm->InheritsFrom(TGHorizontalLayout::Class())) { }
   } else if (GetOptions() & kVerticalFrame) {
      if (lm->InheritsFrom(TGVerticalLayout::Class())) { }
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      GetLayoutManager()->SavePrimitive(out, option);
      out << ");" << endl;
   }
   
}

//______________________________________________________________________________
void TGMainFrame::SaveSource(const char *filename, Option_t *option)
{
   // Save the GUI main frame widget in a C++ macro file

   //   iteration over all active classes to exclude the base ones

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
               char *tname = new char[strlen(iname)];
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
   Int_t lenfile = strlen(filename);
   char *fname;

   //	 if filename is given, open this file, otherwise create a file Rootappl.C

   if (lenfile) {
      fname = (char *)filename;
   } else {
      fname="Rootappl.C"; lenfile = 10;
   }

   out.open(fname, ios::out);

   if (!out.good()) {
       Error("SaveSource", "cannot open file: %s", fname);
       if (!lenfile) delete [] fname;
       return;
   }

   //   writes include files in C++ macro
   TObjString *inc;
   ilist = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfIncludes");

   if (!ilist) return;

   //  Write macro header, date/time stamp as string, and the used Root version
   TDatime t;
   out <<"// Mainframe macro generated from application: "<< gApplication->Argv(0) << endl;
   out <<"// By ROOT version "<< gROOT->GetVersion() <<" on "<<t.AsSQLString()<< endl;
   out << endl;

   out << "#if !defined( __CINT__) || defined (__MAKECINT__)" << endl << endl;

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
   out << endl << "#endif" << endl;
   //    deletes created ListOfIncludes
   gROOT->GetListOfSpecials()->Remove(ilist);
   ilist->Delete();
   delete ilist;
   delete bc;

   // Does not work when filename contains dots (RDM)
   // writes the macro entry point equal to the filename
   char *sname = new char[lenfile];
   Int_t i=0;
   while (*fname != '.') {
       sname[i] = *fname;
       i++; fname++;
   }
   sname[i] = 0;

   out << endl;
   out << "void " << sname << "()" << endl;
   delete [] sname;

   out <<"{"<< endl;

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
   out << "   " <<GetName()<< "->Resize("<< GetName()<< "->GetDefaultSize());" << endl;
   out << "   " <<GetName()<< "->MapWindow();" <<endl;

   GetWMPosition(fWMX, fWMY);
   if ((fWMX != -1) || (fWMY != -1)) {
      out <<"   "<<GetName()<<"->Move("<<fWMX<<","<<fWMY<<");"<<endl;
   }

   // needed in case the frame was resized
   // otherwhice the frame became bigger showing all hidden widgets (layout algorithm)
   out << "   " <<GetName()<< "->Resize("<< GetWidth()<<","<<GetHeight()<<");"<<endl;

   out << "}  " << endl;

   out.close();

   Printf(" C++ macro file %s has been generated", fname-i);

   // reset bit TClass::kClassSaved for all classes
   nextc1.Reset();
   while((c1=(TClass*)nextc1())) {
      c1->ResetBit(TClass::kClassSaved);
   }
   if (!lenfile) delete [] fname;
}

//______________________________________________________________________________
void TGMainFrame::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a main frame widget as a C++ statement(s) on output stream out.

   if (fParent!=gClient->GetDefaultRoot()) { // frame is embedded 
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

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");" << endl;
   }
   out << endl;

   // setting layout manager if it differs from the main frame type

   TGLayoutManager * lm = GetLayoutManager();

   if (GetOptions() & kHorizontalFrame) {
      if (lm->InheritsFrom(TGHorizontalLayout::Class())) { }
   } else if (GetOptions() & kVerticalFrame) {
      if (lm->InheritsFrom(TGVerticalLayout::Class())) { }
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      GetLayoutManager()->SavePrimitive(out, option);
      out << ");"<< endl;
   }

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
void TGHorizontalFrame::SavePrimitive(ofstream &out, Option_t *option)
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

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");" << endl;
   }
}

//______________________________________________________________________________
void TGVerticalFrame::SavePrimitive(ofstream &out, Option_t *option)
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

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");" << endl;
   }
}

//______________________________________________________________________________
void TGFrame::SavePrimitive(ofstream &out, Option_t *option)
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
}

//______________________________________________________________________________
void TGGroupFrame::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a group frame widget as a C++ statement(s) on output stream out

   char quote = '"';

   // font + GC
   option = GetName()+5;         // unique digit id of the name
   char ParGC[50], ParFont[50];
   sprintf(ParFont,"%s::GetDefaultFontStruct()",IsA()->GetName());
   sprintf(ParGC,"%s::GetDefaultGC()()",IsA()->GetName());
   
   if ((GetDefaultFontStruct() != fFontStruct) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = gClient->GetResourcePool()->GetFontPool()->FindFont(fFontStruct);
      if (ufont) {
         ufont->SavePrimitive(out, option);
         sprintf(ParFont,"ufont->GetFontStruct()");
      } 

      TGGC *userGC = gClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         sprintf(ParGC,"uGC->GetGC()");
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
            out << "," << GetOptionString() << "," << ParGC <<");" << endl;
         }
      } else {
         out << "," << GetOptionString() << "," << ParGC << "," << ParFont << ");" << endl;
      }
   } else {
      out << "," << GetOptionString() << "," << ParGC << "," << ParFont << ",ucolor);"  << endl;
   }

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");" << endl;
   }

   if (GetTitlePos() != -1)
      out << "   " << GetName() <<"->SetTitlePos(";
   if (GetTitlePos() == 0)
      out << "TGGroupFrame::kCenter);" << endl;
   if (GetTitlePos() == 1)
      out << "TGGroupFrame::kRight);" << endl;

   // setting layout manager if different from frame type
   out << "   " << GetName() <<"->SetLayoutManager(";
   GetLayoutManager()->SavePrimitive(out, option);
   out << ");"<< endl;

   out << "   " << GetName() <<"->Resize();" << endl;
}


//______________________________________________________________________________
void TGTransientFrame::SaveSource(const char *filename, Option_t *option)
{
   // Save the GUI tranzient frame widget in a C++ macro file

   // iterate over all active classes to exclude the base ones

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
               char *tname = new char[strlen(iname)];
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
   Int_t lenfile = strlen(filename);
   char *fname;

   // if filename is given, open this file, otherwise create a file Rootappl.C

   if (lenfile) {
      fname = (char *)filename;
   } else {
      fname = "Rootdialog.C";
      lenfile = 10;
   }

   out.open(fname, ios::out);

   if (!out.good()) {
       Error("SaveSource", "cannot open file: %s", fname);
       if (!lenfile) delete [] fname;
       return;
   }

   //   writes include files in C++ macro
   TObjString *inc;
   ilist = (TList *)gROOT->GetListOfSpecials()->FindObject("ListOfIncludes");

   if (!ilist) return;

   //  Write macro header, date/time stamp as string, and the used Root version
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
   out << endl << "#endif" << endl;
   // deletes created ListOfIncludes
   gROOT->GetListOfSpecials()->Remove(ilist);
   ilist->Delete();
   delete ilist;
   delete bc;

   // Does not work when filename contains more dots (RDM)
   // writes the macro entry point equal to the filename
   char *sname = new char[lenfile];
   Int_t i=0;
   while (*fname != '.') {
      sname[i] = *fname;
      i++; fname++;
   }
   sname[i] = 0;

   out << endl;
   out <<"void "<< sname << "()" << endl;
   delete [] sname;

   //  Save GUI widgets as a C++ macro in a file
   out <<"{"<< endl;
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
   out << "   " <<GetName()<< "->Resize("<< GetName()<< "->GetDefaultSize());" << endl;
   out << "   " <<GetName()<< "->MapWindow();" <<endl;
   out << "   " <<GetName()<< "->Resize();" << endl;
   out << "}  " << endl;

   out.close();

   Printf(" C++ macro file %s has been generated", fname-i);

   // reset bit TClass::kClassSaved for all classes
   nextc1.Reset();
   while((c1=(TClass*)nextc1())) {
      c1->ResetBit(TClass::kClassSaved);
   }
   if (!lenfile) delete [] fname;
}

//______________________________________________________________________________
void TGTransientFrame::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a transient frame widget as a C++ statement(s) on output stream out.

   char quote = '"';

   out << endl << "   // transient frame" << endl;
   out << "   TGTransientFrame *";
   out << GetName()<<" = new TGTransientFrame(gClient->GetRoot(),0"
       << "," << GetWidth() << "," << GetHeight() << "," << GetOptionString() <<");" << endl;

   if (!fList) return;

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");" << endl;
   }
   out << endl;

   // setting layout manager if it differs from the main frame type

   TGLayoutManager * lm = GetLayoutManager();

   if (GetOptions() & kHorizontalFrame) {
      if (lm->InheritsFrom(TGHorizontalLayout::Class())) { }
   } else if (GetOptions() & kVerticalFrame) {
      if (lm->InheritsFrom(TGVerticalLayout::Class())) { }
   } else {
      out << "   " << GetName() <<"->SetLayoutManager(";
      GetLayoutManager()->SavePrimitive(out, option);
      out << ");"<< endl;
   }

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
