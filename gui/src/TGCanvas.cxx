// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   11/01/98

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
// TGCanvas and TGViewPort                                              //
//                                                                      //
// A TGCanvas is a frame containing two scrollbars (a horizontal and    //
// a vertical) and a viewport. The viewport acts as the window through  //
// which we look at the contents of the container frame.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGCanvas.h"
#include "TGScrollBar.h"
#include "TGWidget.h"
#include "TMath.h"


ClassImp(TGCanvas)
ClassImp(TGViewPort)

//______________________________________________________________________________
TGCanvas::TGCanvas(const TGWindow *p, UInt_t w, UInt_t h,
                   UInt_t options, ULong_t back) :
    TGFrame(p, w, h, options, back)
{
   // Create a canvas object.

   fVport      = new TGViewPort(this, w-4, h-4, kChildFrame, fgWhitePixel);
   fHScrollbar = new TGHScrollBar(this, w-4, kDefaultScrollBarWidth);
   fVScrollbar = new TGVScrollBar(this, kDefaultScrollBarWidth, h-4);

   fHScrollbar->Associate(this);
   fVScrollbar->Associate(this);
}

//______________________________________________________________________________
TGCanvas::~TGCanvas()
{
   // Delete canvas.

   delete fHScrollbar;
   delete fVScrollbar;
   delete fVport;
}

//______________________________________________________________________________
void TGCanvas::MapSubwindows()
{
   // Map all canvas sub windows.

   TGWindow::MapSubwindows();
   fHScrollbar->MapSubwindows();
   fVScrollbar->MapSubwindows();
   fVport->MapSubwindows();
}

//______________________________________________________________________________
void TGCanvas::AddFrame(TGFrame *f, TGLayoutHints *l)
{
   // Adding a frame to a canvas is actually adding the frame to the
   // viewport container. The viewport container is at least a
   // TGCompositeFrame.

   TGCompositeFrame *container = (TGCompositeFrame *) fVport->GetContainer();
   if (!container) {
      Error("AddFrame", "no canvas container set yet");
      return;
   }
   if (container->InheritsFrom(TGCompositeFrame::Class()))
       container->AddFrame(f, l);
   else
       Error("AddFrame", "canvas container must inherit from TGCompositeFrame");
}

//______________________________________________________________________________
void TGCanvas::DrawBorder()
{
   // Draw canvas border.

   switch (fOptions & (kSunkenFrame | kRaisedFrame | kDoubleBorder)) {
      case kSunkenFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, fgShadowGC, 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, fgShadowGC, 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, fgBlackGC, 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, fgBlackGC, 1, 1, 1, fHeight-3);

         gVirtualX->DrawLine(fId, fgHilightGC, 0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, fgHilightGC, fWidth-1, fHeight-1, fWidth-1, 0);
         gVirtualX->DrawLine(fId, fgBckgndGC,  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, fgBckgndGC,  fWidth-2, 1, fWidth-2, fHeight-2);
         break;

      default:
         TGFrame::DrawBorder();
         break;
   }
}

//______________________________________________________________________________
void TGCanvas::Layout()
{
   // Create layout for canvas. Depending on the size of the container
   // we need to add the scrollbars.

   Bool_t   need_vsb, need_hsb;
   UInt_t   cw, ch, tcw, tch;

   need_vsb = need_hsb = kFALSE;

   TGFrame *container = fVport->GetContainer();
   if (!container) {
      Error("Layout", "no canvas container set yet");
      return;
   }

   Bool_t fixedw = (container->GetOptions() & kFixedWidth) ? kTRUE : kFALSE;
   Bool_t fixedh = (container->GetOptions() & kFixedHeight) ? kTRUE : kFALSE;

   // test whether we need scrollbars

   cw = fWidth  - (fBorderWidth << 1);
   ch = fHeight - (fBorderWidth << 1);

   if (!fixedw) container->SetWidth(cw);
   if (!fixedh) container->SetHeight(ch);

   if (container->GetDefaultWidth() > cw) {
      need_hsb = kTRUE;
      ch -= fHScrollbar->GetDefaultHeight();
      if ((Int_t) ch < 0) {
         //Warning("Layout", "height would become too small, setting to 10");
         ch = 10;
      }
      if (!fixedh) container->SetHeight(ch);
   }

   if (container->GetDefaultHeight() > ch) {
      need_vsb = kTRUE;
      cw -= fVScrollbar->GetDefaultWidth();
      if ((Int_t) cw < 0) {
         //Warning("Layout", "width would become too small, setting to 10");
         cw = 10;
      }
      if (!fixedw) container->SetWidth(cw);
   }

   // re-check again (putting the vertical scrollbar could have changed things)

   if (container->GetDefaultWidth() > cw) {
      if (!need_hsb) {
         need_hsb = kTRUE;
         ch -= fHScrollbar->GetDefaultHeight();
         if ((Int_t) ch < 0) {
            //Warning("Layout", "height would become too small, setting to 10");
            ch = 10;
         }
         if (!fixedh) container->SetHeight(ch);
      }
   }

   if (need_hsb) {
      fHScrollbar->MoveResize(fBorderWidth, ch+fBorderWidth, cw, fHScrollbar->GetDefaultHeight());
      fHScrollbar->MapWindow();
   } else {
      fHScrollbar->UnmapWindow();
      fHScrollbar->SetPosition(0);
   }

   if (need_vsb) {
      fVScrollbar->MoveResize(cw+fBorderWidth, fBorderWidth, fVScrollbar->GetDefaultWidth(), ch);
      fVScrollbar->MapWindow();
   } else {
      fVScrollbar->UnmapWindow();
      fVScrollbar->SetPosition(0);
   }

   fVport->MoveResize(fBorderWidth, fBorderWidth, cw, ch);

   tcw = TMath::Max(container->GetDefaultWidth(), cw);
   tch = TMath::Max(container->GetDefaultHeight(), ch);
   UInt_t curw = container->GetDefaultWidth();
   container->SetWidth(0); // force a resize in TGFrame::Resize
   if (fixedw && fixedh)
      container->Resize(curw, container->GetDefaultHeight());
   else if (fixedw)
      container->Resize(curw, tch);
   else if (fixedh)
      container->Resize(tcw, container->GetDefaultHeight());
   else
      container->Resize(tcw, tch);

   if (need_hsb)
      fHScrollbar->SetRange(container->GetWidth(), fVport->GetWidth());
   if (need_vsb)
      fVScrollbar->SetRange(container->GetHeight(), fVport->GetHeight());
}

//______________________________________________________________________________
Bool_t TGCanvas::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle message generated by the canvas scrollbars.

   switch (GET_MSG(msg)) {
      case kC_HSCROLL:
         switch (GET_SUBMSG(msg)) {
            case kSB_SLIDERTRACK:
            case kSB_SLIDERPOS:
               fVport->SetHPos((Int_t)-parm1);
               break;
         }
         break;

      case kC_VSCROLL:
         switch (GET_SUBMSG(msg)) {
            case kSB_SLIDERTRACK:
            case kSB_SLIDERPOS:
               fVport->SetVPos((Int_t)-parm1);
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}


//______________________________________________________________________________
TGViewPort::TGViewPort(const TGWindow *p, UInt_t w, UInt_t h,
                       UInt_t options, ULong_t back) :
    TGCompositeFrame(p, w, h, options, back)
{
   // Create a viewport object.

   fContainer = 0;
   fX0 = fY0  = 0;
   MapSubwindows();
}

//______________________________________________________________________________
void TGViewPort::SetContainer(TGFrame *f)
{
   // Add container frame to the viewport. We must make sure that the added
   // container is at least a TGCompositeFrame (TGCanvas::AddFrame depends
   // on it).

   if (!fContainer) {
      fContainer = f;
      AddFrame(f, 0);
   }
}
