// @(#)root/gui:$Name:  $:$Id: TGCanvas.cxx,v 1.5 2002/06/12 16:46:11 rdm Exp $
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
// TGCanvas and TGViewPort and TGContainer                              //
//                                                                      //
// A TGCanvas is a frame containing two scrollbars (a horizontal and    //
// a vertical) and a viewport. The viewport acts as the window through  //
// which we look at the contents of the container frame.                //
//                                                                      //
// A TGContainer frame manages a content area. It can display and       //
// control a hierarchy of multi-column items, and provides the ability  //
// to add new items at any time. By default it doesn't map subwindows   //
// which are itmes of the container. In this case subwindow must        //
// provide DrawCopy method, see for example TGLVEntry class.            //
// It is also possible to use option which allow to map subwindows.     //
// This option has much slower drawing speed in case of more than 1000  //
// items placed in container. To activate this option the fMapSubwindows//
// data member must be set to kTRUE (for example TTVLVContainer class)  //
//                                                                      //
//   The TGContainer class can handle the keys:                         //
//                                                                      //
//    o  F7, Ctnrl-F - activate search dialog                           //
//    o  F3, Ctnrl-G - continue search                                  //
//    o  End - go to the last item in container                         //
//    o  Home - go to the first item in container                       //
//    o  PageUp,PageDown,arrow keys - navigate inside container         //
//    o  Return/Enter - equivalent to double click of the mouse button  //
//    o  Contrl-A - select/activate all items.                          //
//    o  Space - invert selection.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGCanvas.h"
#include "TGListView.h"
#include "TGScrollBar.h"
#include "TGWidget.h"
#include "TTimer.h"
#include "TMath.h"
#include "KeySymbols.h"
#include "TSystem.h"
#include "TGTextEditDialogs.h"
#include "TGMsgBox.h"

ClassImp(TGCanvas)
ClassImp(TGViewPort)
ClassImp(TGContainer)

const Int_t gAutoScrollFudge = 10;
const Int_t gAcceleration[gAutoScrollFudge+1] = {1,1,1,2,3,4,6,7,8,16,32};
const Int_t gKeyboardTime = 700;

//______________________________________________________________________________
class TGContainerKeyboardTimer : public TTimer {

private:
   TGContainer   *fContainer;
public:
   TGContainerKeyboardTimer(TGContainer *t) : TTimer(gKeyboardTime) { fContainer = t; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TGContainerKeyboardTimer::Notify()
{
   // single shot timer

   fContainer->SearchPattern();
   Reset();
   if (gSystem) gSystem->RemoveTimer(this);
   return kFALSE;
}

//______________________________________________________________________________
class TGContainerScrollTimer : public TTimer {

private:
   TGContainer   *fContainer;
public:
   TGContainerScrollTimer(TGContainer *t) : TTimer(50) { fContainer = t; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TGContainerScrollTimer::Notify()
{
   // on-timeout

   fContainer->OnAutoScroll();
   Reset();
   return kFALSE;
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
   AddInput(kStructureNotifyMask);
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
      if (fContainer->InheritsFrom(TGContainer::Class())) {
         ((TGContainer*)fContainer)->fViewPort = this;
         ((TGContainer*)fContainer)->fCanvas = (TGCanvas*)this->GetParent();
      }
   }
}

//______________________________________________________________________________
void TGViewPort::SetHPos(Int_t xpos)
{
   // moves content of container frame in horizontal direction

   Int_t diff;

   if (!fContainer) return;

   if (!fContainer->InheritsFrom(TGContainer::Class())) {
      fContainer->Move(fX0 = xpos, fY0);
      return;
   } else {
      if (((TGContainer*)fContainer)->fMapSubwindows) {
         fContainer->Move(fX0 = xpos, fY0);
         return;
      }
   }

   if (-xpos<0) diff = 0;
   else diff = xpos - fX0;

   if (!diff) return;

   fX0 = xpos;
    
   gVirtualX->ClearArea(fContainer->GetId(),0,0,fWidth,fHeight);
   ((TGContainer*)fContainer)->DrawRegion(0,0,fWidth,fHeight);
}

//______________________________________________________________________________
void TGViewPort::SetVPos(Int_t ypos)
{
   //  moves content of container frame in vertical direction

   Int_t diff;

   if (!fContainer) return;

   // for backward comatibility
   if (!fContainer->InheritsFrom(TGContainer::Class())) {
      fContainer->Move(fX0, fY0 = ypos);
      return;
   } else {
     if (((TGContainer*)fContainer)->fMapSubwindows) {
         fContainer->Move(fX0, fY0 = ypos);
         return;
      }
   }

   //
   if (-ypos<0) diff = 0;
   else diff = ypos - fY0;

   if (!diff) return;

   fY0 = ypos;

   gVirtualX->ClearArea(fContainer->GetId(),0,0,fWidth,fHeight);
   ((TGContainer*)fContainer)->DrawRegion(0,0,fWidth,fHeight);
}

//______________________________________________________________________________
void TGViewPort::SetPos(Int_t xpos, Int_t ypos)
{
   // goto new position

   if (!fContainer) return;

   SetHPos(fX0 = xpos);
   SetVPos(fY0 = ypos);
}

//______________________________________________________________________________
Bool_t TGViewPort::HandleConfigureNotify(Event_t *event)
{
   // handle resize events.

   if (!fContainer->InheritsFrom(TGContainer::Class())) {
      TGFrame::HandleConfigureNotify(event);
      return kTRUE;
   }

   TGContainer *cont = (TGContainer*)fContainer;
   cont->DrawRegion(0, 0, fWidth, fHeight);

   return kTRUE;
}

//______________________________________________________________________________
TGContainer::TGContainer(const TGWindow *p, UInt_t w, UInt_t h,
                             UInt_t options, ULong_t back) :
   TGCompositeFrame(p, w, h, options, back)
{
   // Create a canvas container. This is the (large) frame that contains
   // all the list items. It will be shown through a TGViewPort (which is
   // created by the TGCanvas).

   fMsgWindow  = p;
   fDragging   = kFALSE;
   fTotal = fSelected = 0;
   fMapSubwindows = kFALSE;
   fOnMouseOver = kFALSE;
   fLastActiveEl = 0;
   fLastDir = kTRUE;
   fLastCase = kTRUE;
   fKeyTimer = new TGContainerKeyboardTimer(this);
   fScrollTimer = new TGContainerScrollTimer(this);
   fKeyTimerActive = kFALSE;
   fScrolling = kFALSE;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                        kButtonPressMask | kButtonReleaseMask |
                        kPointerMotionMask, kNone, kNone);

   AddInput(kKeyPressMask | kPointerMotionMask);
}

//______________________________________________________________________________
TGContainer::~TGContainer()
{
   // Delete canvas container.

   if (fScrollTimer) delete fScrollTimer;
   if (fKeyTimer) delete fKeyTimer;
}

//______________________________________________________________________________
void TGContainer::Layout()
{
   // layout container entries

   ClearViewPort();
   TGCompositeFrame::Layout();
}

//______________________________________________________________________________
void TGContainer::CurrentChanged(Int_t x,Int_t y)
{
   // Emit signal when current position changed.

   long args[2];

   args[0] = x;
   args[1] = y;

   Emit("CurrentChanged(Int_t,Int_t)",args);
}

//______________________________________________________________________________
void TGContainer::CurrentChanged(TGFrame* f)
{
   // Emit signal when current selected frame changed.

   Emit("CurrentChanged(TGFrame*)",(long)f);
}

//______________________________________________________________________________
void TGContainer::ReturnPressed(TGFrame* f)
{
   // Signal emitted when Return/Enter key pressed
   //
   // It's equivalent to "double click" of mouse button

   Emit("ReturnPressed(TGFrame*)",(long)f);
}

//______________________________________________________________________________
void TGContainer::SpacePressed(TGFrame* f)
{
   // Signal emitted when space key pressed.
   //
   // Pressing space key inverts selection.

   Emit("SpacePressed(TGFrame*)",(long)f);
}

//______________________________________________________________________________
void TGContainer::OnMouseOver(TGFrame* f)
{
   // Signal emitted when pointer is over entry.
   //

   if (!fOnMouseOver) Emit("OnMouseOver(TGFrame*)",(long)f);
   fOnMouseOver = kTRUE;
}

//______________________________________________________________________________
void TGContainer::Clicked(TGFrame *entry, Int_t btn)
{
   // Emit Clicked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = btn;

   Emit("Clicked(TGFrame*,Int_t)", args);
}

//______________________________________________________________________________
void TGContainer::Clicked(TGFrame *entry, Int_t btn, Int_t x, Int_t y)
{
   // Emit Clicked() signal.

   Long_t args[4];

   args[0] = (Long_t)entry;
   args[1] = btn;
   args[2] = x;
   args[3] = y;

   Emit("Clicked(TGFrame*,Int_t,Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGContainer::DoubleClicked(TGFrame *entry, Int_t btn)
{
   // Emit DoubleClicked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = btn;

   Emit("DoubleClicked(TGFrame*,Int_t)", args);
}

//______________________________________________________________________________
void TGContainer::DoubleClicked(TGFrame *entry, Int_t btn, Int_t x, Int_t y)
{
   // Emit DoubleClicked() signal.

   Long_t args[4];

   args[0] = (Long_t)entry;
   args[1] = btn;
   args[2] = x;
   args[3] = y;

   Emit("DoubleClicked(TGFrame*,Int_t,Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGContainer::SelectAll()
{
   // Select all items in the container.
   // SelectAll signal emitted

   TIter next(fList);
   TGFrameElement* el;

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->Activate(kTRUE);
   }
   fSelected = fTotal;
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                  fTotal, fSelected);

   Emit("SelectAll()");
}

//______________________________________________________________________________
void TGContainer::UnSelectAll()
{
   //  Unselect all items in the container.

   TIter next(fList);
   TGFrameElement* el;

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->Activate(kFALSE);
   }
   fLastActiveEl = 0;
   fSelected = 0;
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                  fTotal, fSelected);
   Emit("UnSelectAll()");
}

//______________________________________________________________________________
void TGContainer::InvertSelection()
{
   // Invert the selection, all selected items become unselected and
   // vice versa.

   int selected = 0;

   TIter next(fList);
   TGFrameElement* el;

   while ((el = (TGFrameElement *) next())) {
      if (!el->fFrame->IsActive()) {
         el->fFrame->Activate(kTRUE);
         ++selected;
      } else {
         el->fFrame->Activate(kFALSE);
      }
   fSelected = selected;
   }
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                  fTotal, fSelected);
   Emit("InvertSelection()");
}

//______________________________________________________________________________
void TGContainer::RemoveAll()
{
   // Remove all items from the container.

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      el->fFrame->DestroyWindow();
      delete el->fFrame;
   }
   fList->Clear();
   fLastActiveEl = 0;
   fSelected = fTotal = 0;
}

//______________________________________________________________________________
void TGContainer::RemoveItem(TGFrame *item)
{
   // Remove item from container.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      if (item == el->fFrame) {
         if (item == fLastActiveEl->fFrame) fLastActiveEl = 0;
         item->DestroyWindow();
         delete item;
         fList->Remove(el);
         delete el;
         break;
      }
   }
}

//______________________________________________________________________________
const TGFrame *TGContainer::GetNextSelected(void **current)
{
   // Return next selected item.

   TGFrame *f;
   TObjLink *lnk = (TObjLink *) *current;

   lnk = (lnk == 0) ? fList->FirstLink() : lnk->Next();
   while (lnk) {
      f = (TGFrame *) ((TGFrameElement *) lnk->GetObject())->fFrame;
      if (f->IsActive()) {
         *current = (void *) lnk;
         return f;
      }
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
void TGContainer::ActivateItem(TGFrameElement* el)
{
   // Activate item.

   fLastActiveEl = el;
   el->fFrame->Activate(kTRUE);

   if (fLastActiveEl!=el) {
      CurrentChanged(fLastActiveEl->fFrame->GetX(),fLastActiveEl->fFrame->GetY());
      CurrentChanged(fLastActiveEl->fFrame);
      fSelected++;
   }

   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),fTotal, fSelected);
}

//______________________________________________________________________________
TGPosition TGContainer::GetPagePosition() const
{
   // Returns page position.

   TGPosition ret;

   ret.fX = -fViewPort->GetHPos();
   ret.fY = -fViewPort->GetVPos();

   return ret;
}

//______________________________________________________________________________
TGDimension TGContainer::GetPageDimension() const
{
   // Returns page dimension.

   TGDimension ret;

   ret.fWidth = fViewPort->GetWidth();
   ret.fHeight = fViewPort->GetHeight();
   return ret;
}

//______________________________________________________________________________
void TGContainer::SetPagePosition(const TGPosition& pos)
{
   // Set page position.

   fViewPort->SetPos(pos.fX,pos.fY);
}

//______________________________________________________________________________
void TGContainer::SetPagePosition(Int_t x, Int_t y)
{
   // Set page position.

  fViewPort->SetPos(x,y);
}


//______________________________________________________________________________
void TGContainer::SetPageDimension(const TGDimension& dim)
{
   // Set page dimension.

   fViewPort->Resize(dim);
}

//______________________________________________________________________________
void TGContainer::SetPageDimension(UInt_t w, UInt_t h)
{
   // Set page dimension.

   fViewPort->Resize(w,h);
}

//______________________________________________________________________________
void TGContainer::MapSubwindows()
{
   // Map subwindows.

   if (!fMapSubwindows) return;
   else TGCompositeFrame::MapSubwindows();
}

//______________________________________________________________________________
void TGContainer::DoRedraw()
{
   // Redraw content of container in the viewport region.

   DrawRegion(0,0,fViewPort->GetWidth(),fViewPort->GetHeight());
}

//______________________________________________________________________________
void TGContainer::DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a region of container in viewport.

   TGFrameElement *el;
   Handle_t id = fId;

   TGPosition pos = GetPagePosition();

   Int_t xx = pos.fX + x; // translate coordinates to current page position
   Int_t yy = pos.fY + y;

   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      if ( ( Int_t(el->fFrame->GetY())> yy-(Int_t)el->fFrame->GetHeight()) &&
          ( Int_t(el->fFrame->GetX())> xx-(Int_t)el->fFrame->GetWidth()) &&
          ( Int_t(el->fFrame->GetY())< yy+Int_t(h+el->fFrame->GetHeight())) &&
          ( Int_t(el->fFrame->GetX())< xx+Int_t(w+el->fFrame->GetWidth()))) {

            // draw either in container window or in double-buffer
            if (!fMapSubwindows) el->fFrame->DrawCopy(id,el->fFrame->GetX()-pos.fX,el->fFrame->GetY()-pos.fY);
            else fClient->NeedRedraw(el->fFrame);
      }
   }
}

//______________________________________________________________________________
void TGContainer::ClearViewPort()
{
   // Clear view port.

   gVirtualX->ClearArea(fId,0,0,fViewPort->GetWidth(),fViewPort->GetHeight());
}

//______________________________________________________________________________
Bool_t TGContainer::HandleExpose(Event_t *event)
{
   // Handle expose events. Do not use double buffer.

   TGFrameElement *el;

   TGPosition pos = GetPagePosition();

   Int_t xx = pos.fX + event->fX; // translate coordinates
   Int_t yy = pos.fY + event->fY;

   TIter next(fList);

   if (event->fWindow == GetId()) {
      while ((el = (TGFrameElement *) next())) {
         if ( ( Int_t(el->fFrame->GetY())> yy-(Int_t)el->fFrame->GetHeight()) &&
            ( Int_t(el->fFrame->GetX())> xx-(Int_t)el->fFrame->GetWidth()) &&
            ( Int_t(el->fFrame->GetY())< yy+Int_t(event->fHeight+el->fFrame->GetHeight())) &&
            ( Int_t(el->fFrame->GetX())< xx+Int_t(event->fWidth+el->fFrame->GetWidth())) ) {

            el->fFrame->DrawCopy(fId,el->fFrame->GetX()-pos.fX,el->fFrame->GetY()-pos.fY); //
         }
      }
   } else
      TGCompositeFrame::HandleExpose(event);

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TGContainer::HandleButton(Event_t *event)
{
   // Handle mouse button event in container.

   Int_t total, selected, page = 0;

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   if (event->fCode == kButton4 || event->fCode == kButton5) {
      page = dim.fHeight;
   }

   if (event->fCode == kButton4) {
      //scroll up
      Int_t newpos = pos.fY - page;
      fCanvas->SetVsbPosition(newpos);
      return kTRUE;
   }
   if (event->fCode == kButton5) {
      // scroll down
      Int_t newpos = fCanvas->GetVsbPosition() + page;
      fCanvas->SetVsbPosition(newpos);
      return kTRUE;
   }

   Int_t xx = pos.fX + event->fX; // translate coordinates
   Int_t yy = pos.fY + event->fY;

   if (event->fType == kButtonPress) {

      fXp = pos.fX + event->fX;
      fYp = pos.fY + event->fY;

      UnSelectAll();
      total = selected = 0;

      TGFrameElement *el;
      TIter next(fList);
      Bool_t select_frame = kFALSE;

      while ((el = (TGFrameElement *) next())) {
         select_frame = kFALSE;

         if (!fMapSubwindows) {
            if ( ( Int_t(el->fFrame->GetY()) + (Int_t)el->fFrame->GetHeight() > yy ) &&
               ( Int_t(el->fFrame->GetX()) + (Int_t)el->fFrame->GetWidth() > xx ) &&
               ( Int_t(el->fFrame->GetY()) < yy ) &&
               ( Int_t(el->fFrame->GetX()) < xx ) )  {
               select_frame = kTRUE;
            }
         } else {
            if (el->fFrame->GetId() == (Window_t)event->fUser[0]) {
               select_frame = kTRUE;
            }
         }

         if (select_frame) {
            selected++;
            ActivateItem(el);
            Clicked(el->fFrame, event->fCode);
            Clicked(el->fFrame, event->fCode, event->fXRoot, event->fYRoot);
         }
         total++;
      }

      if (fTotal != total || fSelected != selected) {
         fTotal = total;
         fSelected = selected;
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                     fTotal, fSelected);
      }

      if ( selected == 0 ) {
         fDragging = kTRUE;
         fX0 = fXf = fXp;
         fY0 = fYf = fYp;
         if (fMapSubwindows) gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);
      }
   }

   if (event->fType == kButtonRelease) {
      gVirtualX->SetInputFocus(fId);

      if (fDragging) {
         fDragging = kFALSE;
         fScrolling = kFALSE;

         if (gSystem) gSystem->RemoveTimer(fScrollTimer);
         if (fMapSubwindows) gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);
      } else {
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);
      }
   }
   fClient->NeedRedraw(this);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGContainer::HandleDoubleClick(Event_t *event)
{
   // Handle double click mouse event.

   TGFrameElement *el;
   TIter next(fList);

   TGPosition pos = GetPagePosition();

   Int_t xx = pos.fX + event->fX; // translate coordinates
   Int_t yy = pos.fY + event->fY;

   Bool_t select_frame = kFALSE;

   while ((el = (TGFrameElement *) next())) {
      select_frame = kFALSE;

      if (!fMapSubwindows) {
         if ( ( Int_t(el->fFrame->GetY()) + (Int_t)el->fFrame->GetHeight() > yy ) &&
            ( Int_t(el->fFrame->GetX()) + (Int_t)el->fFrame->GetWidth() > xx ) &&
            ( Int_t(el->fFrame->GetY()) < yy ) &&
            ( Int_t(el->fFrame->GetX()) < xx ) )  {
            select_frame = kTRUE;
         }
      } else {
         if (el->fFrame->GetId() == (Window_t)event->fUser[0]) {
            select_frame = kTRUE;
         }
      }

      if (select_frame) {
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMDBLCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);

         DoubleClicked(el->fFrame, event->fCode);
         DoubleClicked(el->fFrame, event->fCode, event->fXRoot, event->fYRoot);
         return kTRUE;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGContainer::HandleMotion(Event_t *event)
{
   // Handle mouse motion events.

   int xf0, yf0, xff, yff, total, selected;

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();
   int x = pos.fX + event->fX;
   int y = pos.fY + event->fY;
   TGFrameElement* el = 0;
   TGFrame* f = 0;

   if (fDragging) {
      if (fMapSubwindows) gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);

      fX0 =  TMath::Min(fXp,x);
      fY0 =  TMath::Min(fYp,y);
      fXf =  TMath::Max(fXp,x);
      fYf =  TMath::Max(fYp,y);

      total = selected = 0;

      if (event->fX>Int_t(dim.fWidth)-gAutoScrollFudge) {
         fCanvas->SetHsbPosition(x - dim.fWidth);
         fScrolling = kTRUE;
      } else if (event->fX<gAutoScrollFudge) {
         fCanvas->SetHsbPosition(x);
         fScrolling = kTRUE;
      } else if (event->fY>Int_t(dim.fHeight)-gAutoScrollFudge) {
         fCanvas->SetVsbPosition(y - dim.fHeight);
         fScrolling = kTRUE;
      } else if (event->fY<gAutoScrollFudge) {
         fCanvas->SetVsbPosition(y);
         fScrolling = kTRUE;
      }

      TIter next(fList);

      while ((el = (TGFrameElement *) next())) {
         f = el->fFrame;
         ++total;
         xf0 = f->GetX() + (f->GetWidth() >> 3);
         yf0 = f->GetY() + (f->GetHeight() >> 3);
         xff = xf0 + f->GetWidth() - (f->GetWidth() >> 2);
         yff = yf0 + f->GetHeight() - (f->GetHeight() >> 2);

         if (((xf0 > fX0 && xf0 < fXf) ||
              (xff > fX0 && xff < fXf)) &&
             ((yf0 > fY0 && yf0 < fYf) ||
              (yff > fY0 && yff < fYf))) {
            f->Activate(kTRUE);
#ifndef WIN32
            gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));
#endif
            OnMouseOver(f);
            ++selected;
         } else {
            f->Activate(kFALSE);
         }
      }

      if (fMapSubwindows) gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);

      if (fTotal != total || fSelected != selected) {
         fTotal = total;
         fSelected = selected;
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                     fTotal, fSelected);
      }
      fClient->NeedRedraw(this);
   } else {
      TIter next(fList);

      fOnMouseOver = kFALSE;

      while ((el = (TGFrameElement *) next())) {
         if ( ( Int_t(el->fFrame->GetY()) + (Int_t)el->fFrame->GetHeight() > y ) &&
            ( Int_t(el->fFrame->GetX()) + (Int_t)el->fFrame->GetWidth() > x ) &&
            ( Int_t(el->fFrame->GetY()) < y ) &&
            ( Int_t(el->fFrame->GetX()) < x ) )  {
#ifndef WIN32
               gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));
#endif
               OnMouseOver(f);
            }
         }
#ifndef WIN32
         gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kPointer));
#endif       
   }

   if (fScrolling) {
      if (gSystem) {
         fScrollTimer->Reset();
         gSystem->AddTimer(fScrollTimer);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGContainer::HandleKey(Event_t *event)
{
   // The key press event handler converts a key press to some line editor
   // action.

   char   input[10];
   Int_t  n;
   UInt_t keysym;

   if (event->fType == kGKeyPress) {
      gVirtualX->LookupString(event, input, sizeof(input), keysym);
      n = strlen(input);

      switch ((EKeySym)keysym) {
         case kKey_Enter:
         case kKey_Return:
            // treat 'Enter' and 'Return' as a double click
            SendMessage(GetMessageWindow(), MK_MSG(kC_CONTAINER, kCT_ITEMDBLCLICK),
                              kButton1, (event->fYRoot << 16) | event->fXRoot);
            ReturnPressed(fLastActiveEl->fFrame);
            break;
         case kKey_Shift:
         case kKey_Control:
         case kKey_Meta:
         case kKey_Alt:
         case kKey_CapsLock:
         case kKey_NumLock:
         case kKey_ScrollLock:
            return kTRUE;
         case kKey_Space:
            fLastActiveEl->fFrame->Activate(!fLastActiveEl->fFrame->IsActive());
            SpacePressed(fLastActiveEl->fFrame);
            break;
         default:
         break;
      }

      if (event->fState & kKeyControlMask) {   // Cntrl key modifier pressed
         switch((EKeySym)keysym & ~0x20) {   // treat upper and lower the same
            case kKey_A:
               SelectAll();
               break;
            case kKey_B:
               LineLeft();
               break;
            case kKey_C:
               return kTRUE;
            case kKey_D:
               break;
            case kKey_E:
               End();
               break;
            case kKey_F:
               Search();
               break;
            case kKey_G:
               RepeatSearch();
               break;
            case kKey_H:
               LineLeft();
               break;
            case kKey_K:
               End();
               break;
            case kKey_U:
               Home();
               break;
            case kKey_V:
            case kKey_Y:
               return kTRUE;
            case kKey_X:
               return kTRUE;
            default:
               return kTRUE;
         }
      }
      if (n && keysym >= 32 && keysym < 127 &&     // printable keys
          !(event->fState & kKeyControlMask) &&
          (EKeySym)keysym != kKey_Delete &&
          (EKeySym)keysym != kKey_Backspace) {

         if (fKeyTimerActive) {
            fKeyInput += input;
         } else {
            fKeyInput = input;
            fKeyTimerActive = kTRUE;
            fKeyTimer->Reset();
            if (gSystem) gSystem->AddTimer(fKeyTimer);
         }
      } else {

         switch ((EKeySym)keysym) {
            case kKey_F3:
               RepeatSearch();
               break;
            case kKey_F5:
               Layout();
               break;
            case kKey_F7:
               Search();
               break;
            case kKey_Left:
               LineLeft(event->fState & kKeyShiftMask);
               break;
            case kKey_Right:
               LineRight(event->fState & kKeyShiftMask);
               break;
            case kKey_Up:
               LineUp(event->fState & kKeyShiftMask);
               break;
            case kKey_Down:
               LineDown(event->fState & kKeyShiftMask);
               break;
            case kKey_PageUp:
               PageUp(event->fState & kKeyShiftMask);
               break;
            case kKey_PageDown:
               PageDown(event->fState & kKeyShiftMask);
               break;
            case kKey_Home:
               Home(event->fState & kKeyShiftMask);
               break;
            case kKey_End:
               End(event->fState & kKeyShiftMask);
               break;
            default:
               break;
         }
      }
   }
   fClient->NeedRedraw(this);
   return kTRUE;
}

//______________________________________________________________________________
void TGContainer::Search()
{
   // invokes search dialog. Looks for item with the entered name

   Int_t ret = 0;
   char msg[256];
   TGFrameElement* fe = 0;
   fSearch = new TGSearchType;

   new TGSearchDialog(fClient->GetRoot(), fCanvas, 400, 150, fSearch, &ret);

   if (ret) {
         fe = FindFrame(fSearch->fBuffer, fSearch->fDirection,fSearch->fCaseSensitive);

      if (!fe) {  // find again
         if (fLastActiveEl) fLastActiveEl->fFrame->Activate(kFALSE);
         fLastActiveEl = 0;
         fe = FindFrame(fLastName.Data(),fLastDir,fLastCase);

         if (!fe) {
            sprintf(msg, "Couldn't find \"%s\"", fLastName.Data());
            gVirtualX->Bell(50);
            new TGMsgBox(fClient->GetRoot(), fCanvas, "Container", msg,
                                       kMBIconExclamation, kMBOk, 0);
         } else {
            if (fLastActiveEl) fLastActiveEl->fFrame->Activate(kFALSE);
            ActivateItem(fe);
            AdjustPosition();
         }
      } else {
         if (fLastActiveEl) fLastActiveEl->fFrame->Activate(kFALSE);
         ActivateItem(fe);
         AdjustPosition();
      }
   }
   delete fSearch;
   fSearch = 0;
}

//______________________________________________________________________________
void TGContainer::OnAutoScroll()
{
   // Autoscroll while close to & beyond  The Wall

   TGFrameElement* el = 0;
   TGFrame* f = 0;
   int xf0, yf0, xff, yff, total, selected;

   TGDimension dim = GetPageDimension();
   TGPosition pos = GetPagePosition();

   Window_t  dum1, dum2;
   Event_t   ev;
   ev.fType    = kButtonPress;
   Int_t x,y;

   // Autoscroll while close to the wall
   Int_t dx = 0;
   Int_t dy = 0;

   // Where's the cursor?
   gVirtualX->QueryPointer(fId,dum1,dum2,ev.fXRoot,ev.fYRoot,x,y,ev.fState);

   // Figure scroll amount x
   if (x<gAutoScrollFudge) dx = gAutoScrollFudge - x;
   else if ((Int_t)dim.fWidth-gAutoScrollFudge<=x) dx = dim.fWidth - gAutoScrollFudge - x;

   // Figure scroll amount y
   if (y<gAutoScrollFudge) dy = gAutoScrollFudge - y;
   else if ((Int_t)dim.fHeight-gAutoScrollFudge<=y) dy = dim.fHeight - gAutoScrollFudge - y;

   if (dx || dy) {
      Int_t adx = TMath::Abs(dx);
      Int_t ady = TMath::Abs(dy);
      if (adx>gAutoScrollFudge) adx = gAutoScrollFudge;
      if (ady>gAutoScrollFudge) ady = gAutoScrollFudge;

      dx *= gAcceleration[adx];
      dy *= gAcceleration[ady];

      fCanvas->SetHsbPosition(pos.fX-dx);
      fCanvas->SetVsbPosition(pos.fY-dy);

      // position inside container
      x += pos.fX;
      y += pos.fY;

      fX0 =  TMath::Min(fXp,x);
      fY0 =  TMath::Min(fYp,y);
      fXf =  TMath::Max(fXp,x);
      fYf =  TMath::Max(fYp,y);

      total = selected = 0;

      TIter next(fList);

      while ((el = (TGFrameElement *) next())) {
         f = el->fFrame;
         ++total;
         xf0 = f->GetX() + (f->GetWidth() >> 3);
         yf0 = f->GetY() + (f->GetHeight() >> 3);
         xff = xf0 + f->GetWidth() - (f->GetWidth() >> 2);
         yff = yf0 + f->GetHeight() - (f->GetHeight() >> 2);

         if (((xf0 > fX0 && xf0 < fXf) ||
            (xff > fX0 && xff < fXf)) &&
            ((yf0 > fY0 && yf0 < fYf) ||
            (yff > fY0 && yff < fYf))) {
            f->Activate(kTRUE);
            ++selected;
         } else {
            f->Activate(kFALSE);
         }
      }

      if (fMapSubwindows) gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);

      if (fTotal != total || fSelected != selected) {
         fTotal = total;
         fSelected = selected;
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                     fTotal, fSelected);
      }
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
void TGContainer::SearchPattern()
{
   // search for entry which name begins with pattern

   TGFrameElement* fe = 0;
   TIter next(fList);
   TGLVEntry* lv;
   TObject* obj;
   TString str;

   while ((fe=( TGFrameElement*)next())) {
      if (!fe->fFrame->InheritsFrom(TGLVEntry::Class())) continue;

      lv = (TGLVEntry*)fe->fFrame;
      obj = (TObject*)lv->GetUserData();
      str = obj->GetName();
      if (str.BeginsWith(fKeyInput,TString::kIgnoreCase)) {
         if (fLastActiveEl && (fLastActiveEl!=fe) )
            fLastActiveEl->fFrame->Activate(kFALSE);
         ActivateItem(fe);
         AdjustPosition();
         break;
      }
   }

   fKeyInput = "";   //clear
   fKeyTimerActive = kFALSE;
}

//______________________________________________________________________________
void TGContainer::RepeatSearch()
{
   // repeats search

   char msg[256];
   TGFrameElement* fe = 0;

   fe = FindFrame(fLastName.Data(),fLastDir,fLastCase);

   if (!fe) {
      if (fLastActiveEl) fLastActiveEl->fFrame->Activate(kFALSE);
      fLastActiveEl = 0;
      fe = FindFrame(fLastName.Data(),fLastDir,fLastCase);

      if (!fe) {
         sprintf(msg, "Couldn't find \"%s\"", fLastName.Data());
         gVirtualX->Bell(50);
         new TGMsgBox(fClient->GetRoot(), fCanvas, "Container", msg,
                        kMBIconExclamation, kMBOk, 0);
      } else {
         if (fLastActiveEl) fLastActiveEl->fFrame->Activate(kFALSE);
         ActivateItem(fe);
         AdjustPosition();
      }
   } else {
      if (fLastActiveEl) fLastActiveEl->fFrame->Activate(kFALSE);
      ActivateItem(fe);
      AdjustPosition();
   }
}

//______________________________________________________________________________
TGFrameElement* TGContainer::FindFrame(Int_t x,Int_t y,Bool_t exclude)
{
   // find frame located int container at position x,y

   TIter next(fList);
   TGFrameElement* el;
   TGFrameElement* ret = 0;
   Int_t dx = 0;
   Int_t dy = 0;
   Int_t d = 0;
   Int_t dd;

   el = (TGFrameElement *) next();
   if (!el) return 0;

   dx = TMath::Abs(el->fFrame->GetX()-x);
   dy = TMath::Abs(el->fFrame->GetY()-y);
   d = dx + dy;

   while ((el = (TGFrameElement *) next())) {
      if (exclude && (el==fLastActiveEl) ) continue;
      dx = TMath::Abs(el->fFrame->GetX()-x);
      dy = TMath::Abs(el->fFrame->GetY()-y);
      dd = dx+dy;

      if (dd<d) {
         d = dd;
         ret = el;
      }
   }
   return ret;
}

//______________________________________________________________________________
TGFrameElement* TGContainer::FindFrame(const TString& name, Bool_t direction,
                                       Bool_t caseSensitive,Bool_t beginWith)
{

   // Find a frame which assosiated object has a name containing a "name" string.

   if (name.IsNull()) return 0;
   int idx = kNPOS;

   TGFrameElement* el = 0;
   TString str;
   TString::ECaseCompare cmp = caseSensitive ? TString::kExact : TString::kIgnoreCase;
   
   fLastDir = direction;
   fLastCase = caseSensitive;
   fLastName = name;

   if (fLastActiveEl) {
      el = fLastActiveEl;

      if (direction) {
         el = (TGFrameElement *)fList->After(el);
      } else {
         el = (TGFrameElement *)fList->Before(el);
      }
   } else {
      if (direction) el = (TGFrameElement *)fList->First();
      else el  = (TGFrameElement *)fList->Last();
   }

   TGLVEntry* lv = 0;
   TObject* obj = 0; 

   while (el) {
      if (!el->fFrame->InheritsFrom(TGLVEntry::Class())) continue;
      
      lv = (TGLVEntry*)el->fFrame;
      obj = (TObject*)lv->GetUserData();
      str = obj->GetName();
      
      idx = str.Index(name,0,cmp);

      if (idx!=kNPOS) {
         if (beginWith) {
            if (idx==0) return el;
         } else { 
            return el;
         }
      }

      if (direction) {
         el = (TGFrameElement *)fList->After(el);
      } else {
         el = (TGFrameElement *)fList->Before(el);
      }
   }
   return 0;
}

//______________________________________________________________________________
void TGContainer::AdjustPosition()
{
   // Move content to position of highlighted/activated frame

   if (!fLastActiveEl) return;
   TGFrame* f = fLastActiveEl->fFrame;

   Int_t vh = 0;
   Int_t v = 0;

   if (fCanvas->GetVScrollbar()->IsMapped()) {
      vh = fCanvas->GetVScrollbar()->GetPosition()+(Int_t)fViewPort->GetHeight();

      if (f->GetY()<fCanvas->GetVScrollbar()->GetPosition()) {
         v = TMath::Max(0,f->GetY()-(Int_t)fViewPort->GetHeight()/2);
         fCanvas->SetVsbPosition(v);
      } else if (f->GetY()+(Int_t)f->GetHeight()>vh) {
         v = TMath::Min((Int_t)GetHeight()-(Int_t)fViewPort->GetHeight(),
                        f->GetY()+(Int_t)f->GetHeight()-(Int_t)fViewPort->GetHeight()/2);
         fCanvas->SetVsbPosition(v);
      }
   }

   Int_t hw = 0;
   Int_t h = 0;

   if (fCanvas->GetHScrollbar()->IsMapped()) {
      hw = fCanvas->GetHScrollbar()->GetPosition()+(Int_t)fViewPort->GetWidth();

      if (f->GetX()<fCanvas->GetHScrollbar()->GetPosition()) {
         h = TMath::Max(0,f->GetX()-(Int_t)fViewPort->GetWidth()/2);
         fCanvas->SetHsbPosition(h);
      } else if (f->GetX()+(Int_t)f->GetWidth()>hw) {
         h = TMath::Min((Int_t)GetWidth()-(Int_t)fViewPort->GetWidth(),
                        f->GetX()+(Int_t)f->GetWidth()-(Int_t)fViewPort->GetWidth()/2);
         fCanvas->SetHsbPosition(h);
      }
   }
}

//______________________________________________________________________________
void TGContainer::LineLeft(Bool_t select)
{
   // Move current position one column left.

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   TGFrameElement* fe = (TGFrameElement*)fList->First();
   if (!fe) return; // empty list

   TGFrameElement* old = fLastActiveEl;

   if (old) old->fFrame->Activate(kFALSE);   //
   else fLastActiveEl = fe;

   TGFrameElement* la = fLastActiveEl;
   Int_t y = la->fFrame->GetY();
   Int_t x = la->fFrame->GetX() - la->fFrame->GetWidth() - la->fLayout->GetPadLeft();

   Int_t hw = pos.fX + dim.fWidth;

   if (x<=0 && !fCanvas->GetHScrollbar()->IsMapped()) { // move to previous line
      x = hw;
      y = y - la->fFrame->GetHeight() - la->fLayout->GetPadTop();
   }

   fe = FindFrame(x,y);
   if (!fe) fe = (TGFrameElement*)fList->First();

   if (!select) fSelected=1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::LineRight(Bool_t select)
{
   // Move current position one column right.

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   TGFrameElement* fe = (TGFrameElement*)fList->Last();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;

   if (old) old->fFrame->Activate(kFALSE);
   else fLastActiveEl = (TGFrameElement*)fList->First();

   Int_t y = fLastActiveEl->fFrame->GetY();
   Int_t x = fLastActiveEl->fFrame->GetX()+
             fLastActiveEl->fFrame->GetWidth()+
             fLastActiveEl->fLayout->GetPadRight();

   Int_t hw = pos.fX + dim.fWidth -
             fLastActiveEl->fFrame->GetWidth()-
             fLastActiveEl->fLayout->GetPadRight();

   if (x>=hw && !fCanvas->GetHScrollbar()->IsMapped()) { // move one line down
      x = 0;
      y = y + fLastActiveEl->fFrame->GetHeight() + fLastActiveEl->fLayout->GetPadBottom();
   }

   fe = FindFrame(x,y);
   if (!fe) fe = (TGFrameElement*)fList->Last();
   if (!select) fSelected=1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::LineUp(Bool_t select)
{
   // Make current position first line in window by scrolling up.

   TGFrameElement* fe = (TGFrameElement*)fList->First();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;

   if (old) old->fFrame->Activate(kFALSE);
   else fLastActiveEl = (TGFrameElement*)fList->First();

   Int_t y = fLastActiveEl->fFrame->GetY()-
             fLastActiveEl->fFrame->GetHeight()-
             fLastActiveEl->fLayout->GetPadTop();
   Int_t x = fLastActiveEl->fFrame->GetX();

   fe = FindFrame(x,y);
   if (!fe) fe = (TGFrameElement*)fList->First();
   if (fe->fFrame->GetY()>fLastActiveEl->fFrame->GetY()) fe = fLastActiveEl;
   if (!select) fSelected=1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::LineDown(Bool_t select)
{
   // Move one line down.

   TGFrameElement* fe = (TGFrameElement*)fList->Last();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;

   if (old) old->fFrame->Activate(kFALSE);
   else fLastActiveEl = (TGFrameElement*)fList->First();

   Int_t y = fLastActiveEl->fFrame->GetY()+
             fLastActiveEl->fFrame->GetHeight()+
             fLastActiveEl->fLayout->GetPadBottom();
   Int_t x = fLastActiveEl->fFrame->GetX();

   fe = FindFrame(x,y);
   if (!fe) fe = (TGFrameElement*)fList->Last();
   if (fe->fFrame->GetY()<fLastActiveEl->fFrame->GetY()) fe = fLastActiveEl;
   if (!select) fSelected=1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::PageUp(Bool_t select)
{
   // Move  position one page up.

   TGDimension dim = GetPageDimension();

   TGFrameElement* fe = (TGFrameElement*)fList->First();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;

   if (old) old->fFrame->Activate(kFALSE);
   else fLastActiveEl = (TGFrameElement*)fList->First();

   Int_t y = fLastActiveEl->fFrame->GetY();
   Int_t x = fLastActiveEl->fFrame->GetX();

   if (fCanvas->GetVScrollbar()->IsMapped()) {
      y -= dim.fHeight;
   } else {
      if (fCanvas->GetHScrollbar()->IsMapped()) x -= dim.fWidth;
      else {
         Home();
         return;
      }
   }

   fe = FindFrame(x,y);

   if (!fe || fe->fFrame->GetY()>fLastActiveEl->fFrame->GetY())
      fe = (TGFrameElement*)fList->First();

   if (!select) fSelected = 1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::PageDown(Bool_t select)
{
   // Move position one page down.

   TGDimension dim = GetPageDimension();

   TList* li = GetList();
   TGFrameElement* fe = (TGFrameElement*)fList->Last();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;

   if (old) old->fFrame->Activate(kFALSE);
   else fLastActiveEl = (TGFrameElement*)fList->First();

   Int_t y = fLastActiveEl->fFrame->GetY();
   Int_t x = fLastActiveEl->fFrame->GetX();

   if (fCanvas->GetVScrollbar()->IsMapped()) {
      y +=  dim.fHeight;
   } else {
      if (fCanvas->GetHScrollbar()->IsMapped()) x += dim.fWidth;
      else {
         End();
         return;
      }
   }

   fe = FindFrame(x,y);
   if (!fe || fe->fFrame->GetY()<fLastActiveEl->fFrame->GetY() )
      fe = (TGFrameElement*)li->Last();

   if (!select) fSelected = 1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::Home(Bool_t select)
{
   // Move to upper-left corner of container.

   TGFrameElement* fe = (TGFrameElement*)fList->First();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;
   if (old) old->fFrame->Activate(kFALSE);

   if (!select) fSelected = 1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
void TGContainer::End(Bool_t select)
{
   // Move to the bottom-right corner of container.

   TGFrameElement* fe = (TGFrameElement*)fList->Last();
   if (!fe) return;

   TGFrameElement* old = fLastActiveEl;
   if (old) old->fFrame->Activate(kFALSE);

   if (!select) fSelected = 1;

   ActivateItem(fe);
   AdjustPosition();
}

//______________________________________________________________________________
TGCanvas::TGCanvas(const TGWindow *p, UInt_t w, UInt_t h,
                   UInt_t options, ULong_t back) :
    TGFrame(p, w, h, options, back)
{
   // Create a canvas object.

   fVport      = new TGViewPort(this, w-4, h-4, kChildFrame | kOwnBackground,
                                fgWhitePixel);
   fHScrollbar = new TGHScrollBar(this, w-4, kDefaultScrollBarWidth);
   fVScrollbar = new TGVScrollBar(this, kDefaultScrollBarWidth, h-4);

   fScrolling  = kCanvasScrollBoth;

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
   // viewport container. The viewport container must be at least a
   // TGCompositeFrame for this method to succeed.

   TGFrame *container = fVport->GetContainer();
   if (!container) {
      Error("AddFrame", "no canvas container set yet");
      return;
   }
   if (container->InheritsFrom(TGCompositeFrame::Class()))
      ((TGCompositeFrame*)container)->AddFrame(f, l);
   else
      Error("AddFrame", "canvas container must inherit from TGCompositeFrame");
}

//______________________________________________________________________________
void TGCanvas::DrawBorder()
{
   // Draw canvas border.

   switch (fOptions & (kSunkenFrame | kRaisedFrame | kDoubleBorder)) {
      case kSunkenFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, fgShadowGC(), 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, fgShadowGC(), 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, fgBlackGC(), 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, fgBlackGC(), 1, 1, 1, fHeight-3);

         gVirtualX->DrawLine(fId, fgHilightGC(), 0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, fgHilightGC(), fWidth-1, fHeight-1, fWidth-1, 0);
         gVirtualX->DrawLine(fId, fgBckgndGC(),  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, fgBckgndGC(),  fWidth-2, 1, fWidth-2, fHeight-2);
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
      if (fScrolling & kCanvasScrollHorizontal) {
         need_hsb = kTRUE;
         ch -= fHScrollbar->GetDefaultHeight();
         if ((Int_t) ch < 0) {
            //Warning("Layout", "height would become too small, setting to 10");
            ch = 10;
         }
         if (!fixedh) container->SetHeight(ch);
      }
   }

   if (container->GetDefaultHeight() > ch) {
      if (fScrolling & kCanvasScrollVertical) {
         need_vsb = kTRUE;
         cw -= fVScrollbar->GetDefaultWidth();
         if ((Int_t) cw < 0) {
            //Warning("Layout", "width would become too small, setting to 10");
            cw = 10;
         }
         if (!fixedw) container->SetWidth(cw);
      }
   }

   // re-check again (putting the vertical scrollbar could have changed things)

   if (container->GetDefaultWidth() > cw) {
      if (!need_hsb) {
         if (fScrolling & kCanvasScrollHorizontal) {
            need_hsb = kTRUE;
            ch -= fHScrollbar->GetDefaultHeight();
            if ((Int_t) ch < 0) {
               //Warning("Layout", "height would become too small, setting to 10");
               ch = 10;
            }
            if (!fixedh) container->SetHeight(ch);
         }
      }
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

   if (need_hsb) {
      fHScrollbar->MoveResize(fBorderWidth, ch+fBorderWidth, cw, fHScrollbar->GetDefaultHeight());
      fHScrollbar->SetRange(container->GetWidth(), fVport->GetWidth());
      fHScrollbar->MapWindow();
   } else {
      fHScrollbar->UnmapWindow();
      fHScrollbar->SetPosition(0);
   }

   if (need_vsb) {
      fVScrollbar->MoveResize(cw+fBorderWidth, fBorderWidth, fVScrollbar->GetDefaultWidth(), ch);
      fVScrollbar->SetRange(container->GetHeight(), fVport->GetHeight());
      fVScrollbar->MapWindow();
   } else {
      fVScrollbar->UnmapWindow();
      fVScrollbar->SetPosition(0);
   }
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
Int_t TGCanvas::GetHsbPosition() const
{
   // Get position of horizontal scrollbar.

   if (fHScrollbar && fHScrollbar->IsMapped())
     return fHScrollbar->GetPosition();
   return 0;
}

//______________________________________________________________________________
Int_t TGCanvas::GetVsbPosition() const
{
   // Get position of vertical scrollbar.

   if (fVScrollbar && fVScrollbar->IsMapped())
     return fVScrollbar->GetPosition();
   return 0;
}

//______________________________________________________________________________
void TGCanvas::SetHsbPosition(Int_t newPos)
{
   // Set position of horizontal scrollbar.

   if (fHScrollbar && fHScrollbar->IsMapped())
     fHScrollbar->SetPosition(newPos);
   else
     fVport->SetHPos(0);
}

//______________________________________________________________________________
void TGCanvas::SetVsbPosition(Int_t newPos)
{
   // Set position of vertical scrollbar.

   if (fVScrollbar && fVScrollbar->IsMapped())
      fVScrollbar->SetPosition(newPos);
   else
      fVport->SetVPos(0);
}

//______________________________________________________________________________
void TGCanvas::SetScrolling(Int_t scrolling)
{
   // Set scrolling policy. Use values defined by the enum: kCanvasNoScroll,
   // kCanvasScrollHorizontal, kCanvasScrollVertical, kCanvasScrollBoth.

   if (scrolling != fScrolling) {
      fScrolling = scrolling;
      Layout();
   }
}
