// @(#)root/gui:$Name:$:$Id:$
// Author: Fons Rademakers   18/9/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGShutter, TGShutterItem                                             //
//                                                                      //
// A shutter widget contains a set of shutter items that can be         //
// open and closed ilike a shutter.                                     //
// This widget is usefull to group a large number of options in         //
// a number of categories.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGShutter.h"
#include "TGButton.h"
#include "TList.h"
#include "TTimer.h"


ClassImp(TGShutterItem)
ClassImp(TGShutter)

//______________________________________________________________________________
TGShutter::TGShutter(const TGWindow *p, UInt_t options) :
   TGCompositeFrame(p, 10, 10, options)
{
   // Create shutter frame.

   fSelectedItem        = 0;
   fClosingItem         = 0;
   fHeightIncrement     = 1;
   fClosingHeight       = 0;
   fClosingHadScrollbar = kFALSE;
   fTimer               = 0;
   fTrash               = new TList;
   fTrash->SetOwner();
}

//______________________________________________________________________________
TGShutter::~TGShutter()
{
   // Cleanup shutter widget.

   if (fTimer) delete fTimer;
   delete fTrash;
}

//______________________________________________________________________________
void TGShutter::AddItem(TGShutterItem *item)
{
   // Add shutter item to shutter frame.

   TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   AddFrame(item, hints);
   fTrash->Add(hints);
}

//______________________________________________________________________________
Bool_t TGShutter::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Handle shutter messages.

   if (!fList) return kFALSE;

   TGFrameElement *el;
   TGShutterItem  *child, *item = 0;

   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      child = (TGShutterItem *) el->fFrame;
      if (parm1 == child->WidgetId()) {
         item = child;
         break;
      }
   }

   if (!item) return kFALSE;

   if (!fSelectedItem)
      fSelectedItem = (TGShutterItem*) ((TGFrameElement*)fList->First())->fFrame;
   if (fSelectedItem == item) return kTRUE;

   fHeightIncrement = 1;
   fClosingItem = fSelectedItem;
   fClosingHeight = fClosingItem->GetHeight();
   fClosingHeight -= fClosingItem->fButton->GetDefaultHeight();
   fSelectedItem = item;
   if (!fTimer) fTimer = new TTimer(this, 6); //10);
   fTimer->Reset();
   fTimer->TurnOn();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGShutter::HandleTimer(TTimer *t)
{
   // Shutter item animation.

   if (!fClosingItem) return kFALSE;
   fClosingHeight -= fHeightIncrement;
   fHeightIncrement += 5;
   if (fClosingHeight > 0) {
      fTimer->Reset();
   } else {
      fClosingItem   = 0;
      fClosingHeight = 0;
      fTimer->TurnOff();
   }
   Layout();

   return kTRUE;
}

//______________________________________________________________________________
void TGShutter::Layout()
{
   // Layout shutter items.

   TGFrameElement *el;
   TGShutterItem  *child;
   Int_t y, bh, exh;

   if (!fList) return;

   if (!fSelectedItem)
      fSelectedItem = (TGShutterItem*) ((TGFrameElement*)fList->First())->fFrame;

   exh = Int_t(fHeight - (fBorderWidth << 1));
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      child = (TGShutterItem *) el->fFrame;
      bh = child->fButton->GetDefaultHeight();
      exh -= bh;
   }

   y = fBorderWidth;
   next.Reset();
   while ((el = (TGFrameElement *) next())) {
      child = (TGShutterItem *) el->fFrame;
      bh = child->fButton->GetDefaultHeight();
      if (child == fSelectedItem) {
         if (fClosingItem)
            child->fCanvas->SetScrolling(TGCanvas::kCanvasNoScroll);
         else
            child->fCanvas->SetScrolling(TGCanvas::kCanvasScrollVertical);
         child->ShowFrame(child->fCanvas);
         child->MoveResize(fBorderWidth, y, fWidth - (fBorderWidth << 1),
                           exh - fClosingHeight + bh);
         y += exh - fClosingHeight + bh;
      } else if (child == fClosingItem) {
         child->fCanvas->SetScrolling(TGCanvas::kCanvasNoScroll);
         child->MoveResize(fBorderWidth, y, fWidth - (fBorderWidth << 1),
                           fClosingHeight + bh);
         y += fClosingHeight + bh;
      } else {
         child->MoveResize(fBorderWidth, y, fWidth - (fBorderWidth << 1), bh);
         child->HideFrame(child->fCanvas);
         y += bh;
      }
   }
}

//______________________________________________________________________________
TGShutterItem::TGShutterItem(const TGWindow *p, TGHotString *s, Int_t id,
                             UInt_t options) :
   TGVerticalFrame (p, 10, 10, options), TGWidget (id)
{
   // Create a shutter item.

   fButton = new TGTextButton(this, s, id);
   fCanvas = new TGCanvas(this, 10, 10, kChildFrame);
   fContainer = new TGVerticalFrame(fCanvas->GetViewPort(), 10, 10, kOwnBackground);
   fCanvas->SetContainer(fContainer);
   fContainer->SetBackgroundColor(fClient->GetShadow(fgDefaultFrameBackground));

   AddFrame(fButton, fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   AddFrame(fCanvas, fL2 = new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));

   fButton->Associate((TGFrame *) p);
}

//______________________________________________________________________________
TGShutterItem::~TGShutterItem()
{
   // Clan up shutter item.

   delete fL1;
   delete fL2;
   delete fButton;
   delete fContainer;
   delete fCanvas;
}
