// @(#)root/gui:$Id$
// Author: Fons Rademakers   25/02/98

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
// TGListTree and TGListTreeItem                                        //
//                                                                      //
// A list tree is a widget that can contain a number of items           //
// arranged in a tree structure. The items are represented by small     //
// folder icons that can be either open or closed.                      //
//                                                                      //
// The TGListTree is user callable. The TGListTreeItem is a service     //
// class of the list tree.                                              //
//                                                                      //
// A list tree can generate the following events:                       //
// kC_LISTTREE, kCT_ITEMCLICK, which button, location (y<<16|x).        //
// kC_LISTTREE, kCT_ITEMDBLCLICK, which button, location (y<<16|x).     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "TROOT.h"
#include "TGListTree.h"
#include "TGPicture.h"
#include "TGCanvas.h"
#include "TGScrollBar.h"
#include "TGToolTip.h"
#include "KeySymbols.h"
#include "TGTextEditDialogs.h"
#include "TGResourcePool.h"
#include "TGMsgBox.h"
#include "TError.h"
#include "TColor.h"
#include "TSystem.h"
#include "TString.h"
#include "TObjString.h"
#include "TGDNDManager.h"
#include "TBufferFile.h"
#include "Riostream.h"
#include "RConfigure.h"

Pixel_t          TGListTree::fgGrayPixel = 0;
const TGFont    *TGListTree::fgDefaultFont = 0;
TGGC            *TGListTree::fgActiveGC = 0;
TGGC            *TGListTree::fgDrawGC = 0;
TGGC            *TGListTree::fgLineGC = 0;
TGGC            *TGListTree::fgHighlightGC = 0;
TGGC            *TGListTree::fgColorGC = 0;
const TGPicture *TGListTree::fgOpenPic = 0;
const TGPicture *TGListTree::fgClosedPic = 0;
const TGPicture *TGListTree::fgCheckedPic = 0;
const TGPicture *TGListTree::fgUncheckedPic = 0;


ClassImp(TGListTreeItem)
ClassImp(TGListTreeItemStd)
ClassImp(TGListTree)

/******************************************************************************/
/******************************************************************************/
// TGListTreeItem
/******************************************************************************/

//______________________________________________________________________________
TGListTreeItem::TGListTreeItem(TGClient *client) :
   fClient(client),
   fParent    (0), fFirstchild(0), fLastchild (0), fPrevsibling(0), 
   fNextsibling(0),fOpen (kFALSE), fDNDState  (0),
   fY         (0), fXtext     (0), fYtext(0), fHeight(0)
{
   // Constructor.
}

//______________________________________________________________________________
UInt_t TGListTreeItem::GetPicWidth() const
{
   // Return width of item's icon.

   const TGPicture *pic = GetPicture();
   return (pic) ? pic->GetWidth() : 0;
}

/******************************************************************************/
/******************************************************************************/
// TGListTreeItemStd
/******************************************************************************/

//______________________________________________________________________________
TGListTreeItemStd::TGListTreeItemStd(TGClient *client, const char *name,
                                     const TGPicture *opened,
                                     const TGPicture *closed,
                                     Bool_t checkbox) :
   TGListTreeItem(client)
{
   // Create list tree item.

   fText = name;
   fCheckBox = checkbox;
   fChecked = kTRUE;

   if (!opened)
      opened = TGListTree::GetOpenPic();
   else
      ((TGPicture *)opened)->AddReference();

   if (!closed)
      closed = TGListTree::GetClosedPic();
   else
      ((TGPicture *)closed)->AddReference();

   fOpenPic   = opened;
   fClosedPic = closed;

   fCheckedPic   = TGListTree::GetCheckedPic();
   fUncheckedPic = TGListTree::GetUncheckedPic();

   fActive = kFALSE;

   fOwnsData = kFALSE;
   fUserData = 0;

   fHasColor = kFALSE;
   fColor = 0;
   fDNDState = 0;
}

//______________________________________________________________________________
TGListTreeItemStd::~TGListTreeItemStd()
{
   // Delete list tree item.

   if (fOwnsData && fUserData) {
      TObject *obj = static_cast<TObject *>(fUserData);
      delete dynamic_cast<TObject *>(obj);
   }
   fClient->FreePicture(fOpenPic);
   fClient->FreePicture(fClosedPic);
   fClient->FreePicture(fCheckedPic);
   fClient->FreePicture(fUncheckedPic);
}

//______________________________________________________________________________
Pixel_t TGListTreeItemStd::GetActiveColor() const
{
   // Return color for marking items that are active or selected.

   return TGFrame::GetDefaultSelectedBackground();
}

//______________________________________________________________________________
Bool_t TGListTreeItemStd::HasCheckedChild(Bool_t first)
{
   // Add all child items of 'item' into the list 'checked'.

   TGListTreeItem *item = this;

   while (item) {
      if (item->IsChecked()) {
         return kTRUE;
      }
      if (item->GetFirstChild()) {
         if (item->GetFirstChild()->HasCheckedChild())
            return kTRUE;
      }
      if (!first)
         item = item->GetNextSibling();
      else
         break;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGListTreeItemStd::HasUnCheckedChild(Bool_t first)
{
   // Add all child items of 'item' into the list 'checked'.

   TGListTreeItem *item = this;

   while (item) {
      if (!item->IsChecked()) {
         return kTRUE;
      }
      if (item->GetFirstChild()) {
         if (item->GetFirstChild()->HasUnCheckedChild())
            return kTRUE;
      }
      if (!first)
         item = item->GetNextSibling();
      else
         break;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGListTreeItemStd::UpdateState()
{
   // Update the state of the node 'item' according to the children states.

   if ((!fChecked && HasCheckedChild(kTRUE)) ||
       (fChecked && HasUnCheckedChild(kTRUE))) {
      SetCheckBoxPictures(gClient->GetPicture("checked_dis_t.xpm"),
                          gClient->GetPicture("unchecked_dis_t.xpm"));
   }
   else {
      SetCheckBoxPictures(gClient->GetPicture("checked_t.xpm"),
                          gClient->GetPicture("unchecked_t.xpm"));
   }
}

//______________________________________________________________________________
void TGListTreeItemStd::CheckAllChildren(Bool_t state)
{
   // Set all child items of this one checked if state=kTRUE,
   // unchecked if state=kFALSE.
   
   if (state) {
      if (!IsChecked())
         CheckItem();
   } else {
      if (IsChecked())
         Toggle();
   }
   CheckChildren(GetFirstChild(), state);
   UpdateState();   
}

//______________________________________________________________________________
void TGListTreeItemStd::CheckChildren(TGListTreeItem *item, Bool_t state)
{
   // Set all child items of 'item' checked if state=kTRUE;
   // unchecked if state=kFALSE.

   if (!item) return;

   while (item) {
      if (state){
         if (!item->IsChecked())
            item->CheckItem();
      } else {
         if (item->IsChecked())
            item->Toggle();
      }
      if (item->GetFirstChild()) {
         CheckChildren(item->GetFirstChild(), state);
      }
      item->UpdateState();   
      item = item->GetNextSibling();
   }
}

//______________________________________________________________________________
void TGListTreeItemStd::SetCheckBox(Bool_t on)
{
   // Set a check box on the tree node.

   fCheckBox = on;
}

//___________________________________________________________________________
void TGListTreeItemStd::SetCheckBoxPictures(const TGPicture *checked,
                                         const TGPicture *unchecked)
{
   // Change list tree check item icons.

   fClient->FreePicture(fCheckedPic);
   fClient->FreePicture(fUncheckedPic);

   if (!checked) {
         ::Warning("TGListTreeItem::SetCheckBoxPictures", "checked picture not specified, defaulting to checked_t");
         checked = fClient->GetPicture("checked_t.xpm");
   } else
      ((TGPicture *)checked)->AddReference();

   if (!unchecked) {
         ::Warning("TGListTreeItem::SetCheckBoxPictures", "unchecked picture not specified, defaulting to unchecked_t");
         unchecked = fClient->GetPicture("unchecked_t.xpm");
   } else
      ((TGPicture *)unchecked)->AddReference();

   fCheckedPic   = checked;
   fUncheckedPic = unchecked;
}

//___________________________________________________________________________
void TGListTreeItemStd::SetPictures(const TGPicture *opened, const TGPicture *closed)
{
   // Change list tree item icons.

   fClient->FreePicture(fOpenPic);
   fClient->FreePicture(fClosedPic);

   if (!opened) {
      ::Warning("TGListTreeItem::SetPictures", "opened picture not specified, defaulting to ofolder_t");
      opened = fClient->GetPicture("ofolder_t.xpm");
   } else
      ((TGPicture *)opened)->AddReference();

   if (!closed) {
      ::Warning("TGListTreeItem::SetPictures", "closed picture not specified, defaulting to folder_t");
      closed = fClient->GetPicture("folder_t.xpm");
   } else
      ((TGPicture *)closed)->AddReference();

   fOpenPic   = opened;
   fClosedPic = closed;
}


/******************************************************************************/
/******************************************************************************/
// TGListTree
/******************************************************************************/

//______________________________________________________________________________
TGListTree::TGListTree(TGWindow *p, UInt_t w, UInt_t h, UInt_t options,
                       ULong_t back) :
   TGContainer(p, w, h, options, back)
{
   // Create a list tree widget.

   fMsgWindow   = p;
   fCanvas      = 0;
   fTip         = 0;
   fTipItem     = 0;
   fAutoTips    = kFALSE;
   fAutoCheckBoxPic = kTRUE;
   fDisableOpen = kFALSE;
   fBdown       = kFALSE;
   fUserControlled = kFALSE;
   fEventHandled   = kFALSE;
   fExposeTop = fExposeBottom = 0;
   fDropItem = 0;
   fLastEventState = 0;

   fGrayPixel   = GetGrayPixel();
   fFont        = GetDefaultFontStruct();

   fActiveGC    = GetActiveGC()();
   fDrawGC      = GetDrawGC()();
   fLineGC      = GetLineGC()();
   fHighlightGC = GetHighlightGC()();
   fColorGC     = GetColorGC()();

   fFirst = fLast = fSelected = fCurrent = fBelowMouse = 0;
   fDefw = fDefh = 1;

   fHspacing = 2;
   fVspacing = 2;  // 0;
   fIndent   = 3;  // 0;
   fMargin   = 2;

   fXDND = fYDND = 0;
   fDNDData.fData = 0;
   fDNDData.fDataLength = 0;
   fDNDData.fDataType = 0;
   fBuf = 0;

   fColorMode = kDefault;
   fCheckMode = kSimple;
   if (fCanvas) fCanvas->GetVScrollbar()->SetSmallIncrement(20);

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone);

   AddInput(kPointerMotionMask | kEnterWindowMask |
            kLeaveWindowMask | kKeyPressMask);
   SetWindowName();

   fDNDTypeList = new Atom_t[3];
   fDNDTypeList[0] = gVirtualX->InternAtom("application/root", kFALSE);
   fDNDTypeList[1] = gVirtualX->InternAtom("text/uri-list", kFALSE);
   fDNDTypeList[2] = 0;
   gVirtualX->SetDNDAware(fId, fDNDTypeList);
   SetDNDTarget(kTRUE);
   fEditDisabled = kEditDisable | kEditDisableGrab | kEditDisableBtnEnable;
}

//______________________________________________________________________________
TGListTree::TGListTree(TGCanvas *p,UInt_t options,ULong_t back) :
   TGContainer(p, options, back)
{
   // Create a list tree widget.

   fMsgWindow   = p;
   fTip         = 0;
   fTipItem     = 0;
   fAutoTips    = kFALSE;
   fAutoCheckBoxPic = kTRUE;
   fDisableOpen = kFALSE;
   fBdown       = kFALSE;
   fUserControlled = kFALSE;
   fEventHandled   = kFALSE;
   fExposeTop = fExposeBottom = 0;
   fDropItem = 0;
   fLastEventState = 0;

   fGrayPixel   = GetGrayPixel();
   fFont        = GetDefaultFontStruct();

   fActiveGC    = GetActiveGC()();
   fDrawGC      = GetDrawGC()();
   fLineGC      = GetLineGC()();
   fHighlightGC = GetHighlightGC()();
   fColorGC     = GetColorGC()();

   fFirst = fLast = fSelected = fCurrent = fBelowMouse = 0;
   fDefw = fDefh = 1;

   fHspacing = 2;
   fVspacing = 2;  // 0;
   fIndent   = 3;  // 0;
   fMargin   = 2;

   fXDND = fYDND = 0;
   fDNDData.fData = 0;
   fDNDData.fDataLength = 0;
   fDNDData.fDataType = 0;
   fBuf = 0;

   fColorMode = kDefault;
   fCheckMode = kSimple;
   if (fCanvas) fCanvas->GetVScrollbar()->SetSmallIncrement(20);

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone);

   AddInput(kPointerMotionMask | kEnterWindowMask |
            kLeaveWindowMask | kKeyPressMask);
   SetWindowName();

   fDNDTypeList = new Atom_t[3];
   fDNDTypeList[0] = gVirtualX->InternAtom("application/root", kFALSE);
   fDNDTypeList[1] = gVirtualX->InternAtom("text/uri-list", kFALSE);
   fDNDTypeList[2] = 0;
   gVirtualX->SetDNDAware(fId, fDNDTypeList);
   SetDNDTarget(kTRUE);
   fEditDisabled = kEditDisable | kEditDisableGrab | kEditDisableBtnEnable;
}

//______________________________________________________________________________
TGListTree::~TGListTree()
{
   // Delete list tree widget.

   TGListTreeItem *item, *sibling;

   delete [] fDNDTypeList;
   delete fTip;

   item = fFirst;
   while (item) {
      PDeleteChildren(item);
      sibling = item->fNextsibling;
      delete item;
      item = sibling;
   }
}

//--- text utility functions

//______________________________________________________________________________
Int_t TGListTree::FontHeight()
{
   // Returns height of currently used font.

   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetIconFont();
   return fgDefaultFont->TextHeight();
}

//______________________________________________________________________________
Int_t TGListTree::FontAscent()
{
   // Returns ascent of currently used font.

   FontMetrics_t m;
   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetIconFont();
   fgDefaultFont->GetFontMetrics(&m);
   return m.fAscent;
}

//______________________________________________________________________________
Int_t TGListTree::TextWidth(const char *c)
{
   // Returns text width relative to currently used font.

   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetIconFont();
   return fgDefaultFont->TextWidth(c);
}

//---- highlighting utilities

//______________________________________________________________________________
void TGListTree::HighlightItem(TGListTreeItem *item, Bool_t state, Bool_t draw)
{
   // Highlight tree item.

   if (item) {
      if ((item == fSelected) && !state) {
         fSelected = 0;
         if (draw) DrawItemName(fId, item);
      } else if (state != item->IsActive()) { // !!!! leave active alone ...
         item->SetActive(state);
         if (draw) DrawItemName(fId, item);
      }
   }
}

//______________________________________________________________________________
void TGListTree::HighlightChildren(TGListTreeItem *item, Bool_t state, Bool_t draw)
{
   // Higlight item children.

   while (item) {
      HighlightItem(item, state, draw);
      if (item->fFirstchild)
         HighlightChildren(item->fFirstchild, state, (item->IsOpen()) ? draw : kFALSE);
      item = item->fNextsibling;
   }
}

//______________________________________________________________________________
void TGListTree::UnselectAll(Bool_t draw)
{
   // Unselect all items.

   ClearViewPort();
   HighlightChildren(fFirst, kFALSE, draw);
}

//______________________________________________________________________________
Bool_t TGListTree::HandleButton(Event_t *event)
{
   // Handle button events in the list tree.

   TGListTreeItem *item;

   if (fTip) fTip->Hide();

   Int_t page = 0;
   if (event->fCode == kButton4 || event->fCode == kButton5) {
      if (!fCanvas) return kTRUE;
      if (fCanvas->GetContainer()->GetHeight()) {
//         page = Int_t(Float_t(fCanvas->GetViewPort()->GetHeight() *
//                              fCanvas->GetViewPort()->GetHeight()) /
//                              fCanvas->GetContainer()->GetHeight());
         // choose page size either 1/5 of viewport or 5 lines (90)
         Int_t r = fCanvas->GetViewPort()->GetHeight() / 5;
         page = TMath::Min(r, 90);
      }
   }

   if (event->fCode == kButton4) {
      //scroll up
      Int_t newpos = fCanvas->GetVsbPosition() - page;
      if (newpos < 0) newpos = 0;
      fCanvas->SetVsbPosition(newpos);
      return kTRUE;
   }
   if (event->fCode == kButton5) {
      // scroll down
      Int_t newpos = fCanvas->GetVsbPosition() + page;
      fCanvas->SetVsbPosition(newpos);
      return kTRUE;
   }

   if (event->fType == kButtonPress) {
      if ((item = FindItem(event->fY)) != 0) {
         if (event->fCode == kButton1) {
            Int_t minx, maxx;
            Int_t minxchk = 0, maxxchk = 0;
            if (item->HasCheckBox() && item->GetCheckBoxPicture()) {
               minxchk = (item->fXtext - item->GetCheckBoxPicture()->GetWidth());
               maxxchk = (item->fXtext - 4);
               maxx = maxxchk - Int_t(item->GetPicWidth()) - 8;
               minx = minxchk - Int_t(item->GetPicWidth()) - 16;
            }
            else {
               maxx = (item->fXtext - Int_t(item->GetPicWidth())) - 8;
               minx = (item->fXtext - Int_t(item->GetPicWidth())) - 16;
            }
            if ((item->HasCheckBox()) && (event->fX < maxxchk) &&
               (event->fX > minxchk))
            {
               ToggleItem(item);
               if (fCheckMode == kRecursive) {
                  CheckAllChildren(item, item->IsChecked());
               }
               UpdateChecked(item, kTRUE);
               Checked((TObject *)item->GetUserData(), item->IsChecked());
               return kTRUE;
            }
            if ((event->fX < maxx) && (event->fX > minx)) {
               item->SetOpen(!item->IsOpen());
               ClearViewPort();
               return kTRUE;
            }
         }
         // DND specific
         if (event->fCode == kButton1) {
            fXDND = event->fX;
            fYDND = event->fY;
            fBdown = kTRUE;
         }
         if (!fUserControlled) {
            if (fSelected) fSelected->SetActive(kFALSE);
            UnselectAll(kTRUE);
            //item->fActive = kTRUE; // this is done below w/redraw
            fCurrent = fSelected = item;
            HighlightItem(item, kTRUE, kTRUE);
            SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMCLICK),
                        event->fCode, (event->fYRoot << 16) | event->fXRoot);
         }
         else {
            fCurrent = fSelected = item;
            ClearViewPort();
         }
         Clicked(item, event->fCode);
         Clicked(item, event->fCode, event->fXRoot, event->fYRoot);
         Clicked(item, event->fCode, event->fState, event->fXRoot, event->fYRoot);
      }
   }
   if (event->fType == kButtonRelease) {
      fBdown = kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleDoubleClick(Event_t *event)
{
   // Handle double click event in the list tree (only for kButton1).

   TGListTreeItem *item = 0;

   if (event->fCode == kButton4 || event->fCode == kButton5) {
      return kFALSE;
   }
   // If fDisableOpen is set, only send message and emit signals.
   // It allows user to customize handling of double click events.
   if (fDisableOpen && event->fCode == kButton1 && (item = FindItem(event->fY)) != 0) {
      SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMDBLCLICK),
                  event->fCode, (event->fYRoot << 16) | event->fXRoot);
      DoubleClicked(item, event->fCode);
      DoubleClicked(item, event->fCode, event->fXRoot, event->fYRoot);
      return kTRUE;
   }
   item = FindItem(event->fY);

   // Otherwise, just use default behaviour (open item).
   if (event->fCode == kButton1 && item) {
      ClearViewPort();
      item->SetOpen(!item->IsOpen());
      if (!fUserControlled) {
         if (item != fSelected) { // huh?!
            if (fSelected) fSelected->SetActive(kFALSE); // !!!!
            UnselectAll(kTRUE);
            HighlightItem(item, kTRUE, kTRUE);
         }
      }
      SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMDBLCLICK),
                  event->fCode, (event->fYRoot << 16) | event->fXRoot);
      DoubleClicked(item, event->fCode);
      DoubleClicked(item, event->fCode, event->fXRoot, event->fYRoot);
   }
   if (!fUserControlled)
      fSelected = item;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (event->fType == kLeaveNotify) {
      if (fTip) {
         fTip->Hide();
         fTipItem = 0;
      }
      if (!fUserControlled) {
         if (fCurrent)
            DrawOutline(fId, fCurrent, 0xffffff, kTRUE);
         if (fBelowMouse)
            DrawOutline(fId, fBelowMouse, 0xffffff, kTRUE);
         fCurrent = 0;
      }
      if (fBelowMouse) {
         fBelowMouse = 0;
         MouseOver(0);
         MouseOver(0, event->fState);
      }
   }
   ClearViewPort();
   return kTRUE;
}

//______________________________________________________________________________
Atom_t TGListTree::HandleDNDPosition(Int_t /*x*/, Int_t y, Atom_t action,
                                      Int_t /*xroot*/, Int_t /*yroot*/)
{
   // Handle dragging position events.

   static TGListTreeItem *olditem = 0;
   TGListTreeItem *item;
   if ((item = FindItem(y)) != 0) {
      if (item->IsDNDTarget()) {
         fDropItem = item;
         if (olditem)
            HighlightItem(olditem, kFALSE, kTRUE);
         HighlightItem(item, kTRUE, kTRUE);
         olditem = item;
         return action;
      }
   }
   fDropItem = 0;
   if (olditem) {
      HighlightItem(olditem, kFALSE, kTRUE);
      olditem = 0;
   }
   return kNone;
}

//______________________________________________________________________________
Atom_t TGListTree::HandleDNDEnter(Atom_t *typelist)
{
   // Handle drag enter events.

   Atom_t ret = kNone;
   for (int i = 0; typelist[i] != kNone; ++i) {
      if (typelist[i] == fDNDTypeList[0])
         ret = fDNDTypeList[0];
      if (typelist[i] == fDNDTypeList[1])
         ret = fDNDTypeList[1];
   }
   return ret;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleDNDLeave()
{
   // Handle drag leave events.

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleDNDDrop(TDNDData *data)
{
   // Handle drop events.

   DataDropped(fDropItem, data);
   HighlightItem(fDropItem, kFALSE, kTRUE);
   //ClearHighlighted();
   return kTRUE;
}

//______________________________________________________________________________
void TGListTree::DataDropped(TGListTreeItem *item, TDNDData *data)
{
   // Emit DataDropped() signal.

   Long_t args[2];

   args[0] = (Long_t)item;
   args[1] = (Long_t)data;

   Emit("DataDropped(TGListTreeItem*,TDNDData*)", args);
}

//______________________________________________________________________________
Bool_t TGListTree::HandleMotion(Event_t *event)
{
   // Handle mouse motion event. Used to set tool tip, to emit
   // MouseOver() signal and for DND handling.

   TGListTreeItem *item;
   TGPosition pos = GetPagePosition();

   if (gDNDManager->IsDragging()) {
      gDNDManager->Drag(event->fXRoot, event->fYRoot,
                        TGDNDManager::GetDNDActionCopy(), event->fTime);
   } else if ((item = FindItem(event->fY)) != 0) {
      if (!fUserControlled) {
         if (fCurrent)
            DrawOutline(fId, fCurrent, 0xffffff, kTRUE);
         if (fBelowMouse)
            DrawOutline(fId, fBelowMouse, 0xffffff, kTRUE);
         DrawOutline(fId, item);
         fCurrent = item;
      }
      if (item != fBelowMouse) {
         fBelowMouse = item;
         MouseOver(fBelowMouse);
         MouseOver(fBelowMouse, event->fState);
      }

      if (item->HasCheckBox() && item->GetCheckBoxPicture()) {
         if ((event->fX < (item->fXtext - 4) &&
             (event->fX > (item->fXtext - (Int_t)item->GetCheckBoxPicture()->GetWidth()))))
         {
            gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kPointer));
            return kTRUE;
         }
         else {
            gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));
         }
      }
      if (!gDNDManager->IsDragging()) {
         if (fBdown && ((abs(event->fX - fXDND) > 2) || (abs(event->fY - fYDND) > 2))) {
            if (gDNDManager && item->IsDNDSource()) {
               if (!fBuf) fBuf = new TBufferFile(TBuffer::kWrite);
               fBuf->Reset();
               // !!!!! Here check virtual Bool_t HandlesDragAndDrop()
               // and let the item handle this.
               if (item->GetUserData()) {
                  TObject *obj = static_cast<TObject *>(item->GetUserData());
                  if (dynamic_cast<TObject *>(obj)) {
                     TObjString *ostr = dynamic_cast<TObjString *>(obj);
                     if (ostr) {
                        TString& str = ostr->String();
                        if (str.BeginsWith("file://")) {
                           fDNDData.fDataType = fDNDTypeList[1];
                           fDNDData.fData = (void *)strdup(str.Data());
                           fDNDData.fDataLength = str.Length()+1;
                        }
                     }
                     else {
                        fDNDData.fDataType = fDNDTypeList[0];
                        fBuf->WriteObject((TObject *)item->GetUserData());
                        fDNDData.fData = fBuf->Buffer();
                        fDNDData.fDataLength = fBuf->Length();
                     }
                  }
               }
               else {
                  fDNDData.fDataType = fDNDTypeList[1];
                  TString str = TString::Format("file://%s/%s\r\n",
                                gSystem->UnixPathName(gSystem->WorkingDirectory()),
                                item->GetText());
                  fDNDData.fData = (void *)strdup(str.Data());
                  fDNDData.fDataLength = str.Length()+1;
               }
               if (item->GetPicture()) {
                  TString xmpname = item->GetPicture()->GetName();
                  if (xmpname.EndsWith("_t.xpm"))
                     xmpname.ReplaceAll("_t.xpm", "_s.xpm");
                  if (xmpname.EndsWith("_t.xpm__16x16"))
                     xmpname.ReplaceAll("_t.xpm__16x16", "_s.xpm");
                  const TGPicture *pic = fClient->GetPicture(xmpname.Data());
                  if (!pic) pic = item->GetPicture();
                  if (pic) SetDragPixmap(pic);
               }
               gDNDManager->StartDrag(this, event->fXRoot, event->fYRoot);
            }
         }
      }
      if (gDNDManager->IsDragging()) {
         gDNDManager->Drag(event->fXRoot, event->fYRoot,
                           TGDNDManager::GetDNDActionCopy(), event->fTime);
      } else {
         if (fTipItem == item) return kTRUE;
         if (!fUserControlled) { // !!!! what is this? It was called above once?
            MouseOver(item);
            MouseOver(item, event->fState);
         }
         gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));
      }

      if (fTip)
         fTip->Hide();

      if (item->GetTipTextLength() > 0) {

         SetToolTipText(item->GetTipText(), item->fXtext,
                        item->fY - pos.fY + item->fHeight, 1000);

      } else if (fAutoTips && item->GetUserData()) {
         // must derive from TObject (in principle user can put pointer
         // to anything in user data field). Add check.
         TObject *obj = (TObject *)item->GetUserData();
         if (obj && obj->InheritsFrom(TObject::Class())) {
            SetToolTipText(obj->GetTitle(), item->fXtext,
                           item->fY - pos.fY + item->fHeight, 1000);
         }
      }
      fTipItem = item;
   } else {
      if (fBelowMouse) {
         fBelowMouse = 0;
         MouseOver(fBelowMouse);
         MouseOver(fBelowMouse, event->fState);
      }
      gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kPointer));
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleKey(Event_t *event)
{
   // The key press event handler converts a key press to some line editor
   // action.

   char   input[10];
   UInt_t keysym;
   TGListTreeItem *item = 0;

   fLastEventState = event->fState;
   if (fTip) fTip->Hide();

   if (event->fType == kGKeyPress) {
      gVirtualX->LookupString(event, input, sizeof(input), keysym);

      if (!event->fState && (EKeySym)keysym == kKey_Escape) {
         if (gDNDManager->IsDragging()) gDNDManager->EndDrag();
      }

      item = fCurrent;
      if (!item) return kFALSE;

      fEventHandled = kFALSE;
      KeyPressed(item, keysym, event->fState);

      if (fUserControlled && fEventHandled)
         return kTRUE;
      
      switch ((EKeySym)keysym) {
         case kKey_Enter:
         case kKey_Return:
            event->fType = kButtonPress;
            event->fCode = kButton1;

            if (fSelected == item) {
               // treat 'Enter' and 'Return' as a double click
               ClearViewPort();
               item->SetOpen(!item->IsOpen());
               DoubleClicked(item, 1);
            } else {
               // treat 'Enter' and 'Return' as a click
               if (fSelected) fSelected->SetActive(kFALSE);
               UnselectAll(kTRUE);
               ClearViewPort();
               fSelected = item;
               fSelected->SetActive(kTRUE);
               HighlightItem(item, kTRUE, kTRUE);
               Clicked(item, 1);
               Clicked(item, 1, event->fXRoot, event->fYRoot);
               Clicked(item, 1, event->fState, event->fXRoot, event->fYRoot);
            }
            break;
         case kKey_Space:
            if (item->HasCheckBox()) {
               ToggleItem(item);
               if (fCheckMode == kRecursive) {
                  CheckAllChildren(item, item->IsChecked());
               }
               UpdateChecked(item, kTRUE);
               Checked((TObject *)item->GetUserData(), item->IsChecked());
            }
            break;
         case kKey_F3:
            Search(kFALSE);
            break;
         case kKey_F5:
            Layout();
            break;
         case kKey_F7:
            Search();
            break;
         case kKey_Left:
            ClearViewPort();
            item->SetOpen(kFALSE);
            break;
         case kKey_Right:
            ClearViewPort();
            item->SetOpen(kTRUE);
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
      if (event->fState & kKeyControlMask) { // Ctrl key modifier pressed
         switch((EKeySym)keysym & ~0x20) {   // treat upper and lower the same
            case kKey_F:
               Search();
               break;
            case kKey_G:
               Search(kFALSE);
               break;
            default:
               return kTRUE;
         }
      }

   }
   return kTRUE;
}

//______________________________________________________________________________
void TGListTree::MouseOver(TGListTreeItem *entry)
{
   // Signal emitted when pointer is over entry.

   Emit("MouseOver(TGListTreeItem*)", (Long_t)entry);
}

//______________________________________________________________________________
void TGListTree::MouseOver(TGListTreeItem *entry, UInt_t mask)
{
   // Signal emitted when pointer is over entry.

   Long_t args[2];
   args[0] = (Long_t)entry;
   args[1] = mask;
   Emit("MouseOver(TGListTreeItem*,UInt_t)", args);
}

//______________________________________________________________________________
void TGListTree::KeyPressed(TGListTreeItem *entry, UInt_t keysym, UInt_t mask)
{
   // Signal emitted when keyboard key pressed
   //
   // entry - selected item
   // keysym - defined in "KeySymbols.h"
   // mask - modifier key mask, defined in "GuiTypes.h"
   //
   // const Mask_t kKeyShiftMask   = BIT(0);
   // const Mask_t kKeyLockMask    = BIT(1);
   // const Mask_t kKeyControlMask = BIT(2);
   // const Mask_t kKeyMod1Mask    = BIT(3);   // typically the Alt key
   // const Mask_t kButton1Mask    = BIT(8);
   // const Mask_t kButton2Mask    = BIT(9);
   // const Mask_t kButton3Mask    = BIT(10);
   // const Mask_t kButton4Mask    = BIT(11);
   // const Mask_t kButton5Mask    = BIT(12);
   // const Mask_t kAnyModifier    = BIT(15);

   Long_t args[3];
   args[0] = (Long_t)entry;
   args[1] = (Long_t)keysym;
   args[2] = (Long_t)mask;
   Emit("KeyPressed(TGListTreeItem*,ULong_t,ULong_t)", args);
   SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_KEY), keysym, mask);
}

//______________________________________________________________________________
void TGListTree::ReturnPressed(TGListTreeItem *entry)
{
   // Emit ReturnPressed() signal.

   Emit("ReturnPressed(TGListTreeItem*)", (Long_t)entry);
}

//______________________________________________________________________________
void TGListTree::Checked(TObject *entry, Bool_t on)
{
   // Emit Checked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = on;

   Emit("Checked(TObject*,Bool_t)", args);
}

//______________________________________________________________________________
void TGListTree::Clicked(TGListTreeItem *entry, Int_t btn)
{
   // Emit Clicked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = btn;

   Emit("Clicked(TGListTreeItem*,Int_t)", args);
}

//______________________________________________________________________________
void TGListTree::Clicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y)
{
   // Emit Clicked() signal.

   Long_t args[4];

   args[0] = (Long_t)entry;
   args[1] = btn;
   args[2] = x;
   args[3] = y;

   Emit("Clicked(TGListTreeItem*,Int_t,Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGListTree::Clicked(TGListTreeItem *entry, Int_t btn, UInt_t mask, Int_t x, Int_t y)
{
   // Emit Clicked() signal.

   Long_t args[5];

   args[0] = (Long_t)entry;
   args[1] = btn;
   args[2] = mask;
   args[3] = x;
   args[4] = y;

   Emit("Clicked(TGListTreeItem*,Int_t,UInt_t,Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGListTree::DoubleClicked(TGListTreeItem *entry, Int_t btn)
{
   // Emit DoubleClicked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = btn;

   Emit("DoubleClicked(TGListTreeItem*,Int_t)", args);
}

//______________________________________________________________________________
void TGListTree::DoubleClicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y)
{
   // Emit DoubleClicked() signal.

   Long_t args[4];

   args[0] = (Long_t)entry;
   args[1] = btn;
   args[2] = x;
   args[3] = y;

   Emit("DoubleClicked(TGListTreeItem*,Int_t,Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGListTree::Home(Bool_t /*select*/)
{
   // Move content to the top.

   if (fCanvas) fCanvas->SetVsbPosition(0);
}

//______________________________________________________________________________
void TGListTree::End(Bool_t /*select*/)
{
   // Move content to the bottom.

   if (fCanvas) fCanvas->SetVsbPosition((Int_t)fHeight);
}

//______________________________________________________________________________
void TGListTree::PageUp(Bool_t /*select*/)
{
   // Move content one page up.

   if (!fCanvas) return;

   TGDimension dim = GetPageDimension();

   Int_t newpos = fCanvas->GetVsbPosition() - dim.fHeight;
   if (newpos<0) newpos = 0;

   fCanvas->SetVsbPosition(newpos);
}

//______________________________________________________________________________
void TGListTree::PageDown(Bool_t /*select*/)
{
   // Move content one page down.

   if (!fCanvas) return;

   TGDimension dim = GetPageDimension();

   Int_t newpos = fCanvas->GetVsbPosition() + dim.fHeight;

   fCanvas->SetVsbPosition(newpos);
}

//______________________________________________________________________________
void TGListTree::LineUp(Bool_t /*select*/)
{
   // Move content one item-size up.

   Int_t height = 0;
   if (!fCurrent) return;
   
   TGDimension dim = GetPageDimension();
   TGPosition pos = GetPagePosition();
   const TGPicture *pic1 = fCurrent->GetPicture();
   if (pic1) height = pic1->GetHeight() + fVspacing;
   else height = fVspacing + 16;
   Int_t findy = (fCurrent->fY - height) + (fMargin - pos.fY);
   TGListTreeItem *next = FindItem(findy);
   if (next && (next != fCurrent)) {
      DrawOutline(fId, fCurrent, 0xffffff, kTRUE);
      if (findy <= 2*height) {
         Int_t newpos = fCanvas->GetVsbPosition() - height;
         if (newpos<0) newpos = 0;
         fCanvas->SetVsbPosition(newpos);
      }
      DrawOutline(fId, next);
      fCurrent = next;
   }
}

//______________________________________________________________________________
void TGListTree::LineDown(Bool_t /*select*/)
{
   // Move content one item-size down.

   Int_t height;
   if (!fCurrent) return;
   
   TGDimension dim = GetPageDimension();
   TGPosition pos = GetPagePosition();
   const TGPicture *pic1 = fCurrent->GetPicture();
   if (pic1) height = pic1->GetHeight() + fVspacing;
   else height = fVspacing + 16;
   Int_t findy = (fCurrent->fY + height) + (fMargin - pos.fY);
   TGListTreeItem *next = FindItem(findy);
   if (next && (next != fCurrent)) {
      DrawOutline(fId, fCurrent, 0xffffff, kTRUE);
      if (findy >= ((Int_t)dim.fHeight - 2*height)) {
         Int_t newpos = fCanvas->GetVsbPosition() + height;
         if (newpos<0) newpos = 0;
         fCanvas->SetVsbPosition(newpos);
      }
      DrawOutline(fId, next);
      fCurrent = next;
   }
}

//______________________________________________________________________________
void TGListTree::AdjustPosition(TGListTreeItem *item)
{
   // Move content to position of item. If item is 0, move to position
   // of currently selected item.

   TGListTreeItem *it = item;

   if (!it) it = fSelected;
   if (!it) {
      HighlightItem(fFirst); // recursive call of this function
      return;
   }

   Int_t y = 0;
   Int_t yparent = 0;
   Int_t vh = 0;
   Int_t v = 0;

   if (it) {
      y = it->fY;
      if (it->GetParent()) yparent = it->GetParent()->fY;
   }

   if (y==0) y = yparent; // item->fY not initiated yet

   if (fCanvas->GetVScrollbar()->IsMapped()) {
      vh = fCanvas->GetVScrollbar()->GetPosition()+(Int_t)fViewPort->GetHeight();

      if (y<fCanvas->GetVScrollbar()->GetPosition()) {
         v = TMath::Max(0,y-(Int_t)fViewPort->GetHeight()/2);
         fCanvas->SetVsbPosition(v);
      } else if (y+(Int_t)it->fHeight>vh) {
         v = TMath::Min((Int_t)GetHeight()-(Int_t)fViewPort->GetHeight(),
                        y+(Int_t)it->fHeight-(Int_t)fViewPort->GetHeight()/2);
         if (v<0) v = 0;
         fCanvas->SetVsbPosition(v);
      }
   }
}

//______________________________________________________________________________
void TGListTree::Search(Bool_t close)
{
   // Invokes search dialog. Looks for item with the entered name.

   Int_t ret = 0;
   char msg[256];
   static TString buf;

   TGSearchType *srch = new TGSearchType;
   srch->fBuffer = (char *)StrDup(buf.Data());

   TGListTreeItem *item;
   if (close || buf.IsNull())
      new TGSearchDialog(fClient->GetDefaultRoot(), fCanvas, 400, 150, srch, &ret);
   else if (!buf.IsNull()) ret = 1;

   if (ret) {
      item = FindItemByPathname(srch->fBuffer);
      if (!item) {
         snprintf(msg, 255, "Couldn't find \"%s\"", srch->fBuffer);
         gVirtualX->Bell(20);
         new TGMsgBox(fClient->GetDefaultRoot(), fCanvas, "Container", msg,
                      kMBIconExclamation, kMBOk, 0);
      } else {
         ClearHighlighted();
         HighlightItem(item);
      }
   }
   buf = srch->fBuffer;
   delete srch;
}

//---- drawing functions

//______________________________________________________________________________
void TGListTree::DrawRegion(Int_t /*x*/, Int_t y, UInt_t /*w*/, UInt_t h)
{
   // Redraw list tree.

   static GContext_t gcBg = 0;

   // sanity checks
   if (y > (Int_t)fViewPort->GetHeight()) {
      return;
   }

   y = y < 0 ? 0 : y;
   UInt_t w = fViewPort->GetWidth();

   // more sanity checks
   if (((Int_t)w < 1) || (w > 32768) || ((Int_t)h < 1)) {
      return;
   }

   Pixmap_t pixmap = gVirtualX->CreatePixmap(fId, w, fViewPort->GetHeight());

   if (!gcBg) {
      GCValues_t gcValues;
      gcValues.fForeground = fBackground;
      gcValues.fForeground = fBackground;
      gcValues.fGraphicsExposures = kTRUE;
      gcValues.fMask = kGCForeground | kGCBackground | kGCGraphicsExposures;
      gcBg = gVirtualX->CreateGC(fId, &gcValues);
   }

   gVirtualX->SetForeground(gcBg, fBackground);
   gVirtualX->FillRectangle(pixmap, gcBg, 0, 0, w, fViewPort->GetHeight());

   Draw(pixmap, 0, fViewPort->GetHeight());

   gVirtualX->CopyArea(pixmap, fId, gcBg, 0, y, w, fViewPort->GetHeight(), 0, y);

   gVirtualX->DeletePixmap(pixmap);
   gVirtualX->Update(kFALSE);
}

//______________________________________________________________________________
void TGListTree::Draw(Handle_t id, Int_t yevent, Int_t hevent)
{
   // Draw list tree widget.

   TGListTreeItem *item;
   Int_t  x, y, xbranch;
   UInt_t width, height, old_width, old_height;

   // Overestimate the expose region to be sure to draw an item that gets
   // cut by the region
   fExposeTop = yevent - FontHeight();
   fExposeBottom = yevent + hevent + FontHeight();
   old_width  = fDefw;
   old_height = fDefh;
   fDefw = fDefh = 1;

   TGPosition pos = GetPagePosition();
   x = 2-pos.fX;
   y = fMargin;
   item = fFirst;

   while (item) {
      xbranch = -1;

      DrawItem(id, item, x, y, &xbranch, &width, &height);

      width += pos.fX + x + fHspacing + fMargin;

      if (width > fDefw) fDefw = width;

      y += height + fVspacing;
      if (item->fFirstchild && item->IsOpen()) {
         y = DrawChildren(id, item->fFirstchild, x, y, xbranch);
      }

      item = item->fNextsibling;
   }

   fDefh = y + fMargin;

   if ((old_width != fDefw) || (old_height != fDefh)) {
      fCanvas->Layout();
   }
}

//______________________________________________________________________________
Int_t TGListTree::DrawChildren(Handle_t id, TGListTreeItem *item,
                               Int_t x, Int_t y, Int_t xroot)
{
   // Draw children of item in list tree.

   UInt_t width, height;
   Int_t  xbranch;
   TGPosition pos = GetPagePosition();

   x += fIndent + (Int_t)item->fParent->GetPicWidth();
   while (item) {
      xbranch = xroot;

      DrawItem(id, item, x, y, &xbranch, &width, &height);

      width += pos.fX + x + fHspacing + fMargin;
      if (width > fDefw) fDefw = width;

      y += height + fVspacing;
      if ((item->fFirstchild) && (item->IsOpen())) {
         y = DrawChildren(id, item->fFirstchild, x, y, xbranch);
      }

      item = item->fNextsibling;
   }
   return y;
}

//______________________________________________________________________________
void TGListTree::DrawItem(Handle_t id, TGListTreeItem *item, Int_t x, Int_t y,
                          Int_t *xroot, UInt_t *retwidth, UInt_t *retheight)
{
   // Draw list tree item.

   Int_t  xpic1, ypic1, xbranch, ybranch, xtext, ytext, xline, yline, xc;
   Int_t  xpic2 = 0;
   UInt_t height;
   const TGPicture *pic1 = item->GetPicture();
   const TGPicture *pic2 = item->GetCheckBoxPicture();

   // Compute the height of this line
   height = FontHeight();

   xline = 0;
   xpic1 = x;
   xtext = x + fHspacing + (Int_t)item->GetPicWidth();
   if (pic2) {
      if (pic2->GetHeight() > height) {
         ytext = y + (Int_t)((pic2->GetHeight() - height) >> 1);
         height = pic2->GetHeight();
      } else {
         ytext = y;
      }
      if (pic1) xpic2 = xpic1 + pic1->GetWidth() + 1;
      else xpic2 = xpic1 + 1;
      xtext += pic2->GetWidth();
   } else {
      ypic1 = y;
      xline = 0;
   }
   if (pic1) {
      if (pic1->GetHeight() > height) {
         ytext = y + (Int_t)((pic1->GetHeight() - height) >> 1);
         height = pic1->GetHeight();
         ypic1 = y;
      } else {
#ifdef R__HAS_COCOA
         if (!pic2)//DO NOT MODIFY ytext, it WAS ADJUSTED already!
#endif
         ytext = y;
         ypic1 = y + (Int_t)((height - pic1->GetHeight()) >> 1);
      }
      xbranch = xpic1 + (Int_t)(pic1->GetWidth() >> 1);
      ybranch = ypic1 + (Int_t)pic1->GetHeight();
      yline = ypic1 + (Int_t)(pic1->GetHeight() >> 1);
      if (xline == 0) xline = xpic1;
   } else {
      if (xline == 0) xline = xpic1;
      ypic1 = ytext = y;
      xbranch = xpic1 + (Int_t)(item->GetPicWidth() >> 1);
      yline = ybranch = ypic1 + (Int_t)(height >> 1);
      yline = ypic1 + (Int_t)(height >> 1);
   }

   // height must be even, otherwise our dashed line wont appear properly
   //++height; height &= ~1;

   // Save the basic graphics info for use by other functions
   item->fY      = y;
   item->fXtext  = xtext;
   item->fYtext  = ytext;
   item->fHeight = height;

   // projected coordinates
   TGPosition  pos = GetPagePosition();
   TGDimension dim = GetPageDimension();
   Int_t yp        = y       - pos.fY;
   Int_t ylinep    = yline   - pos.fY;
   Int_t ybranchp  = ybranch - pos.fY;
   Int_t ypicp     = ypic1   - pos.fY;

   if ((yp >= fExposeTop) && (yp <= (Int_t)dim.fHeight))
   {
      DrawItemName(id, item);
      if (*xroot >= 0) {
         xc = *xroot;

         if (item->fNextsibling) {
            gVirtualX->DrawLine(id, fLineGC, xc, yp, xc, yp+height);
         } else {
            gVirtualX->DrawLine(id, fLineGC, xc, yp, xc, ylinep);
         }

         TGListTreeItem *p = item->fParent;
         while (p) {
            xc -= (fIndent + (Int_t)item->GetPicWidth());
            if (p->fNextsibling) {
               gVirtualX->DrawLine(id, fLineGC, xc, yp, xc, yp+height);
            }
            p = p->fParent;
         }
         gVirtualX->DrawLine(id, fLineGC, *xroot, ylinep, xpic1, ylinep);
         DrawNode(id, item, *xroot, yline);
      }
      if (item->IsOpen() && item->fFirstchild) {
         gVirtualX->DrawLine(id, fLineGC, xbranch, ybranchp, xbranch, 
                             yp+height);
      }
      if (pic1)
         pic1->Draw(id, fDrawGC, xpic1, ypicp);
      if (pic2)
         pic2->Draw(id, fDrawGC, xpic2, ypicp);
   }

   *xroot = xbranch;
   *retwidth  = TextWidth(item->GetText()) + item->GetPicWidth();
   *retheight = height;
}

//______________________________________________________________________________
void TGListTree::DrawOutline(Handle_t id, TGListTreeItem *item, Pixel_t col, 
                             Bool_t clear)
{
   // Draw a outline of color 'col' around an item.

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   if (clear) {
      gVirtualX->SetForeground(fDrawGC, fCanvas->GetContainer()->GetBackground());
      //ClearViewPort();  // time consuming!!!
   }
   else
      gVirtualX->SetForeground(fDrawGC, col);

#ifdef R__HAS_COCOA
   gVirtualX->DrawRectangle(id, fDrawGC, 1, item->fY - pos.fY, dim.fWidth-2, item->fHeight + 1);
#else
   gVirtualX->DrawRectangle(id, fDrawGC, 1, item->fYtext-pos.fY-2, 
                            dim.fWidth-3, FontHeight()+4);
#endif
   gVirtualX->SetForeground(fDrawGC, fgBlackPixel);
}

//______________________________________________________________________________
void TGListTree::DrawActive(Handle_t id, TGListTreeItem *item)
{
   // Draw active item with its active color.

   UInt_t width;
   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   width = dim.fWidth-2;
   gVirtualX->SetForeground(fDrawGC, item->GetActiveColor());

#ifdef R__HAS_COCOA
   gVirtualX->FillRectangle(id, fDrawGC, 1, item->fY - pos.fY, width, item->fHeight + 1);
#else
   gVirtualX->FillRectangle(id, fDrawGC, 1, item->fYtext-pos.fY-1, width, 
                            FontHeight()+3);
#endif
   gVirtualX->SetForeground(fDrawGC, fgBlackPixel);
   gVirtualX->DrawString(id, fActiveGC, item->fXtext, 
                         item->fYtext - pos.fY + FontAscent(),
                         item->GetText(), item->GetTextLength());
}

//______________________________________________________________________________
void TGListTree::DrawItemName(Handle_t id, TGListTreeItem *item)
{
   // Draw name of list tree item.

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   if (item->IsActive()) {
      DrawActive(id, item);
   }
   else { // if (!item->IsActive() && (item != fSelected)) {
      gVirtualX->FillRectangle(id, fHighlightGC, item->fXtext,
                       item->fYtext-pos.fY, dim.fWidth-item->fXtext-2,
                       FontHeight()+1);
      gVirtualX->DrawString(id, fDrawGC,
                       item->fXtext, item->fYtext-pos.fY + FontAscent(),
                       item->GetText(), item->GetTextLength());
   }
   if (item == fCurrent) {
      DrawOutline(id, item);
   }

   if (fColorMode != 0 && item->HasColor()) {
      UInt_t width = TextWidth(item->GetText());
      gVirtualX->SetForeground(fColorGC, TColor::Number2Pixel(item->GetColor()));
      if (fColorMode & kColorUnderline) {
         Int_t y = item->fYtext-pos.fY + FontAscent() + 2;
         gVirtualX->DrawLine(id, fColorGC, item->fXtext, y, 
                             item->fXtext + width, y);
      }
      if (fColorMode & kColorBox) {
         Int_t x = item->fXtext + width + 4;
         Int_t y = item->fYtext - pos.fY  + 3;
         Int_t h = FontAscent()    - 4;
         gVirtualX->FillRectangle(id, fColorGC, x, y, h, h);
         gVirtualX->DrawRectangle(id, fDrawGC,  x, y, h, h);
      }
   }
}

//______________________________________________________________________________
void TGListTree::DrawNode(Handle_t id, TGListTreeItem *item, Int_t x, Int_t y)
{
   // Draw node (little + in box).

   TGPosition pos = GetPagePosition();
   Int_t yp = y - pos.fY;

   if (item->fFirstchild) {
      gVirtualX->DrawLine(id, fHighlightGC, x, yp-2, x, yp+2);
      gVirtualX->SetForeground(fHighlightGC, fgBlackPixel);
      gVirtualX->DrawLine(id, fHighlightGC, x-2, yp, x+2, yp);
      if (!item->IsOpen())
         gVirtualX->DrawLine(id, fHighlightGC, x, yp-2, x, yp+2);
      gVirtualX->SetForeground(fHighlightGC, fGrayPixel);
      gVirtualX->DrawLine(id, fHighlightGC, x-4, yp-4, x+4, yp-4);
      gVirtualX->DrawLine(id, fHighlightGC, x+4, yp-4, x+4, yp+4);
      gVirtualX->DrawLine(id, fHighlightGC, x-4, yp+4, x+4, yp+4);
      gVirtualX->DrawLine(id, fHighlightGC, x-4, yp-4, x-4, yp+4);
      gVirtualX->SetForeground(fHighlightGC, fgWhitePixel);
   }
}

//______________________________________________________________________________
void TGListTree::SetToolTipText(const char *text, Int_t x, Int_t y, Long_t delayms)
{
   // Set tool tip text associated with this item. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // delayms = 0. To change delayms you first have to call this method
   // with delayms=0.

   if (delayms == 0) {
      delete fTip;
      fTip = 0;
      return;
   }

   if (text && strlen(text)) {
      if (!fTip)
         fTip = new TGToolTip(fClient->GetDefaultRoot(), this, text, delayms);
      else
         fTip->SetText(text);
      fTip->SetPosition(x, y);
      fTip->Reset();
   }
}

//______________________________________________________________________________
void TGListTree::RemoveReference(TGListTreeItem *item)
{
   // This function removes the specified item from the linked list.
   // It does not do anything with the data contained in the item, though.

   ClearViewPort();

   // Disentangle from front (previous-sibling, parent's first child)
   if (item->fPrevsibling) {
      item->fPrevsibling->fNextsibling = item->fNextsibling;
   } else {
      if (item->fParent)
         item->fParent->fFirstchild = item->fNextsibling;
      else
         fFirst = item->fNextsibling;
   }
   // Disentangle from end (next-sibling, parent's last child)
   if (item->fNextsibling) {
      item->fNextsibling->fPrevsibling = item->fPrevsibling;
   } else {
      if (item->fParent)
         item->fParent->fLastchild = item->fPrevsibling;
      else
         fLast = item->fPrevsibling;
   }
}

//______________________________________________________________________________
void TGListTree::PDeleteItem(TGListTreeItem *item)
{
   // Delete given item. Takes care of list-tree state members
   // fSelected, fCurrent and fBelowMouse.

   if (fSelected == item) {
      fSelected = 0;
   }
   if (fCurrent == item) {
      DrawOutline(fId, fCurrent, 0xffffff, kTRUE);
      fCurrent = item->GetPrevSibling();
      if (! fCurrent) {
         fCurrent = item->GetNextSibling();
         if (! fCurrent)
            fCurrent = item->GetParent();
      }
   }
   if (fBelowMouse == item) {
      DrawOutline(fId, fBelowMouse, 0xffffff, kTRUE);
      fBelowMouse = 0;
      MouseOver(0);
      MouseOver(0,fLastEventState);
   }

   delete item; 
}

//______________________________________________________________________________
void TGListTree::PDeleteChildren(TGListTreeItem *item)
{
   // Recursively delete all children of an item.

   TGListTreeItem *child = item->fFirstchild;

   while (child) {
      TGListTreeItem *next = child->fNextsibling;
      PDeleteChildren(child);
      PDeleteItem(child);
      child = next;
   }

   item->fFirstchild = item->fLastchild = 0;
}

//______________________________________________________________________________
void TGListTree::InsertChild(TGListTreeItem *parent, TGListTreeItem *item)
{
   // Insert child in list.

   TGListTreeItem *i;

   item->fParent = parent;
   item->fNextsibling = item->fPrevsibling = 0;

   if (parent) {

      if (parent->fFirstchild) {
         if (parent->fLastchild) {
            i = parent->fLastchild;
         }
         else {
            i = parent->fFirstchild;
            while (i->fNextsibling) i = i->fNextsibling;
         }
         i->fNextsibling = item;
         item->fPrevsibling = i;
      } else {
         parent->fFirstchild = item;
      }
      parent->fLastchild = item;

   } else {  // if parent == 0, this is a top level entry

      if (fFirst) {
         if (fLast) {
            i = fLast;
         }
         else {
            i = fFirst;
            while (i->fNextsibling) i = i->fNextsibling;
         }
         i->fNextsibling = item;
         item->fPrevsibling = i;
      } else {
         fFirst = item;
      }
      fLast = item;
   }
   if (item->HasCheckBox())
      UpdateChecked(item);
}

//______________________________________________________________________________
void TGListTree::InsertChildren(TGListTreeItem *parent, TGListTreeItem *item)
{
   // Insert a list of ALREADY LINKED children into another list

   TGListTreeItem *next, *newnext;

   //while (item) {
   //   next = item->fNextsibling;
   //   InsertChild(parent, item);
   //   item = next;
   //}
   //return;

   // Save the reference for the next item in the new list
   next = item->fNextsibling;

   // Insert the first item in the new list into the existing list
   InsertChild(parent, item);

   // The first item is inserted, with its prev and next siblings updated
   // to fit into the existing list. So, save the existing list reference
   newnext = item->fNextsibling;

   // Now, mark the first item's next sibling to point back to the new list
   item->fNextsibling = next;

   // Mark the parents of the new list to the new parent. The order of the
   // rest of the new list should be OK, and the second item should still
   // point to the first, even though the first was reparented.
   while (item->fNextsibling) {
      item->fParent = parent;
      item = item->fNextsibling;
   }

   // Fit the end of the new list back into the existing list
   item->fNextsibling = newnext;
   if (newnext)
      newnext->fPrevsibling = item;
}

//______________________________________________________________________________
Int_t TGListTree::SearchChildren(TGListTreeItem *item, Int_t y, Int_t findy,
                                 TGListTreeItem **finditem)
{
   // Search child item.

   UInt_t height;
   const TGPicture *pic;

   while (item) {
      // Select the pixmap to use
      pic = item->GetPicture();

      // Compute the height of this line
      height = FontHeight();
      if (pic && pic->GetHeight() > height)
         height = pic->GetHeight();

      if ((findy >= y) && (findy <= y + (Int_t)height)) {
         *finditem = item;
         return -1;
      }

      y += (Int_t)height + fVspacing;
      if (item->fFirstchild && item->IsOpen()) {
         y = SearchChildren(item->fFirstchild, y, findy, finditem);
         if (*finditem) return -1;
      }

      item = item->fNextsibling;
   }

   return y;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindItem(Int_t findy)
{
   // Find item at postion findy.

   Int_t  y;
   UInt_t height;
   TGListTreeItem *item, *finditem;
   const TGPicture *pic;
   TGPosition pos = GetPagePosition();

   y = fMargin - pos.fY;
   item = fFirst;
   finditem = 0;
   while (item && !finditem) {
      // Select the pixmap to use
      pic = item->GetPicture();

      // Compute the height of this line
      height = FontHeight();
      if (pic && (pic->GetHeight() > height))
         height = pic->GetHeight();

      if ((findy >= y) && (findy <= y + (Int_t)height))
         return item;

      y += (Int_t)height + fVspacing;
      if ((item->fFirstchild) && (item->IsOpen())) {
         y = SearchChildren(item->fFirstchild, y, findy, &finditem);
         //if (finditem) return finditem;
      }
      item = item->fNextsibling;
   }

   return finditem;
}

//----- Public Functions

//______________________________________________________________________________
void TGListTree::AddItem(TGListTreeItem *parent, TGListTreeItem *item)
{
   // Add given item to list tree.

   InsertChild(parent, item);

   if ((parent == 0) || (parent && parent->IsOpen()))
      ClearViewPort();
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::AddItem(TGListTreeItem *parent, const char *string,
                                    const TGPicture *open, const TGPicture *closed,
                                    Bool_t checkbox)
{
   // Add item to list tree. Returns new item.

   TGListTreeItem *item;

   item = new TGListTreeItemStd(fClient, string, open, closed, checkbox);
   InsertChild(parent, item);

   if ((parent == 0) || (parent && parent->IsOpen()))
      ClearViewPort();
   return item;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::AddItem(TGListTreeItem *parent, const char *string,
                                    void *userData, const TGPicture *open,
                                    const TGPicture *closed,
                                    Bool_t checkbox)
{
   // Add item to list tree. If item with same userData already exists
   // don't add it. Returns new item.

   TGListTreeItem *item = FindChildByData(parent, userData);
   if (!item) {
      item = AddItem(parent, string, open, closed, checkbox);
      if (item) item->SetUserData(userData);
   }

   return item;
}

//______________________________________________________________________________
void TGListTree::RenameItem(TGListTreeItem *item, const char *string)
{
   // Rename item in list tree.

   if (item) {
      item->Rename(string);
   }

   DoRedraw();
}

//______________________________________________________________________________
Int_t TGListTree::DeleteItem(TGListTreeItem *item)
{
   // Delete item from list tree.

   if (!fUserControlled)
      fCurrent = fBelowMouse = 0;

   PDeleteChildren(item);
   RemoveReference(item);
   PDeleteItem(item);

   fClient->NeedRedraw(this);

   return 1;
}

//______________________________________________________________________________
void TGListTree::OpenItem(TGListTreeItem *item)
{
   // Open item in list tree (i.e. show child items).

   if (item) {
      item->SetOpen(kTRUE);
      DoRedraw(); // force layout
      AdjustPosition(item);
   }
}

//______________________________________________________________________________
void TGListTree::CloseItem(TGListTreeItem *item)
{
   // Close item in list tree (i.e. hide child items).

   if (item) {
      item->SetOpen(kFALSE);
      DoRedraw(); // force layout
      AdjustPosition(item);
   }
}

//______________________________________________________________________________
Int_t TGListTree::RecursiveDeleteItem(TGListTreeItem *item, void *ptr)
{
   // Delete item with fUserData == ptr. Search tree downwards starting
   // at item.

   if (item && ptr) {
      if (item->GetUserData() == ptr) {
         DeleteItem(item);
      } else {
         if (item->IsOpen() && item->fFirstchild) {
            RecursiveDeleteItem(item->fFirstchild,  ptr);
         }
         RecursiveDeleteItem(item->fNextsibling, ptr);
      }
   }
   return 1;
}

//______________________________________________________________________________
void TGListTree::SetToolTipItem(TGListTreeItem *item, const char *string)
{
   // Set tooltip text for this item. By default an item for which the
   // userData is a pointer to an TObject the TObject::GetTitle() will
   // be used to get the tip text.

   if (item) {
      item->SetTipText(string);
   }
}

//______________________________________________________________________________
Int_t TGListTree::DeleteChildren(TGListTreeItem *item)
{
   // Delete children of item from list.

   if (!fUserControlled)
      fCurrent = fBelowMouse = 0;

   PDeleteChildren(item);

   DoRedraw();

   return 1;
}

//______________________________________________________________________________
Int_t TGListTree::Reparent(TGListTreeItem *item, TGListTreeItem *newparent)
{
   // Make newparent the new parent of item.

   // Remove the item from its old location.
   RemoveReference(item);

   // The item is now unattached. Reparent it.
   InsertChild(newparent, item);

   DoRedraw();

   return 1;
}

//______________________________________________________________________________
Int_t TGListTree::ReparentChildren(TGListTreeItem *item,
                                 TGListTreeItem *newparent)
{
   // Make newparent the new parent of the children of item.

   TGListTreeItem *first;

   if (item->fFirstchild) {
      first = item->fFirstchild;
      item->fFirstchild = 0;

      InsertChildren(newparent, first);

      DoRedraw();
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
extern "C"
Int_t Compare(const void *item1, const void *item2)
{
   return strcmp((*((TGListTreeItem **) item1))->GetText(),
                 (*((TGListTreeItem **) item2))->GetText());
}

//______________________________________________________________________________
Int_t TGListTree::Sort(TGListTreeItem *item)
{
   // Sort items starting with item.

   TGListTreeItem *first, *parent, **list;
   size_t i, count;

   // Get first child in list;
   while (item->fPrevsibling) item = item->fPrevsibling;

   first = item;
   parent = first->fParent;

   // Count the children
   count = 1;
   while (item->fNextsibling) item = item->fNextsibling, count++;
   if (count <= 1) return 1;

   list = new TGListTreeItem* [count];
   list[0] = first;
   count = 1;
   while (first->fNextsibling) {
      list[count] = first->fNextsibling;
      count++;
      first = first->fNextsibling;
   }

   ::qsort(list, count, sizeof(TGListTreeItem*), ::Compare);

   list[0]->fPrevsibling = 0;
   for (i = 0; i < count; i++) {
      if (i < count - 1)
         list[i]->fNextsibling = list[i + 1];
      if (i > 0)
         list[i]->fPrevsibling = list[i - 1];
   }
   list[count - 1]->fNextsibling = 0;
   if (parent) {
      parent->fFirstchild = list[0];
      parent->fLastchild  = list[count-1];
   }
   else {
      fFirst = list[0];
      fLast  = list[count-1];
   }

   delete [] list;

   DoRedraw();

   return 1;
}

//______________________________________________________________________________
Int_t TGListTree::SortSiblings(TGListTreeItem *item)
{
   // Sort siblings of item.

   return Sort(item);
}

//______________________________________________________________________________
Int_t TGListTree::SortChildren(TGListTreeItem *item)
{
   // Sort children of item.

   TGListTreeItem *first;

   if (item) {
      first = item->fFirstchild;
      if (first) {
         SortSiblings(first);
      }
   } else {
      if (fFirst) {
         first = fFirst->fFirstchild;
         if (first) {
            SortSiblings(first);
         }
      }
   }
   DoRedraw();
   return 1;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindSiblingByName(TGListTreeItem *item, const char *name)
{
   // Find sibling of item by name.

   // Get first child in list
   if (item) {
      while (item->fPrevsibling) {
         item = item->fPrevsibling;
      }

      while (item) {
         if (strcmp(item->GetText(), name) == 0) {
            return item;
         }
         item = item->fNextsibling;
      }
      return item;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindSiblingByData(TGListTreeItem *item, void *userData)
{
   // Find sibling of item by userData.

   // Get first child in list
   if (item) {
      while (item->fPrevsibling) {
         item = item->fPrevsibling;
      }

      while (item) {
         if (item->GetUserData() == userData) {
            return item;
         }
         item = item->fNextsibling;
      }
      return item;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindChildByName(TGListTreeItem *item, const char *name)
{
   // Find child of item by name.

   // Get first child in list
   if (item && item->fFirstchild) {
      item = item->fFirstchild;
   } else if (!item && fFirst) {
      item = fFirst;
   } else {
      item = 0;
   }

   while (item) {
      if (strcmp(item->GetText(), name) == 0) {
         return item;
      }
      item = item->fNextsibling;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindChildByData(TGListTreeItem *item, void *userData)
{
   // Find child of item by userData.

   // Get first child in list
   if (item && item->fFirstchild) {
      item = item->fFirstchild;
   } else if (!item && fFirst) {
      item = fFirst;
   } else {
      item = 0;
   }

   while (item) {
      if (item->GetUserData() == userData) {
         return item;
      }
      item = item->fNextsibling;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindItemByPathname(const char *path)
{
   // Find item by pathname. Pathname is in the form of /xx/yy/zz. If zz
   // in path /xx/yy is found it returns item, 0 otherwise.

   if (!path || !*path) return 0;

   const char *p = path, *s;
   char dirname[1024];
   TGListTreeItem *item = 0;
   item = FindChildByName(item, "/");
   if (!gVirtualX->InheritsFrom("TGX11")) {
      // on Windows, use the current drive instead of root (/)
      TList *curvol  = gSystem->GetVolumes("cur");
      if (curvol) {
         TNamed *drive = (TNamed *)curvol->At(0);
         item = FindChildByName(0, TString::Format("%s\\", drive->GetName()));
      }
   }
   TGListTreeItem *diritem = 0;
   TString fulldir;

   while (1) {
      while (*p && *p == '/') p++;
      if (!*p) break;

      s = strchr(p, '/');

      if (!s) {
         strlcpy(dirname, p, 1024);
      } else {
         strlcpy(dirname, p, (s-p)+1);
      }

      item = FindChildByName(item, dirname);

      if (!diritem && dirname[0]) {
         fulldir += "/";
         fulldir += dirname;

         if ((diritem=FindChildByName(0, fulldir.Data()))) {
            if (!s || !strlen(s)) return diritem;
            p = ++s;
            item = diritem;
            continue;
         }
      }

      if (!s || !strlen(s)) return item;
      p = ++s;
   }
   return 0;
}

//______________________________________________________________________________
void TGListTree::HighlightItem(TGListTreeItem *item)
{
   // Highlight item.

   UnselectAll(kFALSE);
   HighlightItem(item, kTRUE, kFALSE);
   AdjustPosition(item);
}

//______________________________________________________________________________
void TGListTree::ClearHighlighted()
{
   // Un highlight items.

   UnselectAll(kFALSE);
}

//______________________________________________________________________________
void TGListTree::GetPathnameFromItem(TGListTreeItem *item, char *path, Int_t depth)
{
   // Get pathname from item. Use depth to limit path name to last
   // depth levels. By default depth is not limited.

   char tmppath[1024];

   *path = '\0';
   while (item) {
      snprintf(tmppath, 1023, "/%s%s", item->GetText(), path);
      strlcpy(path, tmppath, 1024);
      item = item->fParent;
      if (--depth == 0 && item) {
         snprintf(tmppath, 1023, "...%s", path);
         strlcpy(path, tmppath, 1024);
         return;
      }
   }
}

//______________________________________________________________________________
Pixel_t TGListTree::GetGrayPixel()
{
   // Return gray draw color in use.

   static Bool_t init = kFALSE;
   if (!init) {
      if (!gClient->GetColorByName("#808080", fgGrayPixel))
         fgGrayPixel = fgBlackPixel;
      init = kTRUE;
   }
   return fgGrayPixel;
}

//______________________________________________________________________________
FontStruct_t TGListTree::GetDefaultFontStruct()
{
   // Return default font structure in use.

   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetIconFont();
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGListTree::GetActiveGC()
{
   // Return default graphics context in use.

   if (!fgActiveGC) {
      GCValues_t gcv;

      gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
                  kGCForeground | kGCBackground | kGCFont;
      gcv.fLineStyle  = kLineSolid;
      gcv.fLineWidth  = 0;
      gcv.fFillStyle  = kFillSolid;
      gcv.fFont       = fgDefaultFont->GetFontHandle();
      gcv.fBackground = fgDefaultSelectedBackground;
      const TGGC *selgc = gClient->GetResourcePool()->GetSelectedGC();
      if (selgc)
         gcv.fForeground = selgc->GetForeground();
      else 
         gcv.fForeground = fgWhitePixel;
      fgActiveGC = gClient->GetGC(&gcv, kTRUE);
   }
   return *fgActiveGC;
}

//______________________________________________________________________________
const TGGC &TGListTree::GetDrawGC()
{
   // Return default graphics context in use.

   if (!fgDrawGC) {
      GCValues_t gcv;

      gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
                  kGCForeground | kGCBackground | kGCFont;
      gcv.fLineStyle  = kLineSolid;
      gcv.fLineWidth  = 0;
      gcv.fFillStyle  = kFillSolid;
      gcv.fFont       = fgDefaultFont->GetFontHandle();
      gcv.fBackground = fgWhitePixel;
      gcv.fForeground = fgBlackPixel;

      fgDrawGC = gClient->GetGC(&gcv, kTRUE);
   }
   return *fgDrawGC;
}

//______________________________________________________________________________
const TGGC &TGListTree::GetLineGC()
{
   // Return graphics context in use for line drawing.

   if (!fgLineGC) {
      GCValues_t gcv;

      gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
                  kGCForeground | kGCBackground | kGCFont;
      gcv.fLineStyle  = kLineOnOffDash;
      gcv.fLineWidth  = 0;
      gcv.fFillStyle  = kFillSolid;
      gcv.fFont       = fgDefaultFont->GetFontHandle();
      gcv.fBackground = fgWhitePixel;
      gcv.fForeground = GetGrayPixel();

      fgLineGC = gClient->GetGC(&gcv, kTRUE);
      fgLineGC->SetDashOffset(0);
      fgLineGC->SetDashList("\x1\x1", 2);
   }
   return *fgLineGC;
}

//______________________________________________________________________________
const TGGC &TGListTree::GetHighlightGC()
{
   // Return graphics context for highlighted frame background.

   if (!fgHighlightGC) {
      GCValues_t gcv;

      gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
                  kGCForeground | kGCBackground | kGCFont;
      gcv.fLineStyle  = kLineSolid;
      gcv.fLineWidth  = 0;
      gcv.fFillStyle  = kFillSolid;
      gcv.fFont       = fgDefaultFont->GetFontHandle();
      gcv.fBackground = fgDefaultSelectedBackground;
      gcv.fForeground = fgWhitePixel;

      fgHighlightGC = gClient->GetGC(&gcv, kTRUE);
   }
   return *fgHighlightGC;
}

//______________________________________________________________________________
const TGGC &TGListTree::GetColorGC()
{
   // Return graphics context for highlighted frame background.

   if (!fgColorGC) {
      GCValues_t gcv;

      gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
                  kGCForeground | kGCBackground;
      gcv.fLineStyle  = kLineSolid;
      gcv.fLineWidth  = 1;
      gcv.fFillStyle  = kFillSolid;
      gcv.fBackground = fgDefaultSelectedBackground;
      gcv.fForeground = fgWhitePixel;

      fgColorGC = gClient->GetGC(&gcv, kTRUE);
   }
   return *fgColorGC;
}

//______________________________________________________________________________
const TGPicture *TGListTree::GetOpenPic()
{
   // Returns the icon used by items in open state.

   if (!fgOpenPic)
      fgOpenPic = gClient->GetPicture("ofolder_t.xpm");
   ((TGPicture *)fgOpenPic)->AddReference();
   return fgOpenPic;
}

//______________________________________________________________________________
const TGPicture *TGListTree::GetClosedPic()
{
   // Returns the icon used by items in closed state.

   if (!fgClosedPic)
      fgClosedPic = gClient->GetPicture("folder_t.xpm");
   ((TGPicture *)fgClosedPic)->AddReference();
   return fgClosedPic;
}

//______________________________________________________________________________
const TGPicture *TGListTree::GetCheckedPic()
{
   // Returns the icon used for checked checkbox.

   if (!fgCheckedPic)
      fgCheckedPic = gClient->GetPicture("checked_t.xpm");
   ((TGPicture *)fgCheckedPic)->AddReference();
   return fgCheckedPic;
}

//______________________________________________________________________________
const TGPicture *TGListTree::GetUncheckedPic()
{
   // Returns the icon used for unchecked checkbox.

   if (!fgUncheckedPic)
      fgUncheckedPic = gClient->GetPicture("unchecked_t.xpm");
   ((TGPicture *)fgUncheckedPic)->AddReference();
   return fgUncheckedPic;
}

//______________________________________________________________________________
void TGListTree::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   // Save a list tree widget as a C++ statements on output stream out.

   if (fBackground != GetWhitePixel()) SaveUserColor(out, option);

   out << std::endl << "   // list tree" << std::endl;
   out << "   TGListTree *";

   if ((fParent->GetParent())->InheritsFrom(TGCanvas::Class())) {
      out << GetName() << " = new TGListTree(" << GetCanvas()->GetName();
   } else {
      out << GetName() << " = new TGListTree(" << fParent->GetName();
      out << "," << GetWidth() << "," << GetHeight();
   }

   if (fBackground == GetWhitePixel()) {
      if (GetOptions() == kSunkenFrame) {
         out <<");" << std::endl;
      } else {
         out << "," << GetOptionString() <<");" << std::endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << std::endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << std::endl;

   out << std::endl;

   static Int_t n = 0;

   TGListTreeItem *current;
   current = GetFirstItem();

   out << "   const TGPicture *popen;       //used for list tree items" << std::endl;
   out << "   const TGPicture *pclose;      //used for list tree items" << std::endl;
   out << std::endl;

   while (current) {
      out << "   TGListTreeItem *item" << n << " = " << GetName() << "->AddItem(";
      current->SavePrimitive(out, TString::Format("%d",n), n);
      if (current->IsOpen())
         out << "   " << GetName() << "->OpenItem(item" << n << ");" << std::endl;
      else
         out << "   " << GetName() << "->CloseItem(item" << n << ");" << std::endl;

      if (current == fSelected)
         out << "   " << GetName() << "->SetSelected(item" << n << ");" << std::endl;

      n++;
      if (current->fFirstchild) {
         SaveChildren(out, current->fFirstchild, n);
      }
      current = current->fNextsibling;
   }

   out << std::endl;
}

//______________________________________________________________________________
void TGListTree::SaveChildren(std::ostream &out, TGListTreeItem *item, Int_t &n)
{
   // Save child items as a C++ statements on output stream out.

   Int_t p = n-1;
   while (item) {
      out << "   TGListTreeItem *item" << n << " = " << GetName() << "->AddItem(";
      item->SavePrimitive(out, TString::Format("%d",p),n);
      n++;
      if (item->fFirstchild) {
         SaveChildren(out, item->fFirstchild, n);
      }
      item = item->fNextsibling;
   }
}

//______________________________________________________________________________
void TGListTreeItemStd::SavePrimitive(std::ostream &out, Option_t *option, Int_t n)
{
   // Save a list tree item attributes as a C++ statements on output stream.

   static const TGPicture *oldopen=0;
   static const TGPicture *oldclose=0;
   static const TGPicture *oldcheck=0;
   static const TGPicture *olduncheck=0;
   static Bool_t makecheck = kTRUE;
   static Bool_t makeuncheck = kTRUE;
   static Color_t oldcolor = -1;

   char quote = '"';
   TString s = TString::Format("%d", n);

   if (!fParent)
      out << "NULL,";
   else
      out << "item" << option << ",";
   TString text = GetText();
   text.ReplaceAll('\\', "\\\\");
   text.ReplaceAll("\"", "\\\"");
   out << quote << text << quote;
   out << ");" << std::endl;

   if (oldopen != fOpenPic) {
      oldopen = fOpenPic;
      out << "   popen = gClient->GetPicture(" << quote
          << gSystem->ExpandPathName(gSystem->UnixPathName(fOpenPic->GetName()))
          << quote << ");" << std::endl;
   }
   if (oldclose != fClosedPic) {
      oldclose = fClosedPic;
      out << "   pclose = gClient->GetPicture(" << quote
          << gSystem->ExpandPathName(gSystem->UnixPathName(fClosedPic->GetName()))
          << quote << ");" << std::endl;
   }
   out << "   item" << s.Data() << "->SetPictures(popen, pclose);" << std::endl;
   if (HasCheckBox()) {
      if (fCheckedPic && makecheck) {
         out << "   const TGPicture *pcheck;        //used for checked items" << std::endl;
         makecheck = kFALSE;
      }
      if (fUncheckedPic && makeuncheck) {
         out << "   const TGPicture *puncheck;      //used for unchecked items" << std::endl;
         makeuncheck = kFALSE;
      }
      out << "   item" << s.Data() << "->CheckItem();" << std::endl;
      if (fCheckedPic && oldcheck != fCheckedPic) {
         oldcheck = fCheckedPic;
         out << "   pcheck = gClient->GetPicture(" << quote
             << gSystem->ExpandPathName(gSystem->UnixPathName(fCheckedPic->GetName()))
             << quote << ");" << std::endl;
      }
      if (fUncheckedPic && olduncheck != fUncheckedPic) {
         olduncheck = fUncheckedPic;
         out << "   puncheck = gClient->GetPicture(" << quote
             << gSystem->ExpandPathName(gSystem->UnixPathName(fUncheckedPic->GetName()))
             << quote << ");" << std::endl;
      }
      out << "   item" << s.Data() << "->SetCheckBoxPictures(pcheck, puncheck);" << std::endl;
      out << "   item" << s.Data() << "->SetCheckBox(kTRUE);" << std::endl;
   }
   if (fHasColor) {
      if (oldcolor != fColor) {
         oldcolor = fColor;
         out << "   item" << s.Data() << "->SetColor(" << fColor << ");" << std::endl;
      }
   }
   if (fTipText.Length() > 0) {
      TString tiptext = GetTipText();
      tiptext.ReplaceAll('\\', "\\\\");
      tiptext.ReplaceAll("\n", "\\n");
      tiptext.ReplaceAll("\"", "\\\"");
      out << "   item" << s.Data() << "->SetTipText(" << quote
          << tiptext << quote << ");" << std::endl;
   }

}

//______________________________________________________________________________
void TGListTree::CheckItem(TGListTreeItem *item, Bool_t check)
{
   // Set check button state for the node 'item'.

   item->CheckItem(check);
}

//______________________________________________________________________________
void TGListTree::SetCheckBox(TGListTreeItem *item, Bool_t on)
{
   // Set check button state for the node 'item'.

   item->SetCheckBox(on);
}

//______________________________________________________________________________
void TGListTree::ToggleItem(TGListTreeItem *item)
{
   // Toggle check button state of the node 'item'.

   item->Toggle();
}

//______________________________________________________________________________
void TGListTree::UpdateChecked(TGListTreeItem *item, Bool_t redraw)
{
   // Update the state of the node 'item' according to the children states.

   if (fAutoCheckBoxPic == kFALSE) return;

   TGListTreeItem *parent;
   TGListTreeItem *current;
   current = item->GetFirstChild();
   parent  = current ? current : item;
   // recursively check parent/children status
   while (parent && parent->HasCheckBox()) {
      if ((!parent->IsChecked() && parent->HasCheckedChild(kTRUE)) ||
          (parent->IsChecked() && parent->HasUnCheckedChild(kTRUE))) {
         parent->SetCheckBoxPictures(fClient->GetPicture("checked_dis_t.xpm"),
                                     fClient->GetPicture("unchecked_dis_t.xpm"));
      }
      else {
         parent->SetCheckBoxPictures(fClient->GetPicture("checked_t.xpm"),
                                     fClient->GetPicture("unchecked_t.xpm"));
      }
      parent = parent->GetParent();
      if (parent && fCheckMode == kRecursive) {
         if (!parent->IsChecked() && parent->GetFirstChild() && 
             !parent->GetFirstChild()->HasUnCheckedChild()) {
            parent->SetCheckBoxPictures(fClient->GetPicture("checked_t.xpm"),
                                        fClient->GetPicture("unchecked_t.xpm"));
            parent->CheckItem(kTRUE);
         }
         else if (parent->IsChecked() && parent->GetFirstChild() && 
                  !parent->GetFirstChild()->HasCheckedChild()) {
            parent->SetCheckBoxPictures(fClient->GetPicture("checked_t.xpm"),
                                        fClient->GetPicture("unchecked_t.xpm"));
            parent->CheckItem(kFALSE);
         }
      }
   }
   if (redraw) {
      ClearViewPort();
   }
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindItemByObj(TGListTreeItem *item, void *ptr)
{
   // Find item with fUserData == ptr. Search tree downwards starting
   // at item.

   TGListTreeItem *fitem;
   if (item && ptr) {
      if (item->GetUserData() == ptr)
         return item;
      else {
         if (item->fFirstchild) {
            fitem = FindItemByObj(item->fFirstchild,  ptr);
            if (fitem) return fitem;
         }
         return FindItemByObj(item->fNextsibling, ptr);
      }
   }
   return 0;
}

//______________________________________________________________________________
void TGListTree::GetChecked(TList *checked)
{
   // Add all checked list tree items of this list tree into 
   // the list 'checked'. This list is not adopted and must
   // be deleted by the user later.
   
   if (!checked || !fFirst) return;
   TGListTreeItem *current = fFirst;
   if (current->IsChecked()) {
      checked->Add(new TObjString(current->GetText()));
   }
   while(current) {
      if (current->GetFirstChild())
         GetCheckedChildren(checked, current->GetFirstChild());
      current = current->GetNextSibling();
   }
}

//______________________________________________________________________________
void TGListTree::GetCheckedChildren(TList *checked, TGListTreeItem *item)
{
   // Add all child items of 'item' into the list 'checked'.

   if (!checked || !item) return;

   while (item) {
      if (item->IsChecked()) {
         checked->Add(new TObjString(item->GetText()));
      }
      if (item->GetFirstChild()) {
         GetCheckedChildren(checked, item->GetFirstChild());
      }
      item = item->GetNextSibling();
   }
}

//______________________________________________________________________________
void TGListTree::CheckAllChildren(TGListTreeItem *item, Bool_t state)
{
   // Check all child items of 'item' and 'item' itself according
   // to the state value: kTRUE means check all, kFALSE - uncheck all.
   
   if (item) 
      item->CheckAllChildren(state);
}

