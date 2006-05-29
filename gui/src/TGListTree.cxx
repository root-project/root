// @(#)root/gui:$Name:  $:$Id: TGListTree.cxx,v 1.52 2006/05/28 20:07:59 brun Exp $
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
#include "Riostream.h"


Pixel_t        TGListTree::fgGrayPixel = 0;
const TGFont  *TGListTree::fgDefaultFont = 0;
TGGC          *TGListTree::fgDrawGC = 0;
TGGC          *TGListTree::fgLineGC = 0;
TGGC          *TGListTree::fgHighlightGC = 0;
TGGC          *TGListTree::fgColorGC = 0;


ClassImp(TGListTreeItem)
ClassImp(TGListTree)


//--- Some utility functions ---------------------------------------------------
static Int_t FontHeight(FontStruct_t f)
{
   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(f, max_ascent, max_descent);
   return max_ascent + max_descent;
}

static Int_t FontAscent(FontStruct_t f)
{
   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(f, max_ascent, max_descent);
   return max_ascent;
}

static Int_t FontTextWidth(FontStruct_t f, const char *c)
{
   return gVirtualX->TextWidth(f, c, strlen(c));
}


//______________________________________________________________________________
TGListTreeItem::TGListTreeItem(TGClient *client, const char *name,
                               const TGPicture *opened,
                               const TGPicture *closed,
                               Bool_t checkbox)
{
   // Create list tree item.

   fClient = client;

   fText = name;
   fCheckBox = checkbox;
   fChecked = kTRUE;

   if (!opened)
      opened = fClient->GetPicture("ofolder_t.xpm");
   else
      ((TGPicture *)opened)->AddReference();

   if (!closed)
      closed = fClient->GetPicture("folder_t.xpm");
   else
      ((TGPicture *)closed)->AddReference();

   fOpenPic   = opened;
   fClosedPic = closed;

   fCheckedPic   = fClient->GetPicture("checked_t.xpm");
   fUncheckedPic = fClient->GetPicture("unchecked_t.xpm");

   if (fCheckBox)
      fPicWidth  = TMath::Max(fCheckedPic->GetWidth() + fOpenPic->GetWidth(),
                              fCheckedPic->GetWidth() + fClosedPic->GetWidth());
   else
      fPicWidth  = TMath::Max(fOpenPic->GetWidth(), fClosedPic->GetWidth());

   fOpen = fActive = kFALSE;

   fParent =
   fFirstchild =
   fPrevsibling =
   fNextsibling = 0;
   fY =
   fHeight = 0;

   fUserData = 0;

   fHasColor = kFALSE;
   fColor = 0;
}

//______________________________________________________________________________
TGListTreeItem::TGListTreeItem(const TGListTreeItem& lti) :
  fClient(lti.fClient),
  fParent(lti.fParent),
  fFirstchild(lti.fFirstchild),
  fPrevsibling(lti.fPrevsibling),
  fNextsibling(lti.fNextsibling),
  fOpen(lti.fOpen),
  fActive(lti.fActive),
  fCheckBox(lti.fCheckBox),
  fChecked(lti.fChecked),
  fText(lti.fText),
  fTipText(lti.fTipText),
  fY(lti.fY),
  fXtext(lti.fXtext),
  fYtext(lti.fYtext),
  fHeight(lti.fHeight),
  fPicWidth(lti.fPicWidth),
  fOpenPic(lti.fOpenPic),
  fClosedPic(lti.fClosedPic),
  fCheckedPic(lti.fCheckedPic),
  fUncheckedPic(lti.fUncheckedPic),
  fUserData(lti.fUserData),
  fHasColor(lti.fHasColor),
  fColor(lti.fColor)
{ 
   //copy constructor
}

//______________________________________________________________________________
TGListTreeItem& TGListTreeItem::operator=(const TGListTreeItem& lti)
{
   //equal operator
   if(this!=&lti) {
      fClient=lti.fClient;
      fParent=lti.fParent;
      fFirstchild=lti.fFirstchild;
      fPrevsibling=lti.fPrevsibling;
      fNextsibling=lti.fNextsibling;
      fOpen=lti.fOpen;
      fActive=lti.fActive;
      fCheckBox=lti.fCheckBox;
      fChecked=lti.fChecked;
      fText=lti.fText;
      fTipText=lti.fTipText;
      fY=lti.fY;
      fXtext=lti.fXtext;
      fYtext=lti.fYtext;
      fHeight=lti.fHeight;
      fPicWidth=lti.fPicWidth;
      fOpenPic=lti.fOpenPic;
      fClosedPic=lti.fClosedPic;
      fCheckedPic=lti.fCheckedPic;
      fUncheckedPic=lti.fUncheckedPic;
      fUserData=lti.fUserData;
      fHasColor=lti.fHasColor;
      fColor=lti.fColor;
   } 
   return *this;
}

//______________________________________________________________________________
TGListTreeItem::~TGListTreeItem()
{
   // Delete list tree item.

   fClient->FreePicture(fOpenPic);
   fClient->FreePicture(fClosedPic);
   fClient->FreePicture(fCheckedPic);
   fClient->FreePicture(fUncheckedPic);
}

//______________________________________________________________________________
void TGListTreeItem::Rename(const char *new_name)
{
   // Rename a list tree item.

   fText = new_name;
}
//______________________________________________________________________________
void TGListTreeItem::SetCheckBox(Bool_t on)
{
   // Set a check box on the tree node.

   fCheckBox = on;
   if (fCheckBox)
      fPicWidth  = TMath::Max(fCheckedPic->GetWidth() + fOpenPic->GetWidth(),
                              fCheckedPic->GetWidth() + fClosedPic->GetWidth());
   else
      fPicWidth  = TMath::Max(fOpenPic->GetWidth(), fClosedPic->GetWidth());
}

//___________________________________________________________________________
void TGListTreeItem::SetCheckBoxPictures(const TGPicture *checked,
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
void TGListTreeItem::SetPictures(const TGPicture *opened, const TGPicture *closed)
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
   fDisableOpen = kFALSE;

   fGrayPixel   = GetGrayPixel();
   fFont        = GetDefaultFontStruct();

   fDrawGC      = GetDrawGC()();
   fLineGC      = GetLineGC()();
   fHighlightGC = GetHighlightGC()();
   fColorGC     = GetColorGC()();

   fFirst = fSelected = 0;
   fDefw = fDefh = 1;

   fHspacing = 2;
   fVspacing = 2;  // 0;
   fIndent   = 3;  // 0;
   fMargin   = 2;

   fColorMode = kDefault;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone);

   AddInput(kPointerMotionMask | kEnterWindowMask |
            kLeaveWindowMask | kKeyPressMask);
   SetWindowName();

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
   fDisableOpen = kFALSE;

   fGrayPixel   = GetGrayPixel();
   fFont        = GetDefaultFontStruct();

   fDrawGC      = GetDrawGC()();
   fLineGC      = GetLineGC()();
   fHighlightGC = GetHighlightGC()();
   fColorGC     = GetColorGC()();

   fFirst = fSelected = 0;
   fDefw = fDefh = 1;

   fHspacing = 2;
   fVspacing = 2;  // 0;
   fIndent   = 3;  // 0;
   fMargin   = 2;

   fColorMode = kDefault;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone);

   AddInput(kPointerMotionMask | kEnterWindowMask |
            kLeaveWindowMask | kKeyPressMask);
   SetWindowName();

   fEditDisabled = kEditDisable | kEditDisableGrab | kEditDisableBtnEnable;
}

//______________________________________________________________________________
TGListTree::TGListTree(const TGListTree& lt) : 
  TGContainer(lt),
  fFirst(lt.fFirst),
  fSelected(lt.fSelected),
  fHspacing(lt.fHspacing),
  fVspacing(lt.fVspacing),
  fIndent(lt.fIndent),
  fMargin(lt.fMargin),
  fLastY(lt.fLastY),
  fGrayPixel(lt.fGrayPixel),
  fDrawGC(lt.fDrawGC),
  fLineGC(lt.fLineGC),
  fHighlightGC(lt.fHighlightGC),
  fFont(lt.fFont),
  fDefw(lt.fDefw),
  fDefh(lt.fDefh),
  fExposeTop(lt.fExposeTop),
  fExposeBottom(lt.fExposeBottom),
  fTip(lt.fTip),
  fTipItem(lt.fTipItem),
  fAutoTips(lt.fAutoTips),
  fDisableOpen(lt.fDisableOpen),
  fColorMode(lt.fColorMode),
  fColorGC(lt.fColorGC)
{ 
   //copy constructor
}

//______________________________________________________________________________
TGListTree& TGListTree::operator=(const TGListTree& lt) 
{
   //equal operator
   if(this!=&lt) {
      TGContainer::operator=(lt);
      fFirst=lt.fFirst;
      fSelected=lt.fSelected;
      fHspacing=lt.fHspacing;
      fVspacing=lt.fVspacing;
      fIndent=lt.fIndent;
      fMargin=lt.fMargin;
      fLastY=lt.fLastY;
      fGrayPixel=lt.fGrayPixel;
      fDrawGC=lt.fDrawGC;
      fLineGC=lt.fLineGC;
      fHighlightGC=lt.fHighlightGC;
      fFont=lt.fFont;
      fDefw=lt.fDefw;
      fDefh=lt.fDefh;
      fExposeTop=lt.fExposeTop;
      fExposeBottom=lt.fExposeBottom;
      fTip=lt.fTip;
      fTipItem=lt.fTipItem;
      fAutoTips=lt.fAutoTips;
      fDisableOpen=lt.fDisableOpen;
      fColorMode=lt.fColorMode;
      fColorGC=lt.fColorGC;
   } 
   return *this;
}

//______________________________________________________________________________
TGListTree::~TGListTree()
{
   // Delete list tree widget.

   TGListTreeItem *item, *sibling;

   delete fTip;

   item = fFirst;
   while (item) {
      if (item->fFirstchild)
         PDeleteChildren(item->fFirstchild);
      sibling = item->fNextsibling;
      delete item;
      item = sibling;
   }
}


//---- highlighting utilities

//______________________________________________________________________________
void TGListTree::HighlightItem(TGListTreeItem *item, Bool_t state, Bool_t draw)
{
   // Highlight tree item.

   if (item) {
      if ((item == fSelected) && !state) {
         fSelected = 0;
         if (draw) DrawItemName(item);
      } else if (state != item->fActive) {
         item->fActive = state;
         if (draw) DrawItemName(item);
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
         HighlightChildren(item->fFirstchild, state, (item->fOpen) ? draw : kFALSE);
      item = item->fNextsibling;
   }
}

//______________________________________________________________________________
void TGListTree::UnselectAll(Bool_t draw)
{
   // Unselect all items.

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
      if (fCanvas->GetContainer()->GetHeight())
         page = Int_t(Float_t(fCanvas->GetViewPort()->GetHeight() *
                              fCanvas->GetViewPort()->GetHeight()) /
                              fCanvas->GetContainer()->GetHeight());
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
            if (item->HasCheckBox()) {
               minxchk = (item->fXtext - Int_t(item->fPicWidth));
               maxxchk = (item->fXtext - Int_t(item->fPicWidth)) +
                       Int_t(item->fCheckedPic->GetWidth()) - 4;
               maxx = maxxchk - 8;
               minx = minxchk - 16;
            }
            else {
               maxx = (item->fXtext - Int_t(item->fPicWidth)) - 8;
               minx = (item->fXtext - Int_t(item->fPicWidth)) - 16;
            }
            if ((item->HasCheckBox()) && (event->fX < maxxchk) &&
               (event->fX > minxchk)) {
               fLastY = event->fY;
               ToggleItem(item);
               UpdateChecked(item, kTRUE);
               Checked((TObject *)item->GetUserData(), item->IsChecked());
               return kTRUE;
            }
            if ((event->fX < maxx) && (event->fX > minx)) {
               ClearViewPort();
               item->fOpen = !item->fOpen;
               fClient->NeedRedraw(this);
               return kTRUE;
            }
         }
         if (fSelected) fSelected->fActive = kFALSE;
         fLastY = event->fY;
         UnselectAll(kTRUE);
         fSelected = item;
         //item->fActive = kTRUE; // this is done below w/redraw
         HighlightItem(item, kTRUE, kTRUE);
         SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);
         Clicked(item, event->fCode);
         Clicked(item, event->fCode, event->fXRoot, event->fYRoot);
      }
   }
   if (event->fType == kButtonRelease) gVirtualX->SetInputFocus(fId);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleDoubleClick(Event_t *event)
{
   // Handle double click event in the list tree (only for kButton1).

   TGListTreeItem *item;

   // If fDisableOpen is set, only send message and emit signals.
   // It allows user to customize handling of double click events.
   if (fDisableOpen && event->fCode == kButton1 && (item = FindItem(event->fY)) != 0) {
      SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMDBLCLICK),
                  event->fCode, (event->fYRoot << 16) | event->fXRoot);
      DoubleClicked(item, event->fCode);
      DoubleClicked(item, event->fCode, event->fXRoot, event->fYRoot);
      return kTRUE;
   }
   // Otherwise, just use default behaviour (open item).
   if (event->fCode == kButton1 && (item = FindItem(event->fY)) != 0) {
      ClearViewPort();
      item->fOpen = !item->fOpen;
      if (item != fSelected) { // huh?!
         if (fSelected) fSelected->fActive = kFALSE;
         UnselectAll(kTRUE);
         fSelected = item;
         //item->fActive = kTRUE; // this is done below w/redraw
         HighlightItem(item, kTRUE, kTRUE);
      }
      SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMDBLCLICK),
                  event->fCode, (event->fYRoot << 16) | event->fXRoot);
      DoubleClicked(item, event->fCode);
      DoubleClicked(item, event->fCode, event->fXRoot, event->fYRoot);
      fClient->NeedRedraw(this);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleExpose(Event_t * /*event*/)
{
   // Handle expose event in the list tree.

   Draw(0, (Int_t)fHeight);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kLeaveNotify) {
         fTip->Hide();
         fTipItem = 0;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleMotion(Event_t *event)
{
   // Handle mouse motion event. Only used to set tool tip.

   TGListTreeItem *item;
   fOnMouseOver = kFALSE;
   TGPosition pos = GetPagePosition();

   if ((item = FindItem(event->fY)) != 0) {

      if (item->HasCheckBox()) {
         if ((event->fX < (item->fXtext - Int_t(item->fPicWidth)) +
                           Int_t(item->fCheckedPic->GetWidth()) - 4) &&
             (event->fX > (item->fXtext - Int_t(item->fPicWidth)))) {
            gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kPointer));
            return kTRUE;
         }
         else {
            gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));
         }
      }
      if (fTipItem == item) return kTRUE;

      OnMouseOver(item);
      gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));

      if (fTip)
         fTip->Hide();

      if (item->fTipText.Length() > 0) {

         SetToolTipText(item->fTipText.Data(), item->fXtext,
                        item->fY -pos.fY +item->fHeight -4, 1000);

      } else if (fAutoTips && item->GetUserData()) {
         // must derive from TObject (in principle user can put pointer
         // to anything in user data field). Add check.
         TObject *obj = (TObject *)item->GetUserData();
         if (obj->InheritsFrom(TObject::Class())) {
            SetToolTipText(obj->GetTitle(), item->fXtext,
                           item->fY -pos.fY +item->fHeight -4, 1000);
         }
      }
      fTipItem = item;
   } else {
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
   Int_t  n;
   UInt_t keysym;

   if (event->fType == kGKeyPress) {
      gVirtualX->LookupString(event, input, sizeof(input), keysym);
      n = strlen(input);

      KeyPressed(fSelected, keysym, event->fState);

      switch ((EKeySym)keysym) {
         case kKey_Enter:
         case kKey_Return:
            // treat 'Enter' and 'Return' as a double click
            if (fSelected) fSelected->fOpen = !fSelected->fOpen;
            SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMDBLCLICK),
                        kButton1, (event->fYRoot << 16) | event->fXRoot);

            if (fSelected) ReturnPressed(fSelected);
            break;
         default:
         break;
      }

      if (event->fState & kKeyControlMask) {   // Cntrl key modifier pressed
         switch((EKeySym)keysym & ~0x20) {   // treat upper and lower the same
            case kKey_A:
//               SelectAll();
               break;
            case kKey_B:
//               LineLeft();
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
//               RepeatSearch();
               break;
            case kKey_H:
//               LineLeft();
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

//         if (fKeyTimerActive) {
//            fKeyInput += input;
//         } else {
//            fKeyInput = input;
//            fKeyTimerActive = kTRUE;
//            fKeyTimer->Reset();
//            if (gSystem) gSystem->AddTimer(fKeyTimer);
//         }
      } else {

         switch ((EKeySym)keysym) {
            case kKey_F3:
//               RepeatSearch();
               break;
            case kKey_F5:
               Layout();
               break;
            case kKey_F7:
               Search();
               break;
            case kKey_Left:
//               LineLeft(event->fState & kKeyShiftMask);
               break;
            case kKey_Right:
//               LineRight(event->fState & kKeyShiftMask);
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
//   fClient->NeedRedraw(this);
   return kTRUE;
}

//______________________________________________________________________________
void TGListTree::OnMouseOver(TGListTreeItem *entry)
{
   // Signal emitted when pointer is over entry.

   if (!fOnMouseOver) Emit("OnMouseOver(TGListTreeItem*)", (Long_t)entry);
   fOnMouseOver = kTRUE;
}

//______________________________________________________________________________
void TGListTree::KeyPressed(TGListTreeItem *entry, UInt_t keysym, UInt_t mask)
{
   // Signal emitted when keyboard key pressed
   //
   // item - selected item
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
   Emit("KeyPressed(TGFame*,ULong_t,ULong_t)", args);
   SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_KEY), keysym, mask);
}

//______________________________________________________________________________
void TGListTree::ReturnPressed(TGListTreeItem *entry)
{
   // Emit ReturnPressed() signal.

   Emit("ReturnPressed(TGListTreeItem*)", (Long_t)entry);
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
void TGListTree::Checked(TObject *entry, Bool_t on)
{
   // Emit Checked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = on;

   Emit("Checked(TObject*,Bool_t)", args);
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

   if (!fCanvas || !fSelected) return;

   TGDimension dim = GetPageDimension();

   Int_t newpos = fCanvas->GetVsbPosition() - dim.fHeight;
   if (newpos<0) newpos = 0;

   fCanvas->SetVsbPosition(newpos);
}

//______________________________________________________________________________
void TGListTree::PageDown(Bool_t /*select*/)
{
   // Move content one page down.

   if (!fCanvas || !fSelected) return;

   TGDimension dim = GetPageDimension();

   Int_t newpos = fCanvas->GetVsbPosition() + dim.fHeight;

   fCanvas->SetVsbPosition(newpos);
}

//______________________________________________________________________________
void TGListTree::LineUp(Bool_t /*select*/)
{
   // Move content one item-size up.

   if (!fCanvas || !fSelected) return;

   const TGPicture *pic1, *pic2;
   Int_t height;

   if (fSelected->fOpen)
      pic2 = fSelected->fOpenPic;
   else
      pic2 = fSelected->fClosedPic;

   if (fSelected->HasCheckBox()){
      if (fSelected->IsChecked())
         pic1 = fSelected->fOpenPic;
      else
         pic1 = fSelected->fClosedPic;
      height  = TMath::Max(pic2->GetHeight() + fVspacing, pic1->GetHeight() + fVspacing);
   }
   else {
      height = pic2->GetHeight() + fVspacing;
   }

   Int_t newpos = fCanvas->GetVsbPosition() - height;
   if (newpos<0) newpos = 0;

   fCanvas->SetVsbPosition(newpos);
}

//______________________________________________________________________________
void TGListTree::LineDown(Bool_t /*select*/)
{
   // Move content one item-size down.

   if (!fCanvas || !fSelected) return;

   const TGPicture *pic1, *pic2;
   Int_t height;

   if (fSelected->fOpen)
      pic2 = fSelected->fOpenPic;
   else
      pic2 = fSelected->fClosedPic;

   if (fSelected->HasCheckBox()){
      if (fSelected->IsChecked())
         pic1 = fSelected->fOpenPic;
      else
         pic1 = fSelected->fClosedPic;
      height  = TMath::Max(pic2->GetHeight() + fVspacing, pic1->GetHeight() + fVspacing);
   }
   else {
      height = pic2->GetHeight() + fVspacing;
   }

   Int_t newpos = fCanvas->GetVsbPosition() + height;
   if (newpos<0) newpos = 0;

   fCanvas->SetVsbPosition(newpos);
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
void TGListTree::Search(Bool_t /*close*/)
{
   // Invokes search dialog. Looks for item with the entered name.

   Int_t ret = 0;
   char msg[256];
   static TString buf;

   TGSearchType *srch = new TGSearchType;
   srch->fBuffer = (char*)buf.Data();

   TGListTreeItem *item;
   new TGSearchDialog(fClient->GetDefaultRoot(), fCanvas, 400, 150, srch, &ret);

   if (ret) {
      item = FindItemByPathname(srch->fBuffer);
      if (!item) {
         sprintf(msg, "Couldn't find \"%s\"", srch->fBuffer);
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
void TGListTree::DrawRegion(Int_t /*x*/, Int_t /*y*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Redraw list tree.

   Draw(0, (Int_t)fHeight);
}

//______________________________________________________________________________
void TGListTree::DoRedraw()
{
   // Redraw list tree.

   ClearViewPort();
   Draw(0, (Int_t)fHeight);
}

//______________________________________________________________________________
void TGListTree::Draw(Int_t yevent, Int_t hevent)
{
   // Draw list tree widget.

   TGListTreeItem *item;
   Int_t  x, y, xbranch;
   UInt_t width, height, old_width, old_height;

   // Overestimate the expose region to be sure to draw an item that gets
   // cut by the region
   fExposeTop = yevent - FontHeight(fFont);
   fExposeBottom = yevent + hevent + FontHeight(fFont);
   old_width  = fDefw;
   old_height = fDefh;
   fDefw = fDefh = 1;

   TGPosition pos = GetPagePosition();
   x = -pos.fX;
   y = fMargin + yevent;
   item = fFirst;

   while (item) {
      xbranch = -1;

      DrawItem(item, x, y, &xbranch, &width, &height);

      width += pos.fX + x + fHspacing + fMargin;

      if (width > fDefw) fDefw = width;

      y += height + fVspacing;
      if (item->fFirstchild && item->fOpen)
         y = DrawChildren(item->fFirstchild, x, y, xbranch);

      item = item->fNextsibling;
   }

   fDefh = y + fMargin;

   if ((old_width != fDefw) || (old_height != fDefh)) {
      fCanvas->Layout();
   }
}

//______________________________________________________________________________
Int_t TGListTree::DrawChildren(TGListTreeItem *item, Int_t x, Int_t y, Int_t xroot)
{
   // Draw children of item in list tree.

   UInt_t width, height;
   Int_t  xbranch;
   TGPosition pos = GetPagePosition();

   x += fIndent + (Int_t)item->fParent->fPicWidth;
   while (item) {
      xbranch = xroot;

      DrawItem(item, x, y, &xbranch, &width, &height);

      width += pos.fX + x + fHspacing + fMargin;
      if (width > fDefw) fDefw = width;

      y += height + fVspacing;
      if ((item->fFirstchild) && (item->fOpen))
         y = DrawChildren(item->fFirstchild, x, y, xbranch);

      item = item->fNextsibling;
   }
   return y;
}

//______________________________________________________________________________
void TGListTree::DrawItem(TGListTreeItem *item, Int_t x, Int_t y, Int_t *xroot,
                          UInt_t *retwidth, UInt_t *retheight)
{
   // Draw list tree item.

   Int_t  xpic2, ypic2, xbranch, ybranch, xtext, ytext, xline, yline, xc;
   Int_t  xpic1 = 0, ypic1 = 0;
   UInt_t height;
   const TGPicture *pic1 = 0, *pic2;

   TGPosition pos = GetPagePosition();
   TGDimension dim = GetPageDimension();

   // Select the pixmap to use, if any
   if (item->fOpen)
      pic2 = item->fOpenPic;
   else
      pic2 = item->fClosedPic;

   if (item->HasCheckBox()) {
      if (item->IsChecked())
         pic1 = item->fCheckedPic;
      else
         pic1 = item->fUncheckedPic;
   }

   // Compute the height of this line
   height = FontHeight(fFont);
   xpic2 = x;
   xtext = x + fHspacing + (Int_t)item->fPicWidth;
   if ((item->HasCheckBox()) && (pic1)) {
      if (pic1->GetHeight() > height) {
         ytext = y + (Int_t)((pic1->GetHeight() - height) >> 1);
         height = pic1->GetHeight();
         ypic1 = y;
      } else {
         ytext = y;
         ypic1 = y + (Int_t)((height - pic1->GetHeight()) >> 1);
      }
      xpic1 = x;
      xpic2 = xpic1 + pic1->GetWidth();
      xline = xpic1;
   } else {
      xpic2 = x;
      ypic2 = y;
      xline = 0;
   }
   if (pic2) {
      if (pic2->GetHeight() > height) {
         ytext = y + (Int_t)((pic2->GetHeight() - height) >> 1);
         height = pic2->GetHeight();
         ypic2 = y;
      } else {
         ytext = y;
         ypic2 = y + (Int_t)((height - pic2->GetHeight()) >> 1);
      }
      xbranch = xpic2 + (Int_t)(pic2->GetWidth() >> 1);
      ybranch = ypic2 + (Int_t)pic2->GetHeight();
      yline = ypic2 + (Int_t)(pic2->GetHeight() >> 1);
      if (xline == 0) xline = xpic2;
   } else {
      if (xline == 0) xline = xpic2;
      ypic2 = ytext = y;
      xbranch = xpic2 + (Int_t)(item->fPicWidth >> 1);
      yline = ybranch = ypic2 + (Int_t)(height >> 1);
      yline = ypic2 + (Int_t)(height >> 1);
   }

   // height must be even, otherwise our dashed line wont appear properly
   ++height; height &= ~1;

   // Save the basic graphics info for use by other functions
   item->fY      = y;
   item->fXtext  = xtext;
   item->fYtext  = ytext;
   item->fHeight = height;

   // projected coordinates
   Int_t yp = y - pos.fY;
   Int_t ylinep = yline - pos.fY;
   Int_t ybranchp = ybranch - pos.fY;
   Int_t ypicp = ypic2 - pos.fY;

   if ((yp >= fExposeTop) && (yp <= (Int_t)dim.fHeight)) {

      if (*xroot >= 0) {
         xc = *xroot;

         if (item->fNextsibling)
            gVirtualX->DrawLine(fId, fLineGC, xc, yp, xc, yp+height);
         else
            gVirtualX->DrawLine(fId, fLineGC, xc, yp, xc, ylinep);

         TGListTreeItem *p = item->fParent;
         while (p) {
            xc -= (fIndent + (Int_t)item->fPicWidth);
            if (p->fNextsibling)
               gVirtualX->DrawLine(fId, fLineGC, xc, yp, xc, yp+height);
            p = p->fParent;
         }
         gVirtualX->DrawLine(fId, fLineGC, *xroot, ylinep, xline /*xpic2*/ /*xbranch*/, ylinep);
         DrawNode(item, *xroot, yline);
      }
      if (item->fOpen && item->fFirstchild)
         gVirtualX->DrawLine(fId, fLineGC, xbranch, ybranchp/*yline*/,
                        xbranch, yp+height);

      // if (pic2)
      //    pic2->Draw(fId, fDrawGC, xpic2, ypic2);
      if (item->HasCheckBox()) {
         if (item->IsChecked())
            item->fCheckedPic->Draw(fId, fDrawGC, xpic1, ypicp);
         else
            item->fUncheckedPic->Draw(fId, fDrawGC, xpic1, ypicp);
      }
      if (item->fActive || item == fSelected)
         item->fOpenPic->Draw(fId, fDrawGC, xpic2, ypicp);
      else
         item->fClosedPic->Draw(fId, fDrawGC, xpic2, ypicp);

      DrawItemName(item);
   }

   *xroot = xbranch;
   *retwidth = FontTextWidth(fFont, item->fText.Data()) + item->fPicWidth;
   *retheight = height;
}

//______________________________________________________________________________
void TGListTree::DrawItemName(TGListTreeItem *item)
{
   // Draw name of list tree item.

   UInt_t width;
   TGPosition pos = GetPagePosition();

   width = FontTextWidth(fFont, item->fText.Data());
   if (item->fActive || item == fSelected) {
      gVirtualX->SetForeground(fDrawGC, fgDefaultSelectedBackground);
      gVirtualX->FillRectangle(fId, fDrawGC,
                          item->fXtext, item->fYtext-pos.fY, width, FontHeight(fFont));
      gVirtualX->SetForeground(fDrawGC, fgBlackPixel);
      gVirtualX->DrawString(fId, fHighlightGC,
                       item->fXtext, item->fYtext - pos.fY + FontAscent(fFont),
                       item->fText.Data(), item->fText.Length());
   } else {
      gVirtualX->FillRectangle(fId, fHighlightGC,
                          item->fXtext, item->fYtext-pos.fY, width, FontHeight(fFont));
      gVirtualX->DrawString(fId, fDrawGC,
                       item->fXtext, item->fYtext-pos.fY + FontAscent(fFont),
                       item->fText.Data(), item->fText.Length());
   }

   if (fColorMode != 0 && item->fHasColor) {
      gVirtualX->SetForeground(fColorGC, TColor::Number2Pixel(item->fColor));
      if (fColorMode | kColorUnderline) {
         Int_t y = item->fYtext-pos.fY + FontAscent(fFont) + 2;
         gVirtualX->DrawLine(fId, fColorGC, item->fXtext, y, item->fXtext + width, y);
      }
      if (fColorMode | kColorBox) {
         Int_t x = item->fXtext + width + 4;
         Int_t y = item->fYtext-pos.fY  + 3;
         Int_t h = FontAscent(fFont)    - 4;
         gVirtualX->FillRectangle(fId, fColorGC, x, y, h, h);
         gVirtualX->DrawRectangle(fId, fDrawGC,  x, y, h, h);
      }
   }
}

//______________________________________________________________________________
void TGListTree::DrawNode(TGListTreeItem *item, Int_t x, Int_t y)
{
   // Draw node (little + in box).

   TGPosition pos = GetPagePosition();
   Int_t yp = y - pos.fY;

   if (item->fFirstchild) {
      gVirtualX->DrawLine(fId, fHighlightGC, x, yp-2, x, yp+2);
      gVirtualX->SetForeground(fHighlightGC, fgBlackPixel);
      gVirtualX->DrawLine(fId, fHighlightGC, x-2, yp, x+2, yp);
      if (!item->fOpen)
         gVirtualX->DrawLine(fId, fHighlightGC, x, yp-2, x, yp+2);
      gVirtualX->SetForeground(fHighlightGC, fGrayPixel);
      gVirtualX->DrawLine(fId, fHighlightGC, x-4, yp-4, x+4, yp-4);
      gVirtualX->DrawLine(fId, fHighlightGC, x+4, yp-4, x+4, yp+4);
      gVirtualX->DrawLine(fId, fHighlightGC, x-4, yp+4, x+4, yp+4);
      gVirtualX->DrawLine(fId, fHighlightGC, x-4, yp-4, x-4, yp+4);
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

   // if there exists a previous sibling, just skip over item to be dereferenced
   if (item->fPrevsibling) {
      item->fPrevsibling->fNextsibling = item->fNextsibling;
      if (item->fNextsibling)
         item->fNextsibling->fPrevsibling = item->fPrevsibling;
   } else {
      // if not, then the deleted item is the first item in some branch
      if (item->fParent)
         item->fParent->fFirstchild = item->fNextsibling;
      else
         fFirst = item->fNextsibling;
      if (item->fNextsibling)
         item->fNextsibling->fPrevsibling = 0;
   }
}

//______________________________________________________________________________
void TGListTree::PDeleteChildren(TGListTreeItem *item)
{
   // Delete children of item from list.

   TGListTreeItem *sibling;

   while (item) {
      if (item->fFirstchild) {
         PDeleteChildren(item->fFirstchild);
         item->fFirstchild = 0;
      }
      sibling = item->fNextsibling;
      if (fSelected == item)
         fSelected = 0;
      delete item;
      item = sibling;
   }
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
         i = parent->fFirstchild;
         while (i->fNextsibling) i = i->fNextsibling;
         i->fNextsibling = item;
         item->fPrevsibling = i;
      } else {
         parent->fFirstchild = item;
      }

   } else {  // if parent == 0, this is a top level entry

      if (fFirst) {
         i = fFirst;
         while (i->fNextsibling) i = i->fNextsibling;
         i->fNextsibling = item;
         item->fPrevsibling = i;
      } else {
         fFirst = item;
      }

   }
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
      // Select the pixmap to use, if any

      if (item->fOpen)
         pic = item->fOpenPic;
      else
         pic = item->fClosedPic;

      // Compute the height of this line
      height = FontHeight(fFont);
      if (pic && (pic->GetHeight() > height))
         height = pic->GetHeight();

      if ((findy >= y) && (findy <= y + (Int_t)height)) {
         *finditem = item;
         return -1;
      }

      y += (Int_t)height + fVspacing;
      if ((item->fFirstchild) && (item->fOpen)) {
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
      // Select the pixmap to use, if any
      if (item->fOpen)
         pic = item->fOpenPic;
      else
         pic = item->fClosedPic;

      // Compute the height of this line
      height = FontHeight(fFont);
      if (pic && (pic->GetHeight() > height))
         height = pic->GetHeight();

      if ((findy >= y) && (findy <= y + (Int_t)height))
         return item;

      y += (Int_t)height + fVspacing;
      if ((item->fFirstchild) && (item->fOpen)) {
         y = SearchChildren(item->fFirstchild, y, findy, &finditem);
         //if (finditem) return finditem;
      }
      item = item->fNextsibling;
   }

   return finditem;
}

//----- Public Functions

//______________________________________________________________________________
TGListTreeItem *TGListTree::AddItem(TGListTreeItem *parent, const char *string,
                                    const TGPicture *open, const TGPicture *closed,
                                    Bool_t checkbox)
{
   // Add item to list tree. Returns new item.

   TGListTreeItem *item;

   item = new TGListTreeItem(fClient, string, open, closed, checkbox);
   InsertChild(parent, item);

   if (fClient->IsEditable()) {
      if (parent)
         OpenItem(parent);
      fClient->NeedRedraw(this);
   }
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

   if (item)
      item->Rename(string);

   if (fClient->IsEditable()) {
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
Int_t TGListTree::DeleteItem(TGListTreeItem *item)
{
   // Delete item from list tree.

   if (item->fFirstchild)
      PDeleteChildren(item->fFirstchild);

   item->fFirstchild = 0;

   RemoveReference(item);

   if (fSelected == item)
      fSelected = 0;

   delete item;

   if (fClient->IsEditable()) {
      fClient->NeedRedraw(this);
   }

   return 1;
}

//______________________________________________________________________________
void TGListTree::OpenItem(TGListTreeItem *item)
{
   // Open item in list tree (i.e. show child items).

   if (item) {
      item->fOpen = kTRUE;
      AdjustPosition(item);
   }
}

//______________________________________________________________________________
void TGListTree::CloseItem(TGListTreeItem *item)
{
   // Close item in list tree (i.e. hide child items).

   if (item) {
      item->fOpen = kFALSE;
      AdjustPosition(item);
   }
}

//______________________________________________________________________________
Int_t TGListTree::RecursiveDeleteItem(TGListTreeItem *item, void *ptr)
{
   // Delete item with fUserData == ptr. Search tree downwards starting
   // at item.

   if (item && ptr) {
      if (item->fUserData == ptr)
         DeleteItem(item);
      else {
         if (item->fOpen && item->fFirstchild)
            RecursiveDeleteItem(item->fFirstchild,  ptr);
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

   if (item)
      item->fTipText = string;
}

//______________________________________________________________________________
Int_t TGListTree::DeleteChildren(TGListTreeItem *item)
{
   // Delete children of item from list.

   if (item->fFirstchild)
      PDeleteChildren(item->fFirstchild);

   item->fFirstchild = 0;

   //fClient->NeedRedraw(this);

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

   //fClient->NeedRedraw(this);

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

      //fClient->NeedRedraw(this);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
static Int_t Compare(const void *item1, const void *item2)
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
   if (parent)
      parent->fFirstchild = list[0];
   else
      fFirst = list[0];

   delete [] list;

   //fClient->NeedRedraw(this);

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
      if (first)
         SortSiblings(first);
   } else {
      if (fFirst) {
         first = fFirst->fFirstchild;
         if (first)
            SortSiblings(first);
      }
   }
   return 1;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindSiblingByName(TGListTreeItem *item, const char *name)
{
   // Find sibling of item by name.

   // Get first child in list
   if (item) {
      while (item->fPrevsibling)
         item = item->fPrevsibling;

      while (item) {
         if (item->fText == name) {
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
      while (item->fPrevsibling)
         item = item->fPrevsibling;

      while (item) {
         if (item->fUserData == userData) {
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
      if (item->fText == name) {
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
      if (item->fUserData == userData) {
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
   char dirname[256];
   TGListTreeItem *item = 0;
   item = FindChildByName(item, "/");
   TGListTreeItem *diritem = 0;
   TString fulldir;

start:
   while (1) {
      while (*p && *p == '/') p++;
      if (!*p) break;

      s = strchr(p, '/');

      if (!s) {
         strcpy(dirname, p);
      } else {
         strncpy(dirname, p, s-p);
         dirname[s-p] = 0;
      }

      item = FindChildByName(item, dirname);

      if (!diritem && dirname) {
         fulldir += "/";
         fulldir += dirname;

         if ((diritem=FindChildByName(0, fulldir.Data()))) {
            if (!s || !strlen(s)) return diritem;
            p = ++s;
            item = diritem;
            goto start;
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

   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGListTree::ClearHighlighted()
{
   // Un highlight items.

   UnselectAll(kFALSE);
   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGListTree::GetPathnameFromItem(TGListTreeItem *item, char *path, Int_t depth)
{
   // Get pathname from item. Use depth to limit path name to last
   // depth levels. By default depth is not limited.

   char tmppath[1024];

   *path = '\0';
   while (item) {
      sprintf(tmppath, "/%s%s", item->fText.Data(), path);
      strcpy(path, tmppath);
      item = item->fParent;
      if (--depth == 0 && item) {
         sprintf(tmppath, "...%s", path);
         strcpy(path, tmppath);
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
void TGListTree::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a list tree widget as a C++ statement(s) on output stream out.

   char quote = '"';

   if (fBackground != GetWhitePixel()) SaveUserColor(out, option);

   out << endl << "   // list tree" << endl;
   out << "   TGListTree *";

   if ((fParent->GetParent())->InheritsFrom(TGCanvas::Class())) {
      out << GetName() << " = new TGListTree(" << GetCanvas()->GetName();
   } else {
      out << GetName() << " = new TGListTree(" << fParent->GetName();
      out << "," << GetWidth() << "," << GetHeight();
   }

   if (fBackground == GetWhitePixel()) {
      if (GetOptions() == kSunkenFrame) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }

   out << "   " << GetName() << "->AddItem(0," << quote
       << GetFirstItem()->GetText() << quote << ");" << endl;
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

   Bool_t diff = kFALSE;
   TGListTreeItem *current = item;

   if (item->GetParent()) {
      current = item->GetParent()->GetFirstChild();
   }
   while (current) {
      TGListTreeItem *parent = current->GetParent();
      if ((parent) && (parent->HasCheckBox())) {
         if ( ((parent->IsChecked()) && (!current->IsChecked())) ||
              ((!parent->IsChecked()) && (current->IsChecked())) ) {
            diff = kTRUE;
            break;
         }
      }
      current = current->fNextsibling;
   }
   if ((item->GetParent()) && (item->GetParent()->HasCheckBox())) {
      if (diff) {
         item->GetParent()->SetCheckBoxPictures(fClient->GetPicture("checked_dis_t.xpm"),
                                                fClient->GetPicture("unchecked_dis_t.xpm"));
      }
      else {
         item->GetParent()->SetCheckBoxPictures(fClient->GetPicture("checked_t.xpm"),
                                                fClient->GetPicture("unchecked_t.xpm"));
      }
   }
   diff = kFALSE;
   current = item->GetFirstChild();
   while (current) {
      if (current->HasCheckBox()) {
         if ( ((current->IsChecked()) && (!item->IsChecked())) ||
              ((!current->IsChecked()) && (item->IsChecked())) ) {
            diff = kTRUE;
            break;
         }
      }
      current = current->GetNextSibling();
   }
   if (diff) {
      item->SetCheckBoxPictures(fClient->GetPicture("checked_dis_t.xpm"),
                                fClient->GetPicture("unchecked_dis_t.xpm"));
   }
   else {
      item->SetCheckBoxPictures(fClient->GetPicture("checked_t.xpm"),
                                fClient->GetPicture("unchecked_t.xpm"));
   }

   if (redraw)
      fClient->NeedRedraw(this);
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindItemByObj(TGListTreeItem *item, void *ptr)
{
   // Find item with fUserData == ptr. Search tree downwards starting
   // at item.

   TGListTreeItem *fitem;
   if (item && ptr) {
      if (item->fUserData == ptr)
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

