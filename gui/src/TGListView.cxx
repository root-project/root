// @(#)root/gui:$Name:  $:$Id: TGListView.cxx,v 1.5 2000/10/12 16:53:38 rdm Exp $
// Author: Fons Rademakers   17/01/98

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
// TGListView, TGLVContainer and TGLVEntry                              //
//                                                                      //
// A list view is a widget that can contain a number of items           //
// arranged in a grid or list. The items can be represented either      //
// by a string or by an icon.                                           //
//                                                                      //
// The TGListView is user callable. The other classes are service       //
// classes of the list view.                                            //
//                                                                      //
// A list view can generate the following events:                       //
// kC_CONTAINER, kCT_SELCHANGED, total items, selected items.           //
// kC_CONTAINER, kCT_ITEMCLICK, which button, location (y<<16|x).       //
// kC_CONTAINER, kCT_ITEMDBLCLICK, which button, location (y<<16|x).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGListView.h"
#include "TGPicture.h"
#include "TGButton.h"
#include "TGScrollBar.h"
#include "TList.h"
#include "TMath.h"


ClassImp(TGLVEntry)
ClassImp(TGLVContainer)
ClassImpQ(TGListView)

//______________________________________________________________________________
TGLVEntry::TGLVEntry(const TGWindow *p, const TGPicture *bigpic,
                     const TGPicture *smallpic, TGString *name,
                     TGString **subnames, EListViewMode viewMode,
                     UInt_t options, ULong_t back) :
   TGFrame(p, 10, 10, options, back)
{
   // Create a list view item.

   int i;

   fSelPic = 0;

   fCurrent  =
   fBigPic   = bigpic;
   fSmallPic = smallpic;

   fName = name;
   fSubnames = subnames;
   fUserData = 0;

   fCpos  =
   fJmode = 0;

   fActive = kFALSE;

   fNormGC     = fgDefaultGC();
   fFontStruct = fgDefaultFontStruct;

   int max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fName->GetString(), fName->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   if (fSubnames) {
      for (i = 0; fSubnames[i] != 0; ++i)
         ;
      fCtw = new int[i];
      for (i = 0; fSubnames[i] != 0; ++i)
         fCtw[i] = gVirtualX->TextWidth(fFontStruct, fSubnames[i]->GetString(),
                                        fSubnames[i]->GetLength());
   } else {
      fCtw = 0;
   }

   fViewMode = (EListViewMode)-1;
   SetViewMode(viewMode);
}

//______________________________________________________________________________
TGLVEntry::~TGLVEntry()
{
   // Delete a list view item.

   int i;

   if (fName) delete fName;
   if (fSelPic) delete fSelPic;
   if (fSubnames) {
      for (i = 0; fSubnames[i] != 0; ++i) delete fSubnames[i];
      delete [] fSubnames;
      delete [] fCtw;
   }
}

//______________________________________________________________________________
void TGLVEntry::Activate(Bool_t a)
{
   // Make list view item active.

   if (fActive == a) return;
   fActive = a;

   if (fActive) {
      fSelPic = new TGSelectedPicture(fClient, fCurrent);
   } else {
      if (fSelPic) delete fSelPic;
      fSelPic = 0;
   }
   DoRedraw();
}

//______________________________________________________________________________
void TGLVEntry::SetViewMode(EListViewMode viewMode)
{
   // Set the view mode for this list item.

   if (viewMode != fViewMode) {
      fViewMode = viewMode;
      if (viewMode == kLVLargeIcons)
         fCurrent = fBigPic;
      else
         fCurrent = fSmallPic;
      if (fActive) {
         if (fSelPic) delete fSelPic;
         fSelPic = new TGSelectedPicture(fClient, fCurrent);
      }
      gVirtualX->ClearWindow(fId);
      Resize(GetDefaultSize());
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
void TGLVEntry::DoRedraw()
{
   // Redraw list view item.

   int ix, iy, lx, ly;

   if (fViewMode == kLVLargeIcons) {
      ix = (fWidth - fCurrent->GetWidth()) >> 1;
      iy = 0;
      lx = (fWidth - fTWidth) >> 1;
      ly = fHeight - (fTHeight+1) - 2;
   } else {
      ix = 0;
      iy = (fHeight - fCurrent->GetHeight()) >> 1;
      lx = fCurrent->GetWidth() + 2;
      ly = (fHeight - (fTHeight+1)) >> 1;
   }

   if (fActive) {
      if (fSelPic) fSelPic->Draw(fId, fNormGC, ix, iy);
      gVirtualX->SetForeground(fNormGC, fgDefaultSelectedBackground);
      gVirtualX->FillRectangle(fId, fNormGC, lx, ly, fTWidth, fTHeight+1);
      gVirtualX->SetForeground(fNormGC, fgSelPixel);
   } else {
      fCurrent->Draw(fId, fNormGC, ix, iy);
      gVirtualX->SetForeground(fNormGC, fgWhitePixel);
      gVirtualX->FillRectangle(fId, fNormGC, lx, ly, fTWidth, fTHeight+1);
      gVirtualX->SetForeground(fNormGC, fgBlackPixel);
   }

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fName->Draw(fId, fNormGC, lx, ly + max_ascent);

   gVirtualX->SetForeground(fNormGC, fgBlackPixel);

   if (fViewMode == kLVDetails) {
      if (fSubnames && fCpos && fJmode && fCtw) {
         int i;

         for (i = 0; fSubnames[i] != 0; ++i) {
            if (fJmode[i] == kTextRight)
               lx = fCpos[i+1] - fCtw[i] - 2;
            else if (fJmode[i] == kTextCenterX)
               lx = (fCpos[i] + fCpos[i+1] - fCtw[i]) >> 1;
            else // default to TEXT_LEFT
               lx = fCpos[i] + 2;
            fSubnames[i]->Draw(fId, fNormGC, lx, ly + max_ascent);
         }
      }
   }
}

//______________________________________________________________________________
TGDimension TGLVEntry::GetDefaultSize() const
{
   // Get default size of list item.

   TGDimension size;
   TGDimension isize(fCurrent->GetWidth(), fCurrent->GetHeight());
   TGDimension lsize(fTWidth, fTHeight+1);

   switch (fViewMode) {
      default:
      case kLVLargeIcons:
         size.fWidth = TMath::Max(isize.fWidth, lsize.fWidth);
         size.fHeight = isize.fHeight + lsize.fHeight + 6;
         break;

      case kLVSmallIcons:
      case kLVList:
      case kLVDetails:
         size.fWidth = isize.fWidth + lsize.fWidth + 4;
         size.fHeight = TMath::Max(isize.fHeight, lsize.fHeight);
         break;
   }
   return size;
}



//______________________________________________________________________________
TGLVContainer::TGLVContainer(const TGWindow *p, UInt_t w, UInt_t h,
                             UInt_t options, ULong_t back) :
   TGCompositeFrame(p, w, h, options, back)
{
   // Create a list view container. This is the (large) frame that contains
   // all the list items. It will be show through a TGViewPort (which is
   // created by the TGCanvas derived TGListView).

   fMsgWindow  = p;
   fListView   = 0;
   fLastActive = 0;
   fDragging   = kFALSE;
   fTotal = fSelected = 0;

   fCpos = fJmode = 0;

   fViewMode = kLVLargeIcons;
   fItemLayout = new TGLayoutHints(kLHintsExpandY | kLHintsCenterX);

   SetLayoutManager(new TGTileLayout(this, 8));

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask |
                    kPointerMotionMask, kNone, kNone);
}

//______________________________________________________________________________
TGLVContainer::~TGLVContainer()
{
   // Delete list view container.

   RemoveAll();
   delete fItemLayout;
}

//______________________________________________________________________________
void TGLVContainer::SetViewMode(EListViewMode viewMode)
{
   // Set list view mode for container.

   if (fViewMode != viewMode) {
      TGLayoutHints *oldLayout = fItemLayout;

      fViewMode = viewMode;
      if (viewMode == kLVLargeIcons)
         fItemLayout = new TGLayoutHints(kLHintsExpandY | kLHintsCenterX);
      else
         fItemLayout = new TGLayoutHints(kLHintsLeft | kLHintsCenterY);

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         el->fLayout = fItemLayout;
         ((TGLVEntry *) el->fFrame)->SetViewMode(viewMode);
      }
      delete oldLayout;

      switch (viewMode) {
         default:
         case kLVLargeIcons:
            SetLayoutManager(new TGTileLayout(this, 8));
            break;

         case kLVSmallIcons:
            SetLayoutManager(new TGTileLayout(this, 2));
            break;

         case kLVList:
            SetLayoutManager(new TGListLayout(this, 2));
            break;

         case kLVDetails:
            SetLayoutManager(new TGListDetailsLayout(this, 2));
            break;
      }
      //TGCanvas *canvas = (TGCanvas *) this->GetParent()->GetParent();
      //canvas->Layout();
   }
}

//______________________________________________________________________________
void TGLVContainer::SetColumns(Int_t *cpos, Int_t *jmode)
{
   // Set column information for list items.

   fCpos  = cpos;
   fJmode = jmode;

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next()))
      ((TGLVEntry *) el->fFrame)->SetColumns(fCpos, fJmode);

   Layout();
}

//______________________________________________________________________________
TGDimension TGLVContainer::GetMaxItemSize() const
{
   // Get size of largest item in container.

   TGDimension csize, maxsize(0,0);

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      csize = el->fFrame->GetDefaultSize();
      maxsize.fWidth  = TMath::Max(maxsize.fWidth, csize.fWidth);
      maxsize.fHeight = TMath::Max(maxsize.fHeight, csize.fHeight);
   }
   if (fViewMode == kLVLargeIcons) {
      maxsize.fWidth  += 8;
      maxsize.fHeight += 8;
   } else {
      maxsize.fWidth  += 2;
      maxsize.fHeight += 2;
   }
   return maxsize;
}

//______________________________________________________________________________
Int_t TGLVContainer::GetMaxSubnameWidth(Int_t idx) const
{
   // Get width of largest subname in container.

   if (idx == 0)
      return GetMaxItemSize().fWidth;

   Int_t width, maxwidth = 0;

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVEntry *entry = (TGLVEntry *) el->fFrame;
      width = entry->GetSubnameWidth(idx-1);
      maxwidth = TMath::Max(maxwidth, width);
   }
   return maxwidth;
}

//______________________________________________________________________________
Bool_t TGLVContainer::HandleButton(Event_t *event)
{
   // Handle mouse button event in container.

   Int_t total, selected, page = 0;

   if (event->fCode == kButton4 || event->fCode == kButton5) {
      if (!fListView) return kTRUE;
      if (fListView->GetContainer()->GetHeight())
         page = Int_t(Float_t(fListView->GetViewPort()->GetHeight() *
                              fListView->GetViewPort()->GetHeight()) /
                              fListView->GetContainer()->GetHeight());
   }

   if (event->fCode == kButton4) {
      //scroll up
      Int_t newpos = fListView->GetVsbPosition() - page;
      if (newpos < 0) newpos = 0;
      fListView->SetVsbPosition(newpos);
      return kTRUE;
   }
   if (event->fCode == kButton5) {
      // scroll down
      Int_t newpos = fListView->GetVsbPosition() + page;
      fListView->SetVsbPosition(newpos);
      return kTRUE;
   }

   if (event->fType == kButtonPress) {

      fXp = event->fX;
      fYp = event->fY;

      if (fLastActive) {
         fLastActive->Activate(kFALSE);
         fLastActive = 0;
      }
      total = selected = 0;

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         TGLVEntry *f = (TGLVEntry *) el->fFrame;
         ++total;
         if (f->GetId() == (Window_t)event->fUser[0]) {  // fUser[0] = subwindow
            f->Activate(kTRUE);
            ++selected;
            fLastActive = f;
         } else {
            f->Activate(kFALSE);
         }
      }

      if (fTotal != total || fSelected != selected) {
         fTotal = total;
         fSelected = selected;
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                     fTotal, fSelected);
      }

      if (selected == 0) {
         fDragging = kTRUE;
         fX0 = fXf = fXp;
         fY0 = fYf = fYp;
         gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);
      }
   }

   if (event->fType == kButtonRelease) {
      if (fDragging) {
         fDragging = kFALSE;
         gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);
      } else {
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLVContainer::HandleDoubleClick(Event_t *event)
{
   // Handle double click mouse event.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVEntry *f = (TGLVEntry *) el->fFrame;
      if (f->GetId() == (Window_t)event->fUser[0]) {   // fUser[0] = subwindow
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMDBLCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);
         break;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLVContainer::HandleMotion(Event_t *event)
{
   // Handle mouse motion events.

   int xf0, yf0, xff, yff, total, selected;

   if (fDragging) {
      gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);
      fX0 = TMath::Min(fXp, event->fX);
      fXf = TMath::Max(fXp, event->fX);
      fY0 = TMath::Min(fYp, event->fY);
      fYf = TMath::Max(fYp, event->fY);

      total = selected = 0;

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         TGLVEntry *f = (TGLVEntry *) el->fFrame;
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

      gVirtualX->DrawRectangle(fId, fgLineGC(), fX0, fY0, fXf-fX0, fYf-fY0);

      if (fTotal != total || fSelected != selected) {
         fTotal = total;
         fSelected = selected;
         SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
                     fTotal, fSelected);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGLVContainer::UnSelectAll()
{
   // Unselect all items in the container.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVEntry *f = (TGLVEntry *) el->fFrame;
      f->Activate(kFALSE);
   }

   fSelected = 0;
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
               fTotal, fSelected);
}

//______________________________________________________________________________
void TGLVContainer::SelectAll()
{
   // Select all items in the container.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVEntry *f = (TGLVEntry *) el->fFrame;
      f->Activate(kTRUE);
   }

   fSelected = fTotal;
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
               fTotal, fSelected);
}

//______________________________________________________________________________
void TGLVContainer::InvertSelection()
{
   // Incert the selection, all selected items become unselected and
   // vice versa.

   int selected = 0;

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVEntry *f = (TGLVEntry *) el->fFrame;
      if (f->IsActive()) {
         f->Activate(kFALSE);
      } else {
         f->Activate(kTRUE);
         ++selected;
      }
   }

   fSelected = selected;
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
               fTotal, fSelected);
}

//______________________________________________________________________________
void TGLVContainer::RemoveAll()
{
   // Remove all items from the container.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      el->fFrame->DestroyWindow();
      delete el->fFrame;
   }
   fList->Delete();

   fSelected = fTotal = 0;
   fLastActive = 0;
   SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_SELCHANGED),
               fTotal, fSelected);
}

//______________________________________________________________________________
void TGLVContainer::RemoveItem(TGLVEntry *item)
{
   // Remove item from container.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      if (item == el->fFrame) {
         if (item == fLastActive) fLastActive = 0;
         if (item->IsActive()) fSelected--;
         fTotal--;
         item->DestroyWindow();
         delete item;
         fList->Remove(el);
         delete el;
         break;
      }
   }
}

//______________________________________________________________________________
void TGLVContainer::RemoveItemWithData(void *userData)
{
   // Remove item with fUserData == userData from container.

   TGFrameElement *el;
   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      TGLVEntry *f = (TGLVEntry *) el->fFrame;
      if (f->GetUserData() == userData) {
         RemoveItem(f);
         break;
      }
   }
}

//______________________________________________________________________________
const TGLVEntry *TGLVContainer::GetNextSelected(void **current)
{
   // Return next selected item.

   TGLVEntry *f;
   TObjLink *lnk = (TObjLink *) *current;

   lnk = (lnk == 0) ? fList->FirstLink() : lnk->Next();
   while (lnk) {
      f = (TGLVEntry *) ((TGFrameElement *) lnk->GetObject())->fFrame;
      if (f->IsActive()) {
         *current = (void *) lnk;
         return f;
      }
      lnk = lnk->Next();
   }
   return 0;
}



//______________________________________________________________________________
TGListView::TGListView(const TGWindow *p, UInt_t w, UInt_t h,
                       UInt_t options, ULong_t back) :
   TGCanvas(p, w, h, options, back)
{
   // Create a list view widget.

   fViewMode  = kLVLargeIcons;
   fNColumns  = 0;
   fColumns   = 0;
   fJmode     = 0;
   fColHeader = 0;

   SetDefaultHeaders();
}

//______________________________________________________________________________
TGListView::~TGListView()
{
   // Delete a list view widget.

   if (fNColumns) {
      delete [] fColumns;
      delete [] fJmode;
      for (int i = 0; i < fNColumns; i++)
         delete fColHeader[i];
      delete [] fColHeader;
   }
}

//______________________________________________________________________________
void TGListView::SetHeaders(Int_t ncolumns)
{
   // Set number of headers, i.e. columns that will be shown in detailed view.
   // This method must be followed by exactly ncolumns SetHeader() calls,
   // making sure that every header (i.e. idx) is set (for and example see
   // SetDefaultHeaders()).

   if (ncolumns <= 0) {
      Error("SetHeaders", "number of columns must be > 0");
      return;
   }

   if (fNColumns) {
      delete [] fColumns;
      delete [] fJmode;
      for (int i = 0; i < fNColumns; i++) {
         if (fColHeader[i]) fColHeader[i]->DestroyWindow();
         delete fColHeader[i];
      }
      delete [] fColHeader;
   }

   fNColumns = ncolumns+1;    // one extra for the blank filler header
   fColumns   = new int[fNColumns];
   fJmode     = new int[fNColumns];
   fColHeader = new TGTextButton* [fNColumns];
   for (int i = 0; i < fNColumns; i++)
      fColHeader[i] = 0;

   // create blank filler header
   fColHeader[fNColumns-1] = new TGTextButton(this, new TGHotString(""), -1,
                                              fgDefaultGC(), fgDefaultFontStruct);
   fColHeader[fNColumns-1]->SetTextJustify(kTextCenterX | kTextCenterY);
   fColHeader[fNColumns-1]->SetState(kButtonDisabled);
   fJmode[fNColumns-1]   = kTextCenterX;
   fColumns[fNColumns-1] = 0;
}

//______________________________________________________________________________
void TGListView::SetHeader(const char *s, Int_t hmode, Int_t cmode, Int_t idx)
{
   // Set header button idx [0-fNColumns>, hmode is the x text alignmode
   // (ETextJustification) for the header text and cmode is the x text
   // alignmode for the item text.

   if (idx < 0 || idx >= fNColumns-1) {
      Error("SetHeader", Form("header index must be [0 - %d>", fNColumns-1));
      return;
   }
   delete fColHeader[idx];
   fColHeader[idx] = new TGTextButton(this, new TGHotString(s),
                                      idx, fgDefaultGC(), fgDefaultFontStruct);
   fColHeader[idx]->SetTextJustify(hmode | kTextCenterY);

   // fJmode and fColumns contain values for columns idx > 0. idx==0 is
   // the small icon with the object name
   if (idx > 0)
      fJmode[idx-1] = cmode;

   if (!fColHeader[0]) return;
   int xl = fBorderWidth + fColHeader[0]->GetDefaultWidth() + 20 + 10;
   for (int i = 1; i < fNColumns; i++) {
      fColumns[i-1] = xl;
      if (!fColHeader[i]) break;
      xl += fColHeader[i]->GetDefaultWidth() + 20;
   }
}

//______________________________________________________________________________
const char *TGListView::GetHeader(Int_t idx) const
{
   // Returns name of header idx. If illegal idx or header not set for idx
   // 0 is returned.

   if (idx >= 0 && idx < fNColumns-1 && fColHeader[idx])
      return fColHeader[idx]->GetText()->GetString();
   return 0;
}

//______________________________________________________________________________
void TGListView::SetDefaultHeaders()
{
   // Default headers are: Name, Attributes, Size, Owner, Group, Modified.
   // The default is good for file system items.

   SetHeaders(6);
   SetHeader("Name",       kTextLeft,    kTextLeft,    0);
   SetHeader("Attributes", kTextCenterX, kTextCenterX, 1);
   SetHeader("Size",       kTextRight,   kTextRight,   2);
   SetHeader("Owner",      kTextCenterX, kTextCenterX, 3);
   SetHeader("Group",      kTextCenterX, kTextCenterX, 4);
   SetHeader("Modified",   kTextCenterX, kTextCenterX, 5);
}

//______________________________________________________________________________
void TGListView::SetViewMode(EListViewMode viewMode)
{
   // Set list view mode.

   TGLVContainer *container;

   if (fViewMode != viewMode) {
      fViewMode = viewMode;
      container = (TGLVContainer *) fVport->GetContainer();
      if (container) container->SetViewMode(viewMode);
      Layout();
   }
}

//______________________________________________________________________________
void TGListView::SetContainer(TGFrame *f)
{
   // Set list view container. Container must be at least of type
   // TGLVContainer.

   if (f->InheritsFrom(TGLVContainer::Class())) {
      TGCanvas::SetContainer(f);
      ((TGLVContainer *) f)->SetColumns(fColumns, fJmode);
      ((TGLVContainer *) f)->SetListView(this);
   } else
      Error("SetContainer", "frame must inherit from TGLVContainer");
}

//______________________________________________________________________________
void TGListView::Layout()
{
   // Layout list view components (container and contents of container).

   Int_t  i, xl = fBorderWidth;
   UInt_t w, h = 0;

   TGLVContainer *container = (TGLVContainer *) fVport->GetContainer();
   if (!container) {
      Error("Layout", "no listview container set yet");
      return;
   }
   fMaxSize = container->GetMaxItemSize();

   if (fViewMode == kLVDetails) {

      h = fColHeader[0]->GetDefaultHeight()-4;
      for (i = 0; i < fNColumns-1; ++i) {
         w = fColHeader[i]->GetDefaultWidth()+20;
         if (i == 0) w = TMath::Max(fMaxSize.fWidth + 10, w);
         if (i > 0)  w = TMath::Max(container->GetMaxSubnameWidth(i) + 10, (Int_t)w);
         fColHeader[i]->MoveResize(xl, fBorderWidth, w, h);
         fColHeader[i]->MapWindow();
         xl += w;
         fColumns[i] = xl-fBorderWidth-2;  // -2 is fSep in the layout routine
      }
      fColHeader[i]->MoveResize(xl, fBorderWidth, fVport->GetWidth()-xl+fBorderWidth, h);
      fColHeader[i]->MapWindow();
      fVScrollbar->RaiseWindow();

      container->SetColumns(fColumns, fJmode);

   } else {
      for (i = 0; i < fNColumns; ++i)
        fColHeader[i]->UnmapWindow();
   }

   TGCanvas::Layout();
   if (fViewMode == kLVList) {
      if (fMaxSize.fWidth > 0)
         fHScrollbar->SetRange(container->GetWidth()/fMaxSize.fWidth,
                               fVport->GetWidth()/fMaxSize.fWidth);
   }
   if (fViewMode == kLVDetails) {
      fColHeader[i]->MoveResize(xl, fBorderWidth, fVport->GetWidth()-xl+fBorderWidth, h);
      fVport->MoveResize(fBorderWidth, fBorderWidth+h, fVport->GetWidth(), fVport->GetHeight()-h);
      fVScrollbar->SetRange(container->GetHeight(), fVport->GetHeight());
   }
}

//______________________________________________________________________________
Bool_t TGListView::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Handle messages generated by the list view components.

   if ((fViewMode == kLVList) && (GET_MSG(msg) == kC_HSCROLL)) {
      switch (GET_SUBMSG(msg)) {
         case kSB_SLIDERTRACK:
         case kSB_SLIDERPOS:
            fVport->SetHPos((Int_t)-parm1 * fMaxSize.fWidth);
            break;
         default:
            break;
      }
   } else {
      TGLVContainer *cnt = (TGLVContainer*)GetContainer();
      const TGLVEntry *entry;
      void *p = 0;

      entry = cnt->GetNextSelected(&p);

      switch (GET_SUBMSG(msg)) {
         case kCT_ITEMCLICK:
            if ((cnt->NumSelected() == 1) && (entry != 0))
               Clicked((TGLVEntry*)entry, (Int_t)parm1);
            break;
         case kCT_ITEMDBLCLICK:
            if ((cnt->NumSelected() == 1) && (entry!=0))
               DoubleClicked((TGLVEntry*)entry, (Int_t)parm1);
            break;
         case kCT_SELCHANGED:
            SelectionChanged();
            break;
         default:
            break;
      }
      return TGCanvas::ProcessMessage(msg, parm1, parm2);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGListView::DoubleClicked(TGLVEntry *entry, Int_t btn)
{
   // Emit DoubleClicked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = btn;

   Emit("DoubleClicked(TGLVEntry*,Int_t)", args);
}

//______________________________________________________________________________
void TGListView::Clicked(TGLVEntry *entry, Int_t btn)
{
   // Emit Clicked() signal.

   Long_t args[2];

   args[0] = (Long_t)entry;
   args[1] = btn;

   Emit("Clicked(TGLVEntry*,Int_t)", args);
}
