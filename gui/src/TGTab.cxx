// @(#)root/gui:$Name:  $:$Id: TGTab.cxx,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
// Author: Fons Rademakers   13/01/98

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
// TGTab, TGTabElement, TGTabLayout                                     //
//                                                                      //
// A tab widget contains a set of composite frames each with a little   //
// tab with a name (like a set of folders with tabs).                   //
//                                                                      //
// The TGTab is user callable. The TGTabElement and TGTabLayout are     //
// is a service classes of the tab widget.                              //
//                                                                      //
// Clicking on a tab will bring the associated composite frame to the   //
// front and generate the following event:                              //
// kC_COMMAND, kCM_TAB, tab id, 0.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGTab.h"
#include "TList.h"
#include "TMath.h"


ClassImp(TGTabElement)
ClassImp(TGTabLayout)
ClassImp(TGTab)


//______________________________________________________________________________
TGTabElement::TGTabElement(const TGWindow *p, TGString *text, UInt_t w, UInt_t h,
                           GContext_t norm, FontStruct_t font,
                           UInt_t options, ULong_t back) :
   TGFrame(p, w, h, options, back)
{
   // Create a tab element. Text is adopted by tab element.

   fText        = text;
   fBorderWidth = 0;
   fNormGC      = norm;
   fFontStruct  = font;

   int max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   Resize(TMath::Max(fTWidth+12, (UInt_t)45), fTHeight+6);
}

//______________________________________________________________________________
TGTabElement::~TGTabElement()
{
   // Delete tab element.

   if (fText) delete fText;
}

//______________________________________________________________________________
void TGTabElement::DrawBorder()
{
   // Draw little tab element.

   gVirtualX->DrawLine(fId, fgHilightGC(), 0, fHeight-1, 0, 2);
   gVirtualX->DrawLine(fId, fgHilightGC(), 0, 2, 2, 0);
   gVirtualX->DrawLine(fId, fgHilightGC(), 2, 0, fWidth-3, 0);
   gVirtualX->DrawLine(fId, fgShadowGC(),  fWidth-2, 1, fWidth-2, fHeight-1);
   gVirtualX->DrawLine(fId, fgBlackGC(), fWidth-2, 1, fWidth-1, 2);
   gVirtualX->DrawLine(fId, fgBlackGC(), fWidth-1, 2, fWidth-1, fHeight-2);
   gVirtualX->DrawLine(fId, fgHilightGC(), fWidth-1, fHeight-1, fWidth-1, fHeight-1);

   if (fText) {
      int max_ascent, max_descent;
      gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
      fText->Draw(fId, fNormGC, 6, max_ascent+3);
   }
}

//______________________________________________________________________________
TGDimension TGTabElement::GetDefaultSize() const
{
   // Return default size of tab element.

   return TGDimension(TMath::Max(fTWidth+12, (UInt_t)45), fTHeight+6);
}

//______________________________________________________________________________
void TGTabElement::SetText(TGString *text)
{
   // Set new tab text.

   if (fText) delete fText;
   fText = text;

   int max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   fClient->NeedRedraw(this);
}


//______________________________________________________________________________
TGTabLayout::TGTabLayout(TGTab *main)
{
   // Create a tab layout manager.

   fMain = main;
   fList = fMain->GetList();
}

//______________________________________________________________________________
void TGTabLayout::Layout()
{
   // Layout the tab widget.

   Int_t  i, xtab;
   UInt_t tw;
   UInt_t tabh = fMain->GetTabHeight(), bw = fMain->GetBorderWidth();
   UInt_t w = fMain->GetWidth();
   UInt_t h = fMain->GetHeight();

   xtab = 2;

   fMain->GetContainer()->MoveResize(0, tabh, w, h - tabh);

   // first frame is the container, so take next...
   TGFrameElement *el, *elnxt;
   TIter next(fList);
   i = 0;
   next();   // skip first
   while ((el = (TGFrameElement *) next())) {
      elnxt = (TGFrameElement *) next();
      tw = el->fFrame->GetDefaultWidth();
      if (i == fMain->GetCurrent()) {
         el->fFrame->MoveResize(xtab-2, 0, tw+3, tabh+1);
         elnxt->fFrame->RaiseWindow();
         el->fFrame->RaiseWindow();
      } else {
         el->fFrame->MoveResize(xtab, 2, tw, tabh-1);
         el->fFrame->LowerWindow();
      }
      elnxt->fFrame->MoveResize(bw, tabh + bw, w - (bw << 1), h - tabh - (bw << 1));
      elnxt->fFrame->Layout();
      xtab += (Int_t)tw;
      i++;
   }
}

//______________________________________________________________________________
TGDimension TGTabLayout::GetDefaultSize() const
{
   // Get default size of tab widget.

   TGDimension dsize, dsize_te;
   TGDimension size(0,0), size_te(0,0);

   TGFrameElement *el, *elnxt;
   TIter next(fList);
   next();   // skip first container
   while ((el = (TGFrameElement *)next())) {
      dsize_te = el->fFrame->GetDefaultSize();
      size_te.fWidth += dsize_te.fWidth;
      elnxt = (TGFrameElement *) next();
      dsize = elnxt->fFrame->GetDefaultSize();
      if (size.fWidth < dsize.fWidth) size.fWidth = dsize.fWidth;
      if (size.fHeight < dsize.fHeight) size.fHeight = dsize.fHeight;
   }

   // check if tab elements make a larger width than the containers
   if (size.fWidth < size_te.fWidth) size.fWidth = size_te.fWidth;

   size.fWidth += fMain->GetBorderWidth() << 1;
   size.fHeight += fMain->GetTabHeight() + (fMain->GetBorderWidth() << 1);

   return size;
}


//______________________________________________________________________________
TGTab::TGTab(const TGWindow *p, UInt_t w, UInt_t h,
             GContext_t norm, FontStruct_t font,
             UInt_t options, ULong_t back) :
   TGCompositeFrame(p, w, h, options, back)
{
   // Create tab widget.

   fMsgWindow  = p;

   fBorderWidth = 2;
   fCurrent     = 0;
   fRemoved     = new TList;

   fNormGC     = norm;
   fFontStruct = font;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTabh = max_ascent + max_descent + 6;

   SetLayoutManager(new TGTabLayout(this));

   // we need this in order to avoid border blinking when switching tabs...
   fContainer = new TGCompositeFrame(this, fWidth, fHeight - fTabh,
                       kVerticalFrame | kRaisedFrame | kDoubleBorder);
   AddFrame(fContainer, 0);

   gVirtualX->SelectInput(fId, kButtonPressMask);
}

//______________________________________________________________________________
TGTab::~TGTab()
{
   // Delete tab widget. This deletes the tab windows and the containers.
   // The tab string is deleted by the TGTabElement dtor.

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      delete el->fFrame;
      // delete el->fLayout; // el->fLayout is NULL!
   }

   fRemoved->Delete();
   delete fRemoved;
}

//______________________________________________________________________________
TGCompositeFrame *TGTab::AddTab(TGString *text)
{
   // Add a tab to the tab widget. Returns the new container, which
   // is owned by the tab widget. The text is adopted by the tab widget.

   TGCompositeFrame *cf;

   AddFrame(new TGTabElement(this, text, 50, 20, fNormGC, fFontStruct), 0);
   cf = new TGCompositeFrame(this, fWidth, fHeight-21);
   AddFrame(cf, 0);

   return cf;
}

//______________________________________________________________________________
TGCompositeFrame *TGTab::AddTab(const char *text)
{
   // Add a tab to the tab widget. Returns the new container. The container
   // is owned by the tab widget.

   return AddTab(new TGString(text));
}

//______________________________________________________________________________
void TGTab::RemoveTab(Int_t tabIndex)
{
   // Remove container and tabtab of tab with index tabIndex.
   // Does NOT remove the container contents!

   if (tabIndex < 0) return;

   TGFrameElement *elTab, *elCont ;
   Int_t  count = 0 ;

   TIter next(fList) ;
   next() ; // skip first container

   while ((elTab = (TGFrameElement *) next())) {
      elCont = (TGFrameElement *) next();
      if (count == tabIndex) {
         elTab->fFrame->DestroyWindow();
         elCont->fFrame->DestroyWindow();
         delete elTab->fFrame;
         fRemoved->Add(elCont->fFrame);   // delete only in dtor
         RemoveFrame(elTab->fFrame);
         RemoveFrame(elCont->fFrame);
         if (tabIndex == fCurrent) {
           // select another tab only if the current is the one we delete
           SetTab(0) ;
         } else
            fCurrent--;
         break;
      }
      count++;
   }
}

//______________________________________________________________________________
void TGTab::ChangeTab(Int_t tabIndex)
{
   // Make tabIdx the current tab. Utility method called by SetTab and
   // HandleButton().

   if (tabIndex != fCurrent) {
      TGFrameElement *el, *elnxt;
      UInt_t tw;
      Int_t  xtab  = 2;
      Int_t  count = 0;

      TIter next(fList);
      next();           // skip first container

      fCurrent = tabIndex;
      while ((el = (TGFrameElement *) next())) {
         elnxt = (TGFrameElement *) next();
         tw = el->fFrame->GetDefaultWidth();
         if (count == fCurrent) {
            el->fFrame->MoveResize(xtab-2, 0, tw+3, fTabh+1);
            elnxt->fFrame->RaiseWindow();
            el->fFrame->RaiseWindow();
         } else {
            el->fFrame->MoveResize(xtab, 2, tw, fTabh-1);
            el->fFrame->LowerWindow();
         }
         xtab += tw;
         count++;
      }
      SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_TAB), fCurrent, 0);
      fClient->ProcessLine(fCommand, MK_MSG(kC_COMMAND, kCM_TAB), fCurrent, 0);
   }
}

//______________________________________________________________________________
Bool_t TGTab::SetTab(Int_t tabIndex)
{
   // Brings the composite frame with the index tabIndex to the
   // front and generate the following event if the front tab has
   // changed:
   // kC_COMMAND, kCM_TAB, tab id, 0.
   // Returns kFALSE if tabIndex is a not valid index

   // check if tabIndex is a valid index
   if (tabIndex < 0)
      return kFALSE;

   // count the tabs
   TIter next(fList);
   Int_t count = 0;
   while (next())
      count++;

   count = count / 2 - 1;
   if (tabIndex > count)
      return kFALSE;

   // change tab and generate event
   ChangeTab(tabIndex);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTab::HandleButton(Event_t *event)
{
   // Handle button event in the tab widget. Basically we only handle
   // button events in the small tabs.

   if (event->fType == kButtonPress) {
      TGFrameElement *el;
      TIter next(fList);

      next();   // skip first container

      Int_t i = 0;
      Int_t c = fCurrent;
      while ((el = (TGFrameElement *) next())) {
         if (el->fFrame->GetId() == (Window_t)event->fUser[0])  // fUser[0] is child window
            c = i;
         next(); i++;
      }

      // change tab and generate event
      ChangeTab(c);
   }
   return kTRUE;
}

//______________________________________________________________________________
TGCompositeFrame *TGTab::GetTabContainer(Int_t tabIndex) const
{
   // Return container of tab with index tabIndex.
   // Returns 0 in case tabIndex is out of range.

   if (tabIndex < 0) return 0;

   TGFrameElement *el;
   Int_t  count = 0;

   TIter next(fList);
   next();           // skip first container

   while (next()) {
      el = (TGFrameElement *) next();
      if (count == tabIndex)
         return (TGCompositeFrame *) el->fFrame;
      count++;
   }

   return 0;
}

//______________________________________________________________________________
TGTabElement *TGTab::GetTabTab(Int_t tabIndex) const
{
   // Return container of tab with index tabIndex.
   // Returns 0 in case tabIndex is out of range.

   if (tabIndex < 0) return 0;

   TGFrameElement *el;
   Int_t  count = 0;

   TIter next(fList);
   next();           // skip first container

   while ((el = (TGFrameElement *) next())) {
      next();
      if (count == tabIndex)
         return (TGTabElement *) el->fFrame;
      count++;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TGTab::GetNumberOfTabs() const
{
   // Return number of tabs.

   Int_t count = 0;

   TIter next(fList);
   next();           // skip first container

   while (next()) {
      next();
      count++;
   }

   return count;
}

//______________________________________________________________________________
FontStruct_t TGTab::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGTab::GetDefaultGC()
{ return fgDefaultGC; }
