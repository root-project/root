// @(#)root/gui:$Name:  $:$Id: TGListBox.cxx,v 1.4 2000/10/04 23:40:07 rdm Exp $
// Author: Fons Rademakers   12/01/98

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
// TGListBox, TGLBContainer, TGLBEntry and TGTextLBEntry                //
//                                                                      //
// A listbox is a box, possibly with scrollbar, containing entries.     //
// Currently entries are simple text strings (TGTextLBEntry).           //
// A TGListBox looks a lot like a TGCanvas. It has a TGViewPort         //
// containing a TGLBContainer which contains the entries and it also    //
// has a vertical scrollbar which becomes visible if there are more     //
// items than fit in the visible part of the container.                 //
//                                                                      //
// The TGListBox is user callable. The other classes are service        //
// classes of the listbox.                                              //
//                                                                      //
// Selecting an item in the listbox will generate the event:            //
// kC_COMMAND, kCM_LISTBOX, listbox id, item id.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGListBox.h"
#include "TGScrollBar.h"
#include "TMath.h"

ClassImp(TGLBEntry)
ClassImp(TGTextLBEntry)
ClassImp(TGLBContainer)
ClassImp(TGListBox)


//______________________________________________________________________________
void TGLBEntry::Activate(Bool_t a)
{
   // Toggle active state of listbox entry.

   if (fActive == a) return;
   fActive = a;
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGLBEntry::Toggle()
{
   // Toggle active state of listbox entry.

   fActive = !fActive;
   fClient->NeedRedraw(this);
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextLBEntry                                                        //
//                                                                      //
// Text string listbox entries.                                         //
// A TGTextLBEntry is for TGListBox internal use.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGTextLBEntry::TGTextLBEntry(const TGWindow *p, TGString *s, Int_t id,
      GContext_t norm, FontStruct_t font, UInt_t options, ULong_t back) :
   TGLBEntry(p, id, options, back)
{
   // Create a text listbox entry. The TGString is adopted.

   fText        = s;
   fTextChanged = kTRUE;
   fFontStruct  = font;
   fNormGC      = norm;

   int max_ascent, max_descent;

   fTWidth  = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   Resize(fTWidth, fTHeight + 1);
}

//______________________________________________________________________________
TGTextLBEntry::~TGTextLBEntry()
{
   // Delete text listbox entry.

   if (fText) delete fText;
}

//______________________________________________________________________________
void TGTextLBEntry::DoRedraw()
{
   // Redraw text listbox entry.

   int x, y, max_ascent, max_descent;

   //if (fTextChanged) {
   //   TGFrame::DoRedraw();
   //   fTextChanged = kFALSE;
   //}

   x = 3;
   y = (fHeight - fTHeight) >> 1;

   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   if (fActive) {
      SetBackgroundColor(fgDefaultSelectedBackground);
      gVirtualX->ClearWindow(fId);
      gVirtualX->SetForeground(fNormGC, fgSelPixel);
      fText->Draw(fId, fNormGC, x, y + max_ascent);
   } else {
      SetBackgroundColor(fBkcolor);
      gVirtualX->ClearWindow(fId);
      gVirtualX->SetForeground(fNormGC, fgBlackPixel);
      fText->Draw(fId, fNormGC, x, y + max_ascent);
   }
}

//______________________________________________________________________________
void TGTextLBEntry::SetText(TGString *new_text)
{
   // Set or change text in text entry.

   if (fText) delete fText;
   fText = new_text;
   fTextChanged = kTRUE;

   int max_ascent, max_descent;
   fTWidth = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   Resize(fTWidth, fTHeight + 1);

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
FontStruct_t TGTextLBEntry::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGTextLBEntry::GetDefaultGC()
{ return fgDefaultGC; }


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLBContainer                                                        //
//                                                                      //
// A Composite frame that contains a list of TGLBEnties.                //
// A TGLBContainer is for TGListBox internal use.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGLBContainer::TGLBContainer(const TGWindow *p, UInt_t w, UInt_t h,
                             UInt_t options, ULong_t back) :
   TGCompositeFrame(p, w, h, options, back)
{
   // Create a listbox container.

   fLastActive = 0;
   fMsgWindow  = p;
   fMultiSelect = kFALSE;

   // SetLayoutManager(new TGColumnLayout(this, 0));

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask |
                    kPointerMotionMask, kNone, kNone);
}

//______________________________________________________________________________
TGLBContainer::~TGLBContainer()
{
   // Delete the listbox container.

   TGFrameElement *el;

   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      delete el->fFrame;
      delete el->fLayout;
   }
}

//______________________________________________________________________________
void TGLBContainer::AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints)
{
   // Add listbox entry with hints to container. To show entry call
   // MapSubwindows() and Layout().

   lbe->SetBackgroundColor(fgWhitePixel);
   AddFrame(lbe, lhints);
   // Layout();
}

//______________________________________________________________________________
void TGLBContainer::InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, Int_t afterID)
{
   // Insert listbox entry after specified entry with id afterID. If afterID = -1
   // then add entry at head of list. To show entry call MapSubwindows() and
   // Layout().

   lbe->SetBackgroundColor(fgWhitePixel);

   TGLBEntry      *e;
   TGFrameElement *el, *nw;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      e = (TGLBEntry *) el->fFrame;
      if (e->EntryId() == afterID) break;
   }

   if (!el && afterID != -1)
      AddFrame(lbe, lhints);
   else {
      nw = new TGFrameElement;
      nw->fFrame  = lbe;
      nw->fLayout = lhints;
      nw->fState  = 1;
      if (afterID == -1)
         fList->AddFirst(nw);
      else
         fList->AddAfter(el, nw);
   }
   // Layout();
}

//______________________________________________________________________________
void TGLBContainer::AddEntrySort(TGLBEntry *lbe, TGLayoutHints *lhints)
{
   // Insert listbox entry before the list box entry with a higher id.
   // To show entry call MapSubwindows() and Layout().

   lbe->SetBackgroundColor(fgWhitePixel);

   TGLBEntry      *e;
   TGFrameElement *el, *nw;
   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      e = (TGLBEntry *) el->fFrame;
      if (e->EntryId() > lbe->EntryId()) break;
   }

   if (!el)
      AddFrame(lbe, lhints);
   else {
      nw = new TGFrameElement;
      nw->fFrame  = lbe;
      nw->fLayout = lhints;
      nw->fState  = 1;
      fList->AddBefore(el, nw);
   }
   // Layout();
}

//______________________________________________________________________________
void TGLBContainer::RemoveEntry(Int_t id)
{
   // Remove the entry with specified id from the listbox container.
   // To update the listbox call Layout().

   TGLBEntry      *e;
   TGFrameElement *el;
   TGLayoutHints  *l;

   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      e = (TGLBEntry *) el->fFrame;
      l = el->fLayout;
      if (e->EntryId() == id) {
         if (fLastActive == e) fLastActive = 0;
         e->DestroyWindow();
         fList->Remove(el);  // avoid calling RemoveFrame(e)
         delete el;          // idem
         delete e;
         delete l;
         // Layout();
         break;
      }
   }
}

//______________________________________________________________________________
void TGLBContainer::RemoveEntries(Int_t from_ID, Int_t to_ID)
{
   // Remove entries from from_ID to to_ID (including).
   // To update the listbox call Layout().

   TGLBEntry      *e;
   TGFrameElement *el;
   TGLayoutHints  *l;

   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      e = (TGLBEntry *) el->fFrame;
      l = el->fLayout;
      if ((e->EntryId() >= from_ID) && (e->EntryId() <= to_ID)) {
         if (fLastActive == e) fLastActive = 0;
         e->DestroyWindow();
         fList->Remove(el);  // avoid calling RemoveFrame(e)
         delete el;          // idem
         delete e;
         delete l;
      }
   }
   // Layout();
}

//______________________________________________________________________________
TGLBEntry *TGLBContainer::Select(Int_t id)
{
   return Select(id, kTRUE);
}

//______________________________________________________________________________
TGLBEntry *TGLBContainer::Select(Int_t id, Bool_t sel)
{
   // Select / deselect the entry with the specified id.
   // Returns the selected TGLBEntry.

   TGLBEntry      *f;
   TGFrameElement *el;

   if (fLastActive) {
      fLastActive->Activate(kFALSE);
      fLastActive = 0;
   }

   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      f = (TGLBEntry *) el->fFrame;
      if (f->EntryId() == id) {
         f->Activate(sel);
         if (fMultiSelect == kFALSE && sel == kTRUE)
            fLastActive = f;
         return f;
      }
   }

   return 0;
}

//______________________________________________________________________________
Int_t TGLBContainer::GetSelected() const
{
   // Returns id of selected entry. In case of no selected entry or
   // if multi selection is switched on returns -1.

   if (fLastActive == 0) return -1;
   return fLastActive->EntryId();
}

//______________________________________________________________________________
Bool_t TGLBContainer::GetSelection(Int_t id)
{
   // Returns kTrue if entry id is selected.

   TGLBEntry     *f;
   TGFrameElement *el;

   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      f = (TGLBEntry *) el->fFrame;
      if (f->EntryId() == id)
         return f->IsActive();
   }

   return kFALSE;

}

//______________________________________________________________________________
void TGLBContainer::GetSelectedEntries(TList *selected)
{
   // Adds all selected entries (TGLBEntry) of the list box into
   // the list selected.

   TGLBEntry      *f;
   TGFrameElement *el;

   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      f = (TGLBEntry *) el->fFrame;
      if (f->IsActive())
         selected->Add(f);
   }
}

//______________________________________________________________________________
void TGLBContainer::SetMultipleSelections(Bool_t multi)
{
   // enables and disables multible selections of entries

   TGFrameElement *el;

   fMultiSelect = multi;
   if (fMultiSelect)
      fLastActive = 0;
   else
      {
      // deselect all entries
      TIter next(fList);
      while ((el = (TGFrameElement *) next()))
         ((TGLBEntry *)(el->fFrame))->Activate(kFALSE);
      }
}

//______________________________________________________________________________
Bool_t TGLBContainer::HandleButton(Event_t *event)
{
   // Handle mouse button event in the listbox container.

   TGLBEntry *f;
   TGFrameElement *el;

   if (event->fCode == kButton4) {
      // scroll 2 lines up (a button down is always followed by a button up)
      Int_t newpos = fListBox->GetScrollBar()->GetPosition() - 1;
      if (newpos < 0) newpos = 0;
      fListBox->GetScrollBar()->SetPosition(newpos);
      return kTRUE;
   }
   if (event->fCode == kButton5) {
      // scroll 2 lines down (a button down is always followed by a button up)
      Int_t newpos = fListBox->GetScrollBar()->GetPosition() + 1;
      fListBox->GetScrollBar()->SetPosition(newpos);
      return kTRUE;
   }

   if (fMultiSelect) {
      if (event->fType == kButtonPress) {
         TIter next(fList);
         while ((el = (TGFrameElement *) next())) {
            f = (TGLBEntry *) el->fFrame;
            if (f->GetId() == (Window_t)event->fUser[0]) {    // fUser[0] == child window
               f->Toggle();
               fChangeStatus = f->IsActive() ? 1 : 0;
               SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                           f->EntryId(), 0);
               break;
            }
         }
      } else
         fChangeStatus = -1;
   } else {
      if (event->fType == kButtonPress) {
         if (fLastActive) {
            fLastActive->Activate(kFALSE);
            fLastActive = 0;
         }
         TIter next(fList);
         while ((el = (TGFrameElement *) next())) {
            f = (TGLBEntry *) el->fFrame;
            if (f->GetId() == (Window_t)event->fUser[0]) {    // fUser[0] == child window
               f->Activate(kTRUE);
               fLastActive = f;
            } else {
               f->Activate(kFALSE);
            }
         }
      } else {
         if (fLastActive) {
            f = fLastActive;
            SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                        f->EntryId(), 0);
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLBContainer::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in listbox container.

   TGLBEntry *f;
   TGFrameElement *el;

   if (fMultiSelect) {

      if (fChangeStatus >= 0) {
         TIter next(fList);
         while ((el = (TGFrameElement *) next())) {
            f = (TGLBEntry *) el->fFrame;
            if (f->GetId() == (Window_t)event->fUser[0]) {   // fUser[0] = child window
               if (fChangeStatus != f->IsActive() ? 1 : 0) {
                  f->Toggle();
                  SendMessage(fMsgWindow, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                              f->EntryId(), 0);
               }
               break;
            }
         }
      }
   } else {
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         f = (TGLBEntry *) el->fFrame;
         if (f->GetId() == (Window_t)event->fUser[0]) {   // fUser[0] = child window
            f->Activate(kTRUE);
            fLastActive = f;
         } else {
            f->Activate(kFALSE);
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TGLBContainer::GetPos(Int_t id)
{
   // Returns the position in the list box of the entry id.
   // The first position has position no 0. Returns -1 if entry id
   // is not in the list of entries.

   Int_t          pos = 0;
   TGLBEntry      *f;
   TGFrameElement *el;

   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      f = (TGLBEntry *) el->fFrame;
      if (f->EntryId() == id)
         return pos;
      pos++;
   }

  return -1;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGListBox                                                            //
//                                                                      //
// A listbox contains a container frame which is viewed through a       //
// viewport. If the container is larger than the viewport than a        //
// vertical scrollbar is added.                                         //
//                                                                      //
// Selecting an item in the listbox will generate the event:            //
// kC_COMMAND, kCM_LISTBOX, listbox id, item id.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGListBox::TGListBox(const TGWindow *p, Int_t id,
                     UInt_t options, ULong_t back) :
   TGCompositeFrame(p, 10, 10, options, back)
{
   // Create a listbox.

   fMsgWindow = p;
   fListBoxId = id;

   fItemVsize = 1;
   fIntegralHeight = kTRUE;

   InitListBox();
}

//______________________________________________________________________________
TGListBox::~TGListBox()
{
   // Delete a listbox widget.

   delete fVScrollbar;
   delete fLbc;
   delete fVport;
}

//______________________________________________________________________________
void TGListBox::InitListBox()
{
   // initiate the internal classes of a list box

   fVport = new TGViewPort(this, 6, 6, kChildFrame | kOwnBackground, fgWhitePixel);
   fVScrollbar = new TGVScrollBar(this, kDefaultScrollBarWidth, 6);
   fLbc = new TGLBContainer(fVport, 10, 10, kVerticalFrame, fgWhitePixel);
   fLbc->Associate(this);
   fLbc->SetListBox(this);
   SetContainer(fLbc);

   AddFrame(fVport, 0);
   AddFrame(fVScrollbar, 0);

   fVScrollbar->Associate(this);

   fVScrollbar->AddInput(kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask);
   fLbc->AddInput(kButtonPressMask | kButtonReleaseMask
                  /*| kPointerMotionMask */);
}

//______________________________________________________________________________
void TGListBox::DrawBorder()
{
   // Draw borders of the list box widget.

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
         TGCompositeFrame::DrawBorder();
         break;
   }
}

//______________________________________________________________________________
void TGListBox::AddEntry(TGString *s, Int_t id)
{
   // Add entry with specified string and id to listbox. The id will be
   // used in the event processing routine when the item is selected.
   // The string will be adopted by the listbox.

   TGTextLBEntry *lbe;
   TGLayoutHints *lhints;

   lbe = new TGTextLBEntry(fLbc, s, id);
   lhints = new TGLayoutHints(kLHintsExpandX | kLHintsTop);
   fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());
   fLbc->AddEntry(lbe, lhints);
}

//______________________________________________________________________________
void TGListBox::AddEntry(const char *s, Int_t id)
{
   // Add entry with specified string and id to listbox. The id will be
   // used in the event processing routine when the item is selected.

   AddEntry(new TGString(s), id);
}

//______________________________________________________________________________
void TGListBox::AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints)
{
   // Add specified TGLBEntry and TGLayoutHints to listbox. The
   // entry and layout will be adopted and later deleted by the listbox.

   fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());
   fLbc->AddEntry(lbe, lhints);
}

//______________________________________________________________________________
void TGListBox::AddEntrySort(TGString *s, Int_t id)
{
   // Add entry with specified string and id to listbox sorted by increasing id.
   // This sorting works proberly only if EntrySort functions are used to add
   // entries without mixing them with other add or insert functions.  The id will be
   // used in the event processing routine when the item is selected.
   // The string will be adopted by the listbox.

   TGTextLBEntry *lbe;
   TGLayoutHints *lhints;

   lbe = new TGTextLBEntry(fLbc, s, id);
   lhints = new TGLayoutHints(kLHintsExpandX | kLHintsTop);
   fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());
   fLbc->AddEntrySort(lbe, lhints);
}

//______________________________________________________________________________
void TGListBox::AddEntrySort(const char *s, Int_t id)
{
   // Add entry with specified string and id to listbox sorted by increasing id.
   // This sorting works proberly only if EntrySort functions are used to add
   // entries without mixing them with other add or insert functions. The id will be
   // used in the event processing routine when the item is selected.

   AddEntrySort(new TGString(s), id);
}

//______________________________________________________________________________
void TGListBox::AddEntrySort(TGLBEntry *lbe, TGLayoutHints *lhints)
{
   // Add specified TGLBEntry and TGLayoutHints to listbox sorted by increasing id.
   // This sorting works proberly only if EntrySort functions are used to add
   // entries without mixing them with other add or insert functions. The
   // entry and layout will be adopted and later deleted by the listbox.

   fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());
   fLbc->AddEntrySort(lbe, lhints);
}

//______________________________________________________________________________
void TGListBox::InsertEntry(TGString *s, Int_t id, Int_t afterID)
{
   // Insert entry with specified string and id behind the entry with afterID.
   // The string will be adopted and later deleted by the listbox.

   TGTextLBEntry *lbe;
   TGLayoutHints *lhints;

   lbe = new TGTextLBEntry(fLbc, s, id);
   lhints = new TGLayoutHints(kLHintsExpandX | kLHintsTop);
   fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());
   fLbc->InsertEntry(lbe, lhints, afterID);
}

//______________________________________________________________________________
void TGListBox::InsertEntry(const char *s, Int_t id, Int_t afterID)
{
   // Insert entry with specified string and id behind the entry with afterID.

   InsertEntry(new TGString(s), id, afterID);
}

//______________________________________________________________________________
void TGListBox::InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, int afterID)
{
   // Insert the specified TGLBEntry and layout hints behind afterID.
   // The entry and layout will be adopted and later deleted by the listbox.

   fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());
   fLbc->InsertEntry(lbe, lhints, afterID);
}

//______________________________________________________________________________
void TGListBox::SetTopEntry(Int_t id)
{
   // Scroll the entry with id to the top of the listbox.

   Int_t idPos;

   idPos = fLbc->GetPos(id);

   // id not found in list of entries
   if (idPos < 0)
      return;

   // call layout to define the range of the scroll bars
   Layout();

   // SetPosition will send a message which will handled by
   // the function TGListBox::ProcessMessage. Now ProcessMessage will
   // set the viewport. SetPosition also will check that the idPos is
   // not out of range.
   fVScrollbar->SetPosition(idPos);
}

//______________________________________________________________________________
void TGListBox::Resize(UInt_t w, UInt_t h)
{
   // Resize the listbox widget. If fIntegralHeight is true make the height
   // an integral number of the maximum height of a single entry.

   if (fIntegralHeight)
      h = TMath::Max(fItemVsize, ((h - (fBorderWidth << 1)) / fItemVsize) * fItemVsize)
                     + (fBorderWidth << 1);
   TGCompositeFrame::Resize(w, h);
}

//______________________________________________________________________________
void TGListBox::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize the listbox widget.

   if (fIntegralHeight)
      h = TMath::Max(fItemVsize, ((h - (fBorderWidth << 1)) / fItemVsize) * fItemVsize)
                     + (fBorderWidth << 1);
   TGCompositeFrame::MoveResize(x, y, w, h);
}

//______________________________________________________________________________
TGDimension TGListBox::GetDefaultSize() const
{
   // Return default size of listbox widget.

   UInt_t h;

   if (fIntegralHeight)
      h = TMath::Max(fItemVsize, ((fHeight - (fBorderWidth << 1)) / fItemVsize) * fItemVsize)
                     + (fBorderWidth << 1);
   else
      h = fHeight;

   return TGDimension(fWidth, h);
}

//______________________________________________________________________________
void TGListBox::Layout()
{
   // Layout the listbox components.

   TGFrame *container;
   UInt_t   cw, ch, tch;
   Bool_t   need_vsb;

   need_vsb = kFALSE;

   container = fVport->GetContainer();

   // test whether we need vertical scrollbar or not

   cw = fWidth - (fBorderWidth << 1);
   ch = fHeight - (fBorderWidth << 1);

   container->SetWidth(cw);
   container->SetHeight(ch);

   if (container->GetDefaultHeight() > ch) {
      need_vsb = kTRUE;
      cw -= fVScrollbar->GetDefaultWidth();
      if ((Int_t) cw < 0) {
         Warning("Layout", "width would become too small, setting to 10");
         cw = 10;
      }
      container->SetWidth(cw);
   }

   fVport->MoveResize(fBorderWidth, fBorderWidth, cw, ch);
   container->Layout();
   tch = TMath::Max(container->GetDefaultHeight(), ch);
   container->SetHeight(0); // force a resize in TGFrame::Resize
   container->Resize(cw, tch);
   //fVport->SetPos(0, 0);

   if (need_vsb) {
      fVScrollbar->MoveResize(cw+fBorderWidth, fBorderWidth, fVScrollbar->GetDefaultWidth(), ch);
      fVScrollbar->MapWindow();
   } else {
      fVScrollbar->UnmapWindow();
      fVScrollbar->SetPosition(0);
   }

   fVScrollbar->SetRange(container->GetHeight()/fItemVsize, fVport->GetHeight()/fItemVsize);
}

//______________________________________________________________________________
Int_t TGListBox::GetSelected() const
{
   // Return id of selected listbox item.

   TGLBContainer *ct = (TGLBContainer *) fVport->GetContainer();
   return ct->GetSelected();
}

//______________________________________________________________________________
void TGListBox::GetSelectedEntries(TList *selected)
{
   // Adds all selected entries (TGLBEntry) of the list box into
   // the list selected.

   fLbc->GetSelectedEntries(selected);
}

//______________________________________________________________________________
Bool_t TGListBox::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process messages generated by the listbox container and forward
   // messages to the listbox message handling window.

   switch (GET_MSG(msg)) {
      case kC_VSCROLL:
         switch (GET_SUBMSG(msg)) {
            case kSB_SLIDERTRACK:
            case kSB_SLIDERPOS:
               fVport->SetVPos(Int_t(-parm1 * fItemVsize));
               break;
         }
         break;

      case kC_CONTAINER:
         switch (GET_SUBMSG(msg)) {
            case kCT_ITEMCLICK:
               SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_LISTBOX),
                           fListBoxId, parm1);
            break;
         }
         break;

      default:
         break;

   }
   return kTRUE;
}
