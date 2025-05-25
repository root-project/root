// @(#)root/gui:$Id$
// Author: Fons Rademakers   18/9/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TGShutter
    \ingroup guiwidgets

A shutter widget contains a set of shutter items that can be
open and closed like a shutter.
This widget is usefull to group a large number of options in
a number of categories.

*/


#include "TGShutter.h"
#include "TGButton.h"
#include "TList.h"
#include "TTimer.h"

#include <iostream>


ClassImp(TGShutterItem);
ClassImp(TGShutter);

////////////////////////////////////////////////////////////////////////////////
/// Create shutter frame.

TGShutter::TGShutter(const TGWindow *p, UInt_t options) :
   TGCompositeFrame(p, 10, 10, options)
{
   fSelectedItem        = 0;
   fClosingItem         = 0;
   fHeightIncrement     = 1;
   fClosingHeight       = 0;
   fClosingHadScrollbar = kFALSE;
   fTimer               = 0;
   fTrash               = new TList;

   fDefWidth = fDefHeight = 0;

   // layout manager is not used
   delete fLayoutManager;
   fLayoutManager = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup shutter widget.

TGShutter::~TGShutter()
{
   if (fTimer) delete fTimer;

   if (!MustCleanup()) {
      fTrash->Delete();
   }
   delete fTrash;
   fTrash = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Add shutter item to shutter frame.

void TGShutter::AddItem(TGShutterItem *item)
{
   TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   AddFrame(item, hints);
   fTrash->Add(hints);
   if (!fSelectedItem) {
      fSelectedItem = item;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove item from shutter

void TGShutter::RemoveItem(const char *name)
{
   TGShutterItem *item = GetItem(name);

   if (!item) {
      return;
   }

   if (fList->GetEntries() <= 1) {
      return;
   }

   if (item == fSelectedItem) {
      TGFrameElement *fe = (TGFrameElement*)fList->FindObject(item->GetFrameElement());
      if (fe) {
         TGFrameElement *sel = (TGFrameElement*)fList->Before(fe);
         if (!sel) {
            sel = (TGFrameElement*)fList->After(fe);
         }
         if (!sel) {
            return;
         }
         SetSelectedItem((TGShutterItem*)sel->fFrame);
      }
   }
   RemoveFrame(item);

   item->DestroyWindow();
   delete item;
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove selected page

void TGShutter::RemovePage()
{
   if (!fSelectedItem) {
      return;
   }
   TGTextButton *btn = (TGTextButton*)fSelectedItem->GetButton();
   RemoveItem(btn->GetString().Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Rename selected page

void TGShutter::RenamePage(const char *name)
{
   if (!fSelectedItem) {
      return;
   }
   TGTextButton *btn = (TGTextButton*)fSelectedItem->GetButton();
   btn->SetText(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Add new page (shutter item)

TGShutterItem *TGShutter::AddPage(const char *name)
{
   static int id = 1000;
   TGShutterItem *item = new TGShutterItem(this, new TGHotString(name), id++);
   AddItem(item);
   MapSubwindows();
   Layout();
   return item;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle shutter messages.

Bool_t TGShutter::ProcessMessage(Longptr_t /*msg*/, Longptr_t parm1, Longptr_t /*parm2*/)
{
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
   Selected(fSelectedItem);
   fSelectedItem->Selected();

   if (!fTimer) fTimer = new TTimer(this, 6); //10);
   fTimer->Reset();
   fTimer->TurnOn();

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Shutter item animation.

Bool_t TGShutter::HandleTimer(TTimer *)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Layout shutter items.

void TGShutter::Layout()
{
   TGFrameElement *el;
   TGShutterItem  *child;
   Int_t y, bh, exh;

   if (!fList) return;

   if (!fSelectedItem)
      fSelectedItem = (TGShutterItem*) ((TGFrameElement*)GetList()->First())->fFrame;

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

////////////////////////////////////////////////////////////////////////////////
/// Set item to be the currently open shutter item.

void TGShutter::SetSelectedItem(TGShutterItem *item)
{
   fSelectedItem = item;
   fSelectedItem->Selected(); // emit signal
   Layout();
}

////////////////////////////////////////////////////////////////////////////////
/// Set item to be the currently open shutter item.

void TGShutter::SetSelectedItem(const char *name)
{
   TGShutterItem *item = GetItem(name);
   if (!item) {
      return;
   }
   SetSelectedItem(item);
}

////////////////////////////////////////////////////////////////////////////////
/// Disable/enbale shutter item.

void TGShutter::EnableItem(const char *name, Bool_t on)
{
   TGShutterItem *item = GetItem(name);
   if (!item) {
      return;
   }

   item->GetButton()->SetEnabled(on);
}

////////////////////////////////////////////////////////////////////////////////
/// returns a shutter item by name (name is hot string of shutter item)

TGShutterItem *TGShutter::GetItem(const char *name)
{
   TGFrameElement *el;
   TGShutterItem  *item = 0;

   TIter next(fList);

   while ((el = (TGFrameElement *) next())) {
      TGTextButton *btn;
      item = (TGShutterItem *)el->fFrame;
      btn = (TGTextButton*)item->GetButton();
      if (btn->GetString() == name) return item;
   }

   return item;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the default / minimal size of the widget.

TGDimension TGShutter::GetDefaultSize() const
{
   UInt_t w = (GetOptions() & kFixedWidth)  || (fDefWidth  == 0) ? fWidth  : fDefWidth;
   UInt_t h = (GetOptions() & kFixedHeight) || (fDefHeight == 0) ? fHeight : fDefHeight;
   return TGDimension(w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the default / minimal size of the widget.

void TGShutter::SetDefaultSize(UInt_t w, UInt_t h)
{
   fDefWidth  = w;
   fDefHeight = h;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a shutter item.

TGShutterItem::TGShutterItem(const TGWindow *p, TGHotString *s, Int_t id,
                             UInt_t options) :
   TGVerticalFrame (p, 10, 10, options), TGWidget (id)
{
   if (!p && !s) {
      MakeZombie();
      // coverity [uninit_ctor]
      return;
   }
   fButton = new TGTextButton(this, s, id);
   fCanvas = new TGCanvas(this, 10, 10, kChildFrame);
   fContainer = new TGVerticalFrame(fCanvas->GetViewPort(), 10, 10, kOwnBackground);
   fCanvas->SetContainer(fContainer);
   fContainer->SetBackgroundColor(fClient->GetShadow(GetDefaultFrameBackground()));

   AddFrame(fButton, fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   AddFrame(fCanvas, fL2 = new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));

   fButton->Associate((TGFrame *) p);

   fCanvas->SetEditDisabled(kEditDisableGrab | kEditDisableLayout);
   fButton->SetEditDisabled(kEditDisableGrab | kEditDisableBtnEnable);
   fContainer->SetEditDisabled(kEditDisableGrab);
   fEditDisabled = kEditDisableGrab | kEditDisableLayout;
}

////////////////////////////////////////////////////////////////////////////////
/// Clan up shutter item.

TGShutterItem::~TGShutterItem()
{
   if (!IsZombie() && !MustCleanup()) {
      delete fL1;
      delete fL2;
      delete fButton;
      delete fContainer;
      delete fCanvas;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save a shutter item widget as a C++ statement(s) on output stream out

void TGShutterItem::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   TGTextButton *b = (TGTextButton *)fButton;
   TString outtext = b->GetText()->GetString();
   Int_t hotpos = b->GetText()->GetHotPos();
   if ((hotpos > 0) && (hotpos < outtext.Length()))
      outtext.Insert(hotpos - 1, "&");

   out << "\n   // \"" << outtext << "\" shutter item \n";
   out << "   TGShutterItem *" << GetName() << " = new TGShutterItem(" << fParent->GetName() << ", new TGHotString(\""
       << outtext.ReplaceSpecialCppChars() << "\"), " << fButton->WidgetId() << ", " << GetOptionString()
       << ");\n";

   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");\n";

   TList *list = ((TGCompositeFrame *)GetContainer())->GetList();

   if (!list)
      return;

   out << "   TGCompositeFrame *" << GetContainer()->GetName() << " = (TGCompositeFrame *)" << GetName()
       << "->GetContainer();\n";

   TIter next(list);
   while (auto el = static_cast<TGFrameElement *>(next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetContainer()->GetName() << "->AddFrame(" << el->fFrame->GetName();
      el->fLayout->SavePrimitive(out, option);
      out << ");\n";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save a shutter widget as a C++ statement(s) on output stream out.

void TGShutter::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   out << "\n   // shutter\n";

   out << "   TGShutter *" << GetName() << " = new TGShutter(" << fParent->GetName() << "," << GetOptionString()
       << ");\n";

   if ((fDefWidth > 0) || (fDefHeight > 0)) {
      out << "   " << GetName() << "->SetDefaultSize(";
      out << fDefWidth << "," << fDefHeight << ");\n";
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");\n";

   if (!fList)
      return;

   TIter next(fList);

   while (auto el = static_cast<TGFrameElement *>(next())) {
      el->fFrame->SavePrimitive(out, option);
      out << "   " << GetName() << "->AddItem(" << el->fFrame->GetName();
      // el->fLayout->SavePrimitive(out, option);
      out << ");\n";
   }

   out << "   " << GetName() << "->SetSelectedItem(" << GetSelectedItem()->GetName() << ");\n";
   out << "   " << GetName() << "->Resize(" << GetWidth() << "," << GetHeight() << ");\n";
}
