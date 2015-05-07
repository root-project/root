// @(#)root/gui:$Id$
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
// TGComboBox, TGComboBoxPopup                                          //
//                                                                      //
// A combobox (also known as a drop down listbox) allows the selection  //
// of one item out of a list of items. The selected item is visible in  //
// a little window. To view the list of possible items one has to click //
// on a button on the right of the little window. This will drop down   //
// a listbox. After selecting an item from the listbox the box will     //
// disappear and the newly selected item will be shown in the little    //
// window.                                                              //
//                                                                      //
// The TGComboBox is user callable. The TGComboBoxPopup is a service    //
// class of the combobox.                                               //
//                                                                      //
// Selecting an item in the combobox will generate the event:           //
// kC_COMMAND, kCM_COMBOBOX, combobox id, item id.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGComboBox.h"
#include "TGScrollBar.h"
#include "TGPicture.h"
#include "TGResourcePool.h"
#include "Riostream.h"
#include "TGTextEntry.h"
#include "KeySymbols.h"
#include "RConfigure.h"


ClassImp(TGComboBoxPopup)
ClassImp(TGComboBox)

ClassImp(TGLineStyleComboBox)
ClassImp(TGLineWidthComboBox)
ClassImp(TGFontTypeComboBox)

//______________________________________________________________________________
TGComboBoxPopup::TGComboBoxPopup(const TGWindow *p, UInt_t w, UInt_t h,
                                 UInt_t options, ULong_t back) :
   TGCompositeFrame (p, w, h, options, back), fListBox(0), fSelected(0)
{
   // Create a combo box popup frame.

   SetWindowAttributes_t wattr;

   wattr.fMask = kWAOverrideRedirect | kWASaveUnder |
                 kWABorderPixel      | kWABorderWidth;
   wattr.fOverrideRedirect = kTRUE;
   wattr.fSaveUnder = kTRUE;
   wattr.fBorderPixel = fgBlackPixel;
   wattr.fBorderWidth = 1;
   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   AddInput(kStructureNotifyMask);
   fEditDisabled = kEditDisable | kEditDisableGrab  | kEditDisableBtnEnable;
   SetWindowName();
}

//______________________________________________________________________________
Bool_t TGComboBoxPopup::HandleButton(Event_t *event)
{
   // Handle mouse button event in combo box popup.

   if (event->fType == kButtonPress && event->fCode == kButton1) {
      if ((fListBox != 0) && (fSelected != 0) &&
          fListBox->GetSelectedEntry() != fSelected) {
         // in the case the combo box popup is closed by clicking outside the
         // list box, then select the previously selected entry
         fListBox->Select(fSelected->EntryId());
      }
      EndPopup();
   }
   else {
      // reset the dragging flag of the scrollbar when the button is
      // released outside the scrollbar itself
      fListBox->GetScrollBar()->SetDragging(kFALSE);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGComboBoxPopup::EndPopup()
{
   // Ungrab pointer and unmap popup window.

   if (IsMapped()) {
      Handle_t id = fListBox->GetContainer()->GetId();
      gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Up),
                         kAnyModifier, kFALSE);
      gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Down),
                         kAnyModifier, kFALSE);
      gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Enter),
                         kAnyModifier, kFALSE);
      gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Return),
                         kAnyModifier, kFALSE);
      gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Escape),
                         kAnyModifier, kFALSE);
      gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Space),
                         kAnyModifier, kFALSE);
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
      UnmapWindow();
   }
}

//______________________________________________________________________________
void TGComboBoxPopup::PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Popup combo box popup window at the specified place.

   Int_t  rx, ry;
   UInt_t rw, rh;

   // Parent is root window for the popup:
   gVirtualX->GetWindowSize(fParent->GetId(), rx, ry, rw, rh);

   if (x < 0) x = 0;
   if (x + fWidth > rw) x = rw - fWidth;
   if (y < 0) y = 0;
   if (y + fHeight > rh) y = rh - fHeight;

   // remember the current selected entry
   if (fListBox == 0) {
      // the listbox should be the first in the list
      TGFrameElement *el = (TGFrameElement *)fList->First();
      fListBox = dynamic_cast<TGListBox *>(el->fFrame);
   }
   fSelected = fListBox ? fListBox->GetSelectedEntry() : 0;

   MoveResize(x, y, w, h);
   MapSubwindows();
   Layout();
   MapRaised();

   Handle_t id = fListBox->GetContainer()->GetId();
   gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Up),
                      kAnyModifier, kTRUE);
   gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Down),
                      kAnyModifier, kTRUE);
   gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Enter),
                      kAnyModifier, kTRUE);
   gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Return),
                      kAnyModifier, kTRUE);
   gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Escape),
                      kAnyModifier, kTRUE);
   gVirtualX->GrabKey(id, gVirtualX->KeysymToKeycode(kKey_Space),
                      kAnyModifier, kTRUE);
   fListBox->GetContainer()->RequestFocus();

   gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                          kPointerMotionMask, kNone,
                          fClient->GetResourcePool()->GetGrabCursor());

   if (fClient->IsEditable()) {
      fClient->RegisterPopup(this);
   }

   fClient->WaitForUnmap(this);
   EndPopup();
}

//______________________________________________________________________________
void TGComboBoxPopup::KeyPressed(TGFrame *f, UInt_t keysym, UInt_t)
{
   // Slot handling the key press events.

   switch ((EKeySym)keysym) {
      case kKey_Enter:
      case kKey_Return:
      case kKey_Space:
         if (fListBox && f) {
            TGLBEntry *entry = dynamic_cast<TGLBEntry *>(f);
            if (entry) {
               fListBox->Select(entry->EntryId());
               SendMessage(fListBox, MK_MSG(kC_CONTAINER, kCT_ITEMCLICK),
                           entry->EntryId(), 0);
            }
         }
         EndPopup();
         break;
      case kKey_Escape:
         if (fListBox)
            ((TGContainer *)fListBox->GetContainer())->UnSelectAll();
         EndPopup();
         break;
      default:
         break;
   }
}

//______________________________________________________________________________
TGComboBox::TGComboBox(const TGWindow *p, Int_t id, UInt_t options,
                       ULong_t back) :
   TGCompositeFrame (p, 10, 10, options | kOwnBackground, back)
{
   // Create a combo box widget.

   fWidgetId  = id;
   fMsgWindow = p;
   fTextEntry = 0;

   fSelEntry = new TGTextLBEntry(this, new TGString(""), 0);
   fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);

   AddFrame(fSelEntry, fLhs = new TGLayoutHints(kLHintsLeft |
                                                kLHintsExpandY | kLHintsExpandX));
   Init();
}

//______________________________________________________________________________
TGComboBox::TGComboBox(const TGWindow *p, const char *text, Int_t id,
                       UInt_t options, ULong_t back) :
            TGCompositeFrame (p, 10, 10, options | kOwnBackground, back)
{
   // Create an editable combo box widget.

   fWidgetId  = id;
   fMsgWindow = p;
   fSelEntry = 0;

   fTextEntry = new TGTextEntry(this, text, id);
   fTextEntry->SetFrameDrawn(kFALSE);
   fTextEntry->Connect("ReturnPressed()", "TGComboBox", this, "ReturnPressed()");

   AddFrame(fTextEntry, fLhs = new TGLayoutHints(kLHintsLeft |
                                                 kLHintsExpandY | kLHintsExpandX));
   Init();
}

//______________________________________________________________________________
TGComboBox::~TGComboBox()
{
   // Delete a combo box widget.

   fClient->FreePicture(fBpic);

   if (!MustCleanup()) {
      SafeDelete(fDDButton);
      SafeDelete(fSelEntry);
      SafeDelete(fTextEntry);
      SafeDelete(fLhs);
      SafeDelete(fLhb);
   }

   SafeDelete(fLhdd);
   SafeDelete(fListBox);
   if (fComboFrame) {
      fComboFrame->EndPopup();  // force popdown in case of Qt interface
      SafeDelete(fComboFrame);
   }
}

//______________________________________________________________________________
void TGComboBox::Init()
{
   // Initiate the internal classes of a combo box.

   fBpic = fClient->GetPicture("arrow_down.xpm");

   if (!fBpic)
      Error("TGComboBox", "arrow_down.xpm not found");

   fDDButton = new TGScrollBarElement(this, fBpic, kDefaultScrollBarWidth,
                                      kDefaultScrollBarWidth, kRaisedFrame);

   AddFrame(fDDButton, fLhb = new TGLayoutHints(kLHintsRight |
                                                kLHintsExpandY));

   fComboFrame = new TGComboBoxPopup(fClient->GetDefaultRoot(), 100, 100, kVerticalFrame);

   fListBox = new TGListBox(fComboFrame, fWidgetId, kChildFrame);

   fListBox->Resize(100, 100);
   fListBox->Associate(this);
   fListBox->GetScrollBar()->GrabPointer(kFALSE); // combobox will do a pointergrab

   fComboFrame->AddFrame(fListBox, fLhdd = new TGLayoutHints(kLHintsExpandX |
                                                             kLHintsExpandY));
   fComboFrame->SetListBox(fListBox);
   fComboFrame->MapSubwindows();
   fComboFrame->Resize(fComboFrame->GetDefaultSize());

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier, kButtonPressMask |
                         kButtonReleaseMask | kPointerMotionMask, kNone, kNone);

   fListBox->GetContainer()->Connect("KeyPressed(TGFrame*, UInt_t, UInt_t)",
                                     "TGComboBoxPopup", fComboFrame,
                                     "KeyPressed(TGFrame*, UInt_t, UInt_t)");
   // Drop down listbox of combo box should react to pointer motion
   // so it will be able to Activate() (i.e. highlight) the different
   // items when the mouse crosses.
   fListBox->GetContainer()->AddInput(kButtonPressMask | kButtonReleaseMask |
                                      kPointerMotionMask);

   fListBox->SetEditDisabled(kEditDisable);
   fListBox->GetContainer()->SetEditDisabled(kEditDisable);
   if (fSelEntry) fSelEntry->SetEditDisabled(kEditDisable | kEditDisableEvents | kEditDisableGrab);
   if (fTextEntry) fTextEntry->SetEditDisabled(kEditDisable | kEditDisableGrab | kEditDisableBtnEnable);
   fDDButton->SetEditDisabled(kEditDisable | kEditDisableGrab);
   fEditDisabled = kEditDisableLayout | kEditDisableBtnEnable | kEditDisableHeight;

   SetWindowName();
}

//______________________________________________________________________________
void TGComboBox::DrawBorder()
{
   // Draw border of combo box widget.

   switch (fOptions & (kSunkenFrame | kRaisedFrame | kDoubleBorder)) {
      case kSunkenFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, GetShadowGC()(), 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, GetShadowGC()(), 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, GetBlackGC()(), 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, GetBlackGC()(), 1, 1, 1, fHeight-3);
         if (gClient->GetStyle() > 1) break;
         gVirtualX->DrawLine(fId, GetHilightGC()(), 0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, GetHilightGC()(), fWidth-1, fHeight-1, fWidth-1, 0);
         gVirtualX->DrawLine(fId, GetBckgndGC()(),  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, GetBckgndGC()(),  fWidth-2, 1, fWidth-2, fHeight-2);
         break;

      default:
         TGCompositeFrame::DrawBorder();
         break;
   }
}

//______________________________________________________________________________
void TGComboBox::EnableTextInput(Bool_t on)
{
   // Switch text input or readonly mode of combobox (not perfect yet).

   // UInt_t w, h;
   const char *text = "";
   Pixel_t back = TGFrame::GetWhitePixel(); // default

   if (on) {
      if (fSelEntry) {
         back = fSelEntry->GetBackground();
         text = ((TGTextLBEntry*)fSelEntry)->GetText()->GetString();
         if (fTextEntry && fSelEntry->InheritsFrom(TGTextLBEntry::Class())) {
            fTextEntry->SetText(text);
         }
         RemoveFrame(fSelEntry);
         //w = fSelEntry->GetWidth();
         //h = fSelEntry->GetHeight();
         fSelEntry->DestroyWindow();
         delete fSelEntry;
         fSelEntry = 0;
      }
      if (!fTextEntry) {
         fTextEntry = new TGTextEntry(this, text, 0);
         fTextEntry->SetFrameDrawn(kFALSE);
         fTextEntry->Connect("ReturnPressed()", "TGComboBox", this, "ReturnPressed()");
         AddFrame(fTextEntry, fLhs);
         fTextEntry->SetEditDisabled(kEditDisable | kEditDisableGrab | kEditDisableBtnEnable);
      }
      fTextEntry->SetBackgroundColor(back);
      MapSubwindows();
      // coverity[returned_null]
      // coverity[dereference]
      GetLayoutManager()->Layout();
   } else {
      if (fTextEntry) {
         back = fTextEntry->GetBackground();
         text = fTextEntry->GetText();
         RemoveFrame(fTextEntry);
         fTextEntry->DestroyWindow();
         //w = fTextEntry->GetWidth();
         //h = fTextEntry->GetHeight();
         delete fTextEntry;
         fTextEntry = 0;
      }
      if (!fSelEntry) {
         fSelEntry = new TGTextLBEntry(this, new TGString(text), 0);
         fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);
         AddFrame(fSelEntry, fLhs);
         fSelEntry->SetEditDisabled(kEditDisable | kEditDisableGrab);
      }
      fSelEntry->SetBackgroundColor(back);
      MapSubwindows();
      // coverity[returned_null]
      // coverity[dereference]
      GetLayoutManager()->Layout();
   }
}

//______________________________________________________________________________
TGLBEntry *TGComboBox::FindEntry(const char *s) const
{
   // Find entry by name.

   TGLBEntry *sel = 0;
   sel = fListBox->FindEntry(s);
   return sel;
}

//______________________________________________________________________________
void TGComboBox::SetTopEntry(TGLBEntry *e, TGLayoutHints *lh)
{
   // Set a new combo box value (normally update of text string in
   // fSelEntry is done via fSelEntry::Update()).

   if (!fSelEntry) return;

   RemoveFrame(fSelEntry);
   fSelEntry->DestroyWindow();
   delete fSelEntry;
   delete fLhs;
   fSelEntry = e;
   fLhs = lh;
   AddFrame(fSelEntry, fLhs);
   Layout();
}

//______________________________________________________________________________
void TGComboBox::Select(Int_t id, Bool_t emit)
{
   // Make the selected item visible in the combo box window
   // and emit signals according to the second parameter.

   if (id!=GetSelected()) {
      TGLBEntry *e;
      e = fListBox->Select(id);
      if (e) {
         if (fSelEntry) {
            fSelEntry->Update(e);
            Layout();
            if (emit) {
               Selected(fWidgetId, id);
               Selected(id);
            }
         }
      }
   }
}

//______________________________________________________________________________
Bool_t TGComboBox::HandleButton(Event_t *event)
{
   // Handle mouse button events in the combo box.

   if (!fDDButton || !fDDButton->IsEnabled()) return kFALSE;

   if (event->fType == kButtonPress) {
      Window_t child = (Window_t)event->fUser[0];  // fUser[0] = child window

      if (child == fDDButton->GetId() || (fSelEntry && child == fSelEntry->GetId())) {
         fDDButton->SetState(kButtonDown);

         if (fTextEntry && (child == fTextEntry->GetId())) {
            return fTextEntry->HandleButton(event);
         }
         int      ax, ay;
         Window_t wdummy;
         gVirtualX->TranslateCoordinates(fId, fComboFrame->GetParent()->GetId(),
                                         0, fHeight, ax, ay, wdummy);
         // Drop down listbox of combo box should react to pointer motion...
         fListBox->GetContainer()->AddInput(kPointerMotionMask);
#ifdef R__HAS_COCOA
         gVirtualX->SetWMTransientHint(fComboFrame->GetId(), GetId());
#endif
         fComboFrame->PlacePopup(ax, ay, fWidth-2, fComboFrame->GetDefaultHeight());
         fDDButton->SetState(kButtonUp);
#ifdef R__HAS_COCOA
         //tp: I need this modification - "button" is not repainted correctly
         //with Cocoa, when combobox is closed (reason is quite complex), happens
         //when item is wider than combobox.
         //TODO: find another way :)
         fClient->NeedRedraw(fDDButton);
#endif
      } else if (fTextEntry) {
         return fTextEntry->HandleButton(event);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGComboBox::RemoveEntry(Int_t id)
{
   // Remove entry. If id == -1, the currently selected entry is removed

   fListBox->RemoveEntry(id);

   if (id < 0) {
      if (fSelEntry) {
         ((TGTextLBEntry*)fSelEntry)->SetTitle("");
         fClient->NeedRedraw(fSelEntry);
      } else {
         fTextEntry->SetTitle("");
         fClient->NeedRedraw(fTextEntry);
      }
   }
   Resize();
}

//______________________________________________________________________________
void TGComboBox::Layout()
{
   // layout combobox

   TGCompositeFrame::Layout();
   UInt_t h = fListBox->GetNumberOfEntries()*fListBox->GetItemVsize();

   if (h && (h < 100)) {
      fListBox->Resize(fListBox->GetWidth(), h);
   }
}

//______________________________________________________________________________
Bool_t TGComboBox::HandleDoubleClick(Event_t *event)
{
   // Handle double click in text entry.

   return fTextEntry ? fTextEntry->HandleDoubleClick(event) : kTRUE;
}

//______________________________________________________________________________
Bool_t TGComboBox::HandleMotion(Event_t *event)
{
   // Handle pointer motion in text entry.

   return fTextEntry ? fTextEntry->HandleMotion(event) : kTRUE;
}

//______________________________________________________________________________
Bool_t TGComboBox::HandleSelection(Event_t *event)
{
   // Handle selection in text entry.

   return fTextEntry ? fTextEntry->HandleSelection(event) : kTRUE;
}

//______________________________________________________________________________
Bool_t TGComboBox::HandleSelectionRequest(Event_t *event)
{
   // Handle selection request in text entry.

   return fTextEntry ? fTextEntry->HandleSelectionRequest(event) : kTRUE;
}

//______________________________________________________________________________
Bool_t TGComboBox::ProcessMessage(Long_t msg, Long_t, Long_t parm2)
{
   // Process messages generated by the listbox and forward
   // messages to the combobox message handling window. Parm2 contains
   // the id of the selected listbox entry.

   TGLBEntry *e;

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_LISTBOX:
               e = fListBox->GetSelectedEntry();
               if (fSelEntry) {
                  fSelEntry->Update(e);
               } else if (fTextEntry &&
                          e->InheritsFrom(TGTextLBEntry::Class())) {
                  TGTextLBEntry *te = (TGTextLBEntry*)e;
                  fTextEntry->SetText(te->GetText()->GetString());
               }
               // coverity[returned_null]
               // coverity[dereference]
               GetLayoutManager()->Layout();
               fComboFrame->EndPopup();
               fDDButton->SetState(kButtonUp);
               SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_COMBOBOX),
                           fWidgetId, parm2);
               if (e->InheritsFrom(TGTextLBEntry::Class())) {
                  const char *text;
                  text = ((TGTextLBEntry*)e)->GetText()->GetString();
                  Selected(text);
               }
               Selected(fWidgetId, (Int_t)parm2);
               Selected((Int_t)parm2);
               fClient->NeedRedraw(this);
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGComboBox::Selected(Int_t widgetId, Int_t id)
{
   // Emit signal.

   Long_t args[2];

   args[0] = widgetId;
   args[1] = id;

   Emit("Selected(Int_t,Int_t)", args);
}

//______________________________________________________________________________
void TGComboBox::SetEnabled(Bool_t on)
{
   // Set state of combo box. If kTRUE=enabled, kFALSE=disabled.

   fDDButton->SetEnabled(on);
   if (on) {
      SetFlags(kWidgetIsEnabled);
      fSelEntry->SetBackgroundColor(GetBackground());
   } else {
      ClearFlags(kWidgetIsEnabled);
      fSelEntry->SetBackgroundColor(GetDefaultFrameBackground());
   }
   fClient->NeedRedraw(fSelEntry);
}

//______________________________________________________________________________
void TGComboBox::ReturnPressed()
{
   // Add new entry to combo box when return key pressed inside text entry
   // ReturnPressed signal is emitted.

   if (!fTextEntry) return;

   TGLBContainer *lbc = (TGLBContainer *)fListBox->GetContainer();
   TString text = fTextEntry->GetText();

   TIter next(lbc->GetList());
   TGFrameElement *el;

   Emit("ReturnPressed()");

   while ((el = (TGFrameElement *)next())) {
      TGTextLBEntry *lbe = (TGTextLBEntry *)el->fFrame;
      if (lbe->GetText()->GetString() == text) {
         return;
      }
   }

   Int_t nn = GetNumberOfEntries() + 1;
   AddEntry(text.Data(), nn);
   Select(nn);
}

//______________________________________________________________________________
void TGComboBox::RemoveAll()
{
   // Remove all entries from combo box.

   fListBox->RemoveAll();

   if (fSelEntry) {
      ((TGTextLBEntry*)fSelEntry)->SetTitle("");
      fClient->NeedRedraw(fSelEntry);
   } else {
      fTextEntry->SetTitle("");
      fClient->NeedRedraw(fTextEntry);
   }
}

//______________________________________________________________________________
void TGComboBox::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   // Save a combo box widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << std::endl << "   // combo box" << std::endl;
   out << "   TGComboBox *";

   if (!fTextEntry) {
      out << GetName() << " = new TGComboBox(" << fParent->GetName() << "," << fWidgetId;
   } else {
      out << GetName() << " = new TGComboBox(" << fParent->GetName() << ",";
      out << '\"' <<  fTextEntry->GetText() << '\"' << "," <<fWidgetId;
   }

   if (fBackground == GetWhitePixel()) {
      if (GetOptions() == (kHorizontalFrame | kSunkenFrame | kDoubleBorder)) {
         out <<");" << std::endl;
      } else {
         out << "," << GetOptionString() << ");" << std::endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << std::endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << std::endl;

   TGTextLBEntry *b;
   TGFrameElement *el;
   TGListBox *lb = GetListBox();

   TIter next(((TGLBContainer *)lb->GetContainer())->GetList());

   while ((el = (TGFrameElement *) next())) {
      b = (TGTextLBEntry *) el->fFrame;
      out << "   " << GetName() << "->AddEntry(";
      b->SavePrimitive(out, option);
      out <<  ");" << std::endl;
   }

   out << "   " << GetName() << "->Resize(" << GetWidth()  << ","
       << GetHeight() << ");" << std::endl;
   out << "   " << GetName() << "->Select(" << GetSelected() << ");" << std::endl;
}

//______________________________________________________________________________
TGLineStyleComboBox::TGLineStyleComboBox(const TGWindow *p, Int_t id,
                                         UInt_t options, Pixel_t back)
   : TGComboBox(p, id, options, back)
{
   // Create a line style combo box.

   SetTopEntry(new TGLineLBEntry(this, 0),
               new TGLayoutHints(kLHintsLeft | kLHintsExpandY | kLHintsExpandX));
   fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);

   for (Int_t i = 1; i <= 10; i++)
      AddEntry(new TGLineLBEntry(GetListBox()->GetContainer(), i,
               TString::Format("%d",i), 0, i),
               new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   Select(1, kFALSE);  // to have first entry selected

   SetWindowName();
}

//______________________________________________________________________________
void TGLineStyleComboBox::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   // Save a line style combo box widget as a C++ statement(s).

   out << std::endl << "   // line style combo box" << std::endl;
   out << "   TGLineStyleComboBox *";

   out << GetName() << " = new TGLineStyleComboBox(" << fParent->GetName()
       << "," << fWidgetId << ");" << std::endl;
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << std::endl;
   out << "   " << GetName() << "->Resize(" << GetWidth()  << ","
       << GetHeight() << ");" << std::endl;
   out << "   " << GetName() << "->Select(" << GetSelected() << ");" << std::endl;
}

//______________________________________________________________________________
TGLineWidthComboBox::TGLineWidthComboBox(const TGWindow *p, Int_t id,
                                         UInt_t options, Pixel_t back, Bool_t none)
   : TGComboBox(p, id, options, back)
{
   // Create a line width combo box.
   // If "none" is equal to kTRUE the first entry is "None".

   SetTopEntry(new TGLineLBEntry(this,0),
               new TGLayoutHints(kLHintsLeft | kLHintsExpandY | kLHintsExpandX));
   fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);

   if (none) {
      AddEntry(new TGLineLBEntry(GetListBox()->GetContainer(), 0, "None", 0, 0),
               new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   }

   for (Int_t i = 0; i < 16; i++)
      AddEntry(new TGLineLBEntry(GetListBox()->GetContainer(), i,
               TString::Format("%d",i), i, 0),
               new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   Select(1, kFALSE);  // to have first entry selected
   SetWindowName();
}

//______________________________________________________________________________
void TGLineWidthComboBox::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   // Save a line width combo box widget as a C++ statement(s).

   out << std::endl << "   // line width combo box" << std::endl;
   out << "   TGLineWidthComboBox *";

   out << GetName() << " = new TGLineWidthComboBox(" << fParent->GetName()
       << "," << fWidgetId << ");" << std::endl;
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << std::endl;
   out << "   " << GetName() << "->Resize(" << GetWidth()  << ","
       << GetHeight() << ");" << std::endl;
   out << "   " << GetName() << "->Select(" << GetSelected() << ");" << std::endl;
}

static const char *gFonts[][2] = {    //   unix name,     name
   { "",                                           ""                         }, //not used
   { "-*-times-medium-i-*-*-12-*-*-*-*-*-*-*",     "1. times italic"          },
   { "-*-times-bold-r-*-*-12-*-*-*-*-*-*-*",       "2. times bold"            },
   { "-*-times-bold-i-*-*-12-*-*-*-*-*-*-*",       "3. times bold italic"     },
   { "-*-helvetica-medium-r-*-*-12-*-*-*-*-*-*-*", "4. helvetica"             },
   { "-*-helvetica-medium-o-*-*-12-*-*-*-*-*-*-*", "5. helvetica italic"      },
   { "-*-helvetica-bold-r-*-*-12-*-*-*-*-*-*-*",   "6. helvetica bold"        },
   { "-*-helvetica-bold-o-*-*-12-*-*-*-*-*-*-*",   "7. helvetica bold italic" },
   { "-*-courier-medium-r-*-*-12-*-*-*-*-*-*-*",   "8. courier"               },
   { "-*-courier-medium-o-*-*-12-*-*-*-*-*-*-*",   "9. courier italic"        },
   { "-*-courier-bold-r-*-*-12-*-*-*-*-*-*-*",     "10. courier bold"         },
   { "-*-courier-bold-o-*-*-12-*-*-*-*-*-*-*",     "11. courier bold italic"  },
   { "-*-symbol-medium-r-*-*-12-*-*-*-*-*-*-*",    "12. symbol"               },
   { "-*-times-medium-r-*-*-12-*-*-*-*-*-*-*",     "13. times"                },
   { 0, 0}
};

//______________________________________________________________________________
TGFontTypeComboBox::TGFontTypeComboBox(const TGWindow *p, Int_t id,
                                       UInt_t options, Pixel_t back) :
   TGComboBox(p, id, options, back)
{
   // Create a text font combo box.

   Int_t noFonts = 0;

   for (Int_t i = 1; gFonts[i][0] != 0 && noFonts < kMaxFonts; i++) {

      fFonts[noFonts] = gVirtualX->LoadQueryFont(gFonts[i][0]);

      if (fFonts[noFonts] == 0)
         fFonts[noFonts] = TGTextLBEntry::GetDefaultFontStruct();

      GCValues_t gval;
      gval.fMask = kGCFont;
      gval.fFont = gVirtualX->GetFontHandle(fFonts[noFonts]);

      AddEntry(new TGTextLBEntry(GetListBox()->GetContainer(),
               new TGString(gFonts[i][1]), i,
               fClient->GetGC(&gval, kTRUE)->GetGC(), fFonts[noFonts]),
               new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX));
      noFonts++;
   }

   if (noFonts < kMaxFonts - 1)
      fFonts[noFonts] = 0;

   Select(1, kFALSE);  // to have first entry selected
   SetWindowName();
}

//______________________________________________________________________________
TGFontTypeComboBox::~TGFontTypeComboBox()
{
   // Text font combo box dtor.

   for (int i = 0; i < kMaxFonts && fFonts[i] != 0; i++) {
      if (fFonts[i] != TGTextLBEntry::GetDefaultFontStruct()) gVirtualX->DeleteFont(fFonts[i]);
   }
}
