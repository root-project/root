// @(#)root/gui:$Name$:$Id$
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

ClassImp(TGComboBoxPopup)
ClassImp(TGComboBox)


//______________________________________________________________________________
TGComboBoxPopup::TGComboBoxPopup(const TGWindow *p, UInt_t w, UInt_t h,
                                 UInt_t options, ULong_t back) :
   TGCompositeFrame (p, w, h, options, back)
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

   gVirtualX->SelectInput(fId, kStructureNotifyMask);
}

//______________________________________________________________________________
Bool_t TGComboBoxPopup::HandleButton(Event_t *event)
{
   // Handle mouse button event in combo box popup.

   if (event->fType == kButtonRelease) EndPopup();
   return kTRUE;
}

//______________________________________________________________________________
void TGComboBoxPopup::EndPopup()
{
   // Ungrab pointer and unmap popup window.

   gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
   UnmapWindow();
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

   MoveResize(x, y, w, h);
   MapSubwindows();
   Layout();
   MapRaised();

   gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                     kPointerMotionMask, kNone, fgDefaultCursor);

   fClient->WaitForUnmap(this);
   EndPopup();
}


//______________________________________________________________________________
TGComboBox::TGComboBox(const TGWindow *p, Int_t id, UInt_t options,
                       ULong_t back) :
   TGCompositeFrame (p, 10, 10, options, back)
{
   // Create a combo box widget.

   fComboBoxId = id;
   fMsgWindow = p;
   fBpic = fClient->GetPicture("arrow_down.xpm");

   if (!fBpic)
      Error("TGComboBox", "arrow_down.xpm not found");

   fSelEntry = new TGTextLBEntry(this, new TGString(""), 0);
   fDDButton = new TGScrollBarElement(this, fBpic, kDefaultScrollBarWidth,
                                      kDefaultScrollBarWidth, kRaisedFrame);

   AddFrame(fSelEntry, fLhs = new TGLayoutHints(kLHintsLeft |
                                                kLHintsExpandY | kLHintsExpandX));
                                                //0, 0, 1, 0));
   AddFrame(fDDButton, fLhb = new TGLayoutHints(kLHintsRight |
                                                kLHintsExpandY));

   fComboFrame = new TGComboBoxPopup(fClient->GetRoot(), 100, 100, kVerticalFrame);

   fListBox = new TGListBox(fComboFrame, fComboBoxId, kChildFrame);
   fListBox->Resize(100, 100);
   fListBox->Associate(this);

   fComboFrame->AddFrame(fListBox, fLhdd = new TGLayoutHints(kLHintsExpandX |
                                                             kLHintsExpandY));
   fComboFrame->MapSubwindows();
   fComboFrame->Resize(fComboFrame->GetDefaultSize());

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier, kButtonPressMask |
                    kButtonReleaseMask, kNone, kNone);

   // Drop down listbox of combo box should react to pointer motion
   // so it will be able to Activate() (i.e. highlight) the different
   // items when the mouse crosses.
   gVirtualX->SelectInput(fListBox->GetContainer()->GetId(), kButtonPressMask |
                     kButtonReleaseMask | kPointerMotionMask);
}

//______________________________________________________________________________
TGComboBox::~TGComboBox()
{
   // Delete a combo box widget.

   delete fDDButton;
   delete fSelEntry;
   delete fListBox;
   delete fComboFrame;
   delete fLhs;
   delete fLhb;
   delete fLhdd;
}

//______________________________________________________________________________
void TGComboBox::DrawBorder()
{
   // Draw border of combo box widget.

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
         TGCompositeFrame::DrawBorder();
         break;
   }
}

//______________________________________________________________________________
void TGComboBox::SetTopEntry(TGLBEntry *e, TGLayoutHints *lh)
{
   // Set a new combo box value (normally update of text string in
   // fSelEntry is done via fSelEntry::Update()).

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
void TGComboBox::Select(Int_t id)
{
   // Make the selected item visible in the combo box window.

   TGLBEntry *e;

   e = fListBox->Select(id);
   if (e) {
      fSelEntry->Update(e);
      Layout();
   }
}

//______________________________________________________________________________
Bool_t TGComboBox::HandleButton(Event_t *event)
{
   // Handle mouse button events in the combo box.

   if (event->fType == kButtonPress) {
      if ((Window_t)event->fUser[0] == fDDButton->GetId())   // fUser[0] = child window
         fDDButton->SetState(kButtonDown);
   } else {
      int      ax, ay;
      Window_t wdummy;

      fDDButton->SetState(kButtonUp);
      gVirtualX->TranslateCoordinates(fId, (fComboFrame->GetParent())->GetId(),
                                 0, fHeight, ax, ay, wdummy);

      fComboFrame->PlacePopup(ax, ay, fWidth-2, fComboFrame->GetDefaultHeight());
   }
   return kTRUE;
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
               fSelEntry->Update(e);
               Layout();
               fComboFrame->EndPopup();
               SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_COMBOBOX),
                           fComboBoxId, parm2);
            break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}
