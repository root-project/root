// @(#)root/gui:$Name:  $:$Id: TGComboBox.cxx,v 1.13 2004/02/18 20:13:43 brun Exp $
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


ClassImp(TGComboBoxPopup)
ClassImp(TGComboBox)

ClassImp(TGLineStyleComboBox)
ClassImp(TGLineWidthComboBox)
ClassImp(TGFontTypeComboBox)

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

   AddInput(kStructureNotifyMask);
}

//______________________________________________________________________________
Bool_t TGComboBoxPopup::HandleButton(Event_t *event)
{
   // Handle mouse button event in combo box popup.

   if (event->fType == kButtonRelease && event->fCode == kButton1)
      EndPopup();
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
                          kPointerMotionMask, kNone,
                          fClient->GetResourcePool()->GetGrabCursor());

   fClient->WaitForUnmap(this);
   EndPopup();
}


//______________________________________________________________________________
TGComboBox::TGComboBox(const TGWindow *p, Int_t id, UInt_t options,
                       ULong_t back) :
   TGCompositeFrame (p, 10, 10, options | kOwnBackground, back)
{
   // Create a combo box widget.

   fWidgetId  = id;
   fMsgWindow = p;
   fBpic = fClient->GetPicture("arrow_down.xpm");

   if (!fBpic)
      Error("TGComboBox", "arrow_down.xpm not found");

   fSelEntry = new TGTextLBEntry(this, new TGString(""), 0);
   fDDButton = new TGScrollBarElement(this, fBpic, kDefaultScrollBarWidth,
                                      kDefaultScrollBarWidth, kRaisedFrame);

   fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);

   AddFrame(fSelEntry, fLhs = new TGLayoutHints(kLHintsLeft |
                                                kLHintsExpandY | kLHintsExpandX));
                                                //0, 0, 1, 0));
   AddFrame(fDDButton, fLhb = new TGLayoutHints(kLHintsRight |
                                                kLHintsExpandY));

   fComboFrame = new TGComboBoxPopup(fClient->GetDefaultRoot(), 100, 100, kVerticalFrame);

   fListBox = new TGListBox(fComboFrame, fWidgetId, kChildFrame);
   fListBox->Resize(100, 100);
   fListBox->Associate(this);
   fListBox->GetScrollBar()->GrabPointer(kFALSE); // combobox will do a pointergrab

   fComboFrame->AddFrame(fListBox, fLhdd = new TGLayoutHints(kLHintsExpandX |
                                                             kLHintsExpandY));
   fComboFrame->MapSubwindows();
   fComboFrame->Resize(fComboFrame->GetDefaultSize());

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier, kButtonPressMask |
                    kButtonReleaseMask, kNone, kNone);

   // Drop down listbox of combo box should react to pointer motion
   // so it will be able to Activate() (i.e. highlight) the different
   // items when the mouse crosses.
   fListBox->GetContainer()->AddInput(kButtonPressMask | kButtonReleaseMask |
                                      kPointerMotionMask);
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
         gVirtualX->DrawLine(fId, GetShadowGC()(), 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, GetShadowGC()(), 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, GetBlackGC()(), 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, GetBlackGC()(), 1, 1, 1, fHeight-3);

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
                           fWidgetId, parm2);
               if (e->InheritsFrom(TGTextLBEntry::Class())) {
                  const char *text;
                  text = ((TGTextLBEntry*)e)->GetText()->GetString();
                  Selected(text);
               }
               Selected(fWidgetId, (Int_t)parm2);
               Selected((Int_t)parm2);
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
void TGComboBox::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a combo box widget as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // combo box" << endl;
   out << "   TGComboBox *";
   out << GetName() << " = new TGComboBox(" << fParent->GetName() << "," << fWidgetId;

   if (fBackground == GetWhitePixel()) {
      if (GetOptions() == (kHorizontalFrame | kSunkenFrame | kDoubleBorder)) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() << ");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }

   TGTextLBEntry *b;
   TGFrameElement *el;
   TGListBox *lb = GetListBox();

   TIter next(((TGLBContainer *)lb->GetContainer())->GetList());

   while ((el = (TGFrameElement *) next())) {
      b = (TGTextLBEntry *) el->fFrame;
      out << "   " << GetName() << "->AddEntry(";
      b->SavePrimitive(out, option);
      out <<  ");" << endl;
   }

   out << "   " << GetName() << "->Resize(" << GetWidth()  << ","
       << GetHeight() << ");" << endl;
   out << "   " << GetName() << "->Select(" << GetSelected() << ");" << endl;

}

//______________________________________________________________________________
TGLineStyleComboBox::TGLineStyleComboBox(const TGWindow *p, Int_t id,
                                         UInt_t options, Pixel_t back) 
   : TGComboBox(p, id, options, back)
{
   // Create a line style combo box.

   SetTopEntry(new TGLineStyleLBEntry(this, 0, 0),
               new TGLayoutHints(kLHintsLeft | kLHintsExpandY | kLHintsExpandX));
   fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);

   for (int i = 1; i <= 4; i++)
      AddEntry(new TGLineStyleLBEntry(GetListBox()->GetContainer(), i, i), 
               new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   fComboFrame->SetHeight(25);
   Select(1);  // to have first entry selected
}
         
//______________________________________________________________________________
Bool_t TGLineStyleComboBox::HandleButton(Event_t *event)
{
   // Handle mouse button events in a line style combo box.

   if (event->fType == kButtonPress) {
      if ((Window_t)event->fUser[0] == fDDButton->GetId())   // fUser[0] = child window
         fDDButton->SetState(kButtonDown);
   } else {
      int      ax, ay;
      Window_t wdummy;
      fDDButton->SetState(kButtonUp);
      gVirtualX->TranslateCoordinates(fId,(fComboFrame->GetParent())->GetId(),
                                      0, fHeight, ax, ay, wdummy);
      fComboFrame->PlacePopup(ax, ay, fWidth-2, fSelEntry->GetDefaultHeight()*4);
   }
   return kTRUE;
}

//______________________________________________________________________________
TGLineWidthComboBox::TGLineWidthComboBox(const TGWindow *p, Int_t id,
                                         UInt_t options, Pixel_t back) 
   : TGComboBox(p, id, options, back)
{
   // Create a line width combo box.

   SetTopEntry(new TGLineWidthLBEntry(this, 0, 0),
               new TGLayoutHints(kLHintsLeft | kLHintsExpandY | kLHintsExpandX));
   fSelEntry->ChangeOptions(fSelEntry->GetOptions() | kOwnBackground);

   for (int i = 0; i < 16; i++)
      AddEntry(new TGLineWidthLBEntry(GetListBox()->GetContainer(), i, i), 
               new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   Select(1);  // to have first entry selected
}


const char *fonts[][2] = {    //   unix name,     name
   {"",                                            ""                         }, //not used
   {"-*-times-medium-i-*-*-12-*-*-*-*-*-*-*",      "1. times italic"          },
   {"-*-times-bold-r-*-*-12-*-*-*-*-*-*-*",        "2. times bold"            },
   {"-*-times-bold-i-*-*-12-*-*-*-*-*-*-*",        "3. times bold italic"     },
   {"-*-helvetica-medium-r-*-*-12-*-*-*-*-*-*-*",  "4. helvetica"             },
   {"-*-helvetica-medium-o-*-*-12-*-*-*-*-*-*-*",  "5. helvetica italic"      },
   {"-*-helvetica-bold-r-*-*-12-*-*-*-*-*-*-*",    "6. helvetica bold"        },
   {"-*-helvetica-bold-o-*-*-12-*-*-*-*-*-*-*",    "7. helvetica bold italic" },
   {"-*-courier-medium-r-*-*-12-*-*-*-*-*-*-*",    "8. courier"               },
   {"-*-courier-medium-o-*-*-12-*-*-*-*-*-*-*",    "9. courier italic"        },
   {"-*-courier-bold-r-*-*-12-*-*-*-*-*-*-*",      "10. courier bold"         },
   {"-*-courier-bold-o-*-*-12-*-*-*-*-*-*-*",      "11. courier bold italic"  },
   {"-*-symbol-medium-r-*-*-12-*-*-*-*-*-*-*",     "12. symbol"               },
   {"-*-times-medium-r-*-*-12-*-*-*-*-*-*-*",      "13. times"                },
   {NULL, NULL}
};

//______________________________________________________________________________
TGFontTypeComboBox::TGFontTypeComboBox(const TGWindow *p, Int_t id,
                                       UInt_t options, Pixel_t back) :
   TGComboBox(p, id, options, back)
{
   // Create a text font combo box.

   int noFonts = 0;

   for (int i = 1; fonts[i][0] != 0 && noFonts < kMaxFonts; i++) {

      fFonts[noFonts] = gVirtualX->LoadQueryFont(fonts[i][0]);

      if (fFonts[noFonts] == 0)
         fFonts[noFonts] = TGTextLBEntry::GetDefaultFontStruct();

      GCValues_t gval;
      gval.fMask = kGCFont;
      gval.fFont = gVirtualX->GetFontHandle(fFonts[noFonts]);

      AddEntry(new TGTextLBEntry(GetListBox()->GetContainer(), 
               new TGString(fonts[i][1]), i, 
               gClient->GetGC(&gval, kTRUE)->GetGC(), fFonts[noFonts]), 
               new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX));
      noFonts++;
   }
   
   if (noFonts < kMaxFonts - 1)
      ;
   fFonts[noFonts] = 0;
   Select(1);  // to have first entry selected
}

//______________________________________________________________________________
TGFontTypeComboBox::~TGFontTypeComboBox()
{
   // Text font combo box dtor.

   for (int i = 0; i < kMaxFonts && fFonts[i] != 0; i++)
      gVirtualX->DeleteFont(fFonts[i]);
}
