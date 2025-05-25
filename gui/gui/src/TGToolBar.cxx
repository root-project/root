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


/** \class TGToolBar
\ingroup guiwidgets

A toolbar is a composite frame that contains TGPictureButtons.
Often used in combination with a TGHorizontal3DLine.

*/


#include "TGToolBar.h"
#include "TList.h"
#include "TGButton.h"
#include "TGPicture.h"
#include "TGToolTip.h"
#include "TSystem.h"
#include "TROOT.h"
#include <iostream>
#include "TMap.h"


ClassImp(TGToolBar);

////////////////////////////////////////////////////////////////////////////////

TGToolBar::TGToolBar(const TGWindow *p, UInt_t w, UInt_t h,
                     UInt_t options, Pixel_t back) :
                     TGCompositeFrame(p, w, h, options, back)

{
   // Create toolbar widget.

   fPictures = new TList;
   fTrash    = new TList;
   fMapOfButtons = new TMap();  // map of button/id pairs

   SetWindowName();
}

////////////////////////////////////////////////////////////////////////////////
/// Delete toolbar and its buttons and layout hints.

TGToolBar::~TGToolBar()
{
   if (!MustCleanup()) {
      if (fTrash) fTrash->Clear("nodelete");
   }
   delete fTrash;
   fTrash = 0;

   TIter next(fPictures);
   const TGPicture *p;
   while ((p = (const TGPicture *) next()))
      fClient->FreePicture(p);

   // pictures might already have been deleted above, so avoid access
   // to these objects
   fPictures->Clear("nodelete");

   delete fPictures;
   delete fMapOfButtons;
}

////////////////////////////////////////////////////////////////////////////////
/// Add button to toolbar. All buttons added via this method will be
/// deleted by the toolbar. On return the TGButton field of the
/// ToolBarData_t struct is filled in (if fPixmap was valid).
/// Window w is the window to which the button messages will be send.

TGButton *TGToolBar::AddButton(const TGWindow *w, ToolBarData_t *button, Int_t spacing)
{
   const TGPicture *pic = fClient->GetPicture(button->fPixmap);
   if (!pic) {
      Error("AddButton", "pixmap not found: %s", button->fPixmap);
      return 0;
   }
   fPictures->Add((TObject*)pic);

   TGPictureButton *pbut;
   TGLayoutHints   *layout;

   pbut = new TGPictureButton(this, pic, button->fId);
   pbut->SetStyle(gClient->GetStyle());
   pbut->SetToolTipText(button->fTipText);

   layout = new TGLayoutHints(kLHintsTop | kLHintsLeft, spacing, 0, 2, 2);
   AddFrame(pbut, layout);
   pbut->AllowStayDown(button->fStayDown);
   pbut->Associate(w);
   button->fButton = pbut;

   fTrash->Add(pbut);
   fTrash->Add(layout);

   fMapOfButtons->Add(pbut, (TObject*)((Longptr_t)button->fId));

   Connect(pbut, "Pressed()" , "TGToolBar", this, "ButtonPressed()");
   Connect(pbut, "Released()", "TGToolBar", this, "ButtonReleased()");
   Connect(pbut, "Clicked()" , "TGToolBar", this, "ButtonClicked()");

   return pbut;
}

////////////////////////////////////////////////////////////////////////////////
/// Add button to toolbar. All buttons added via this method will be deleted
/// by the toolbar, w is the window to which the button messages will be send.

TGButton *TGToolBar::AddButton(const TGWindow *w, TGPictureButton *pbut, Int_t spacing)
{
   const TGPicture *pic = pbut->GetPicture();
   fPictures->Add((TObject*)pic);

   TGLayoutHints   *layout;
   layout = new TGLayoutHints(kLHintsTop | kLHintsLeft, spacing, 0, 2, 2);
   pbut->SetStyle(gClient->GetStyle());
   AddFrame(pbut, layout);
   pbut->Associate(w);

   fTrash->Add(pbut);
   fTrash->Add(layout);

   fMapOfButtons->Add(pbut, (TObject*)((Longptr_t)pbut->WidgetId()));

   Connect(pbut, "Pressed()" , "TGToolBar", this, "ButtonPressed()");
   Connect(pbut, "Released()", "TGToolBar", this, "ButtonReleased()");
   Connect(pbut, "Clicked()" , "TGToolBar", this, "ButtonClicked()");

   return pbut;
}

////////////////////////////////////////////////////////////////////////////////
/// Finds and returns a pointer to the button with the specified
/// identifier id. Returns null if the button was not found.

TGButton *TGToolBar::GetButton(Int_t id) const
{
   TIter next(fMapOfButtons);
   TGButton *item = 0;

   while ((item = (TGButton*)next())) {
      if ((Longptr_t)fMapOfButtons->GetValue(item) == id) break;   // found
   }

   return item;
}

////////////////////////////////////////////////////////////////////////////////
/// changes id for button.

void TGToolBar::SetId(TGButton *button, Longptr_t id)
{
   TPair *a = (TPair*) fMapOfButtons->FindObject(button);
   if (a) {
      a->SetValue((TObject*)id);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Finds and returns the id of the button.
/// Returns -1 if the button is not a member of this group.

Longptr_t TGToolBar::GetId(TGButton *button) const
{
   TPair *a = (TPair*) fMapOfButtons->FindObject(button);
   if (a)
      return (Longptr_t)a->Value();
   else
      return (Longptr_t)-1;
}

////////////////////////////////////////////////////////////////////////////////
/// Change the icon of a toolbar button.

void TGToolBar::ChangeIcon(ToolBarData_t *button, const char *new_icon)
{
   const TGPicture *pic = fClient->GetPicture(new_icon);
   if (!pic) {
      Error("ChangeIcon", "pixmap not found: %s", new_icon);
      return;
   }
   fPictures->Add((TObject*)pic);

   ((TGPictureButton *)button->fButton)->SetPicture(pic);
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup and delete all objects contained in this composite frame.
/// This will delete all objects added via AddFrame().
/// CAUTION: all objects (frames and layout hints) must be unique, i.e.
/// cannot be shared.

void TGToolBar::Cleanup()
{
   // avoid double deletion of objects in trash
   delete fTrash;
   fTrash = 0;

   TGCompositeFrame::Cleanup();
}

////////////////////////////////////////////////////////////////////////////////
/// This slot is activated when one of the buttons in the group emits the
/// Pressed() signal.

void TGToolBar::ButtonPressed()
{
   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Longptr_t(a->Value());
      Pressed(id);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This slot is activated when one of the buttons in the group emits the
/// Released() signal.

void TGToolBar::ButtonReleased()
{
   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Longptr_t(a->Value());
      Released(id);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This slot is activated when one of the buttons in the group emits the
/// Clicked() signal.

void TGToolBar::ButtonClicked()
{
   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Longptr_t(a->Value());
      Clicked(id);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save an horizontal slider as a C++ statement(s) on output stream out.

void TGToolBar::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   auto extra_args = SaveCtorArgs(out);

   out << "\n   // tool bar\n";

   out << "   TGToolBar *" << GetName() << " = new TGToolBar(" << fParent->GetName() << "," << GetWidth() << ","
       << GetHeight() << extra_args << ");\n";

   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");\n";

   int i = 0;

   TIter next(GetList());

   while (auto f = static_cast<TGFrameElement *>(next())) {
      if (f->fFrame->InheritsFrom(TGPictureButton::Class())) {
         if (!gROOT->ClassSaved(TGPictureButton::Class())) {
            //  declare a structure used for picture buttons
            out << "   ToolBarData_t toolbardata;\n";
         }

         TGPictureButton *pb = (TGPictureButton *)f->fFrame;
         TString picname = gSystem->UnixPathName(pb->GetPicture()->GetName());
         gSystem->ExpandPathName(picname);

         out << "   toolbardata.fPixmap = \"" << picname.ReplaceSpecialCppChars() << "\";\n";
         out << "   toolbardata.fTipText = \""
             << TString(pb->GetToolTip()->GetText()->GetString()).ReplaceSpecialCppChars() << "\";\n";
         if (pb->GetState() == kButtonDown)
            out << "   toolbardata.fStayDown = kTRUE;\n";
          else
            out << "   toolbardata.fStayDown = kFALSE;\n";
         out << "   toolbardata.fId = " << i + 1 << ";\n";
         out << "   toolbardata.fButton = nullptr;\n";
         out << "   " << GetName() << "->AddButton(" << GetParent()->GetName() << ", &toolbardata, "
             << f->fLayout->GetPadLeft() << ");\n";
         if (pb->GetState() == kButtonDown) {
            out << "   toolbardata.fButton->SetState(kButtonDown);\n";
         }
         if (pb->GetState() == kButtonDisabled) {
            out << "   toolbardata.fButton->SetState(kButtonDisabled);\n";
         }
         if (pb->GetState() == kButtonEngaged) {
            out << "   toolbardata.fButton->SetState(kButtonEngaged);\n";
         }
         i++;
      } else {
         f->fFrame->SavePrimitive(out, option);
         out << "   " << GetName() << "->AddFrame(" << f->fFrame->GetName();
         f->fLayout->SavePrimitive(out, option);
         out << ");\n";
      }
   }
}
