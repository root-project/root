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
// TGToolBar                                                            //
//                                                                      //
// A toolbar is a composite frame that contains TGPictureButtons.       //
// Often used in combination with a TGHorizontal3DLine.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGToolBar.h"
#include "TList.h"
#include "TGButton.h"
#include "TGPicture.h"
#include "TGToolTip.h"
#include "TSystem.h"
#include "TROOT.h"
#include "Riostream.h"
#include "TMap.h"


ClassImp(TGToolBar)

//______________________________________________________________________________
TGToolBar::TGToolBar(const TGWindow *p, UInt_t w, UInt_t h,
                     UInt_t options, ULong_t back) :
                     TGCompositeFrame(p, w, h, options, back)

{
   // Create toolbar widget.

   fPictures = new TList;
   fTrash    = new TList;
   fMapOfButtons = new TMap();  // map of button/id pairs

   SetWindowName();
}

//______________________________________________________________________________
TGToolBar::~TGToolBar()
{
   // Delete toolbar and its buttons and layout hints.

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

//______________________________________________________________________________
TGButton *TGToolBar::AddButton(const TGWindow *w, ToolBarData_t *button, Int_t spacing)
{
   // Add button to toolbar. All buttons added via this method will be
   // deleted by the toolbar. On return the TGButton field of the
   // ToolBarData_t struct is filled in (if fPixmap was valid).
   // Window w is the window to which the button messages will be send.

   const TGPicture *pic = fClient->GetPicture(button->fPixmap);
   if (!pic) {
      Error("AddButton", "pixmap not found: %s", button->fPixmap);
      return 0;
   }
   fPictures->Add((TObject*)pic);

   TGPictureButton *pbut;
   TGLayoutHints   *layout;

   pbut = new TGPictureButton(this, pic, button->fId);
   pbut->SetToolTipText(button->fTipText);

   layout = new TGLayoutHints(kLHintsTop | kLHintsLeft, spacing, 0, 2, 2);
   AddFrame(pbut, layout);
   pbut->AllowStayDown(button->fStayDown);
   pbut->Associate(w);
   button->fButton = pbut;

   fTrash->Add(pbut);
   fTrash->Add(layout);

   fMapOfButtons->Add(pbut, (TObject*)((Long_t)button->fId));

   Connect(pbut, "Pressed()" , "TGToolBar", this, "ButtonPressed()");
   Connect(pbut, "Released()", "TGToolBar", this, "ButtonReleased()");
   Connect(pbut, "Clicked()" , "TGToolBar", this, "ButtonClicked()");

   return pbut;
}

//______________________________________________________________________________
TGButton *TGToolBar::AddButton(const TGWindow *w, TGPictureButton *pbut, Int_t spacing)
{
   // Add button to toolbar. All buttons added via this method will be deleted
   // by the toolbar, w is the window to which the button messages will be send.

   const TGPicture *pic = pbut->GetPicture();
   fPictures->Add((TObject*)pic);

   TGLayoutHints   *layout;
   layout = new TGLayoutHints(kLHintsTop | kLHintsLeft, spacing, 0, 2, 2);
   AddFrame(pbut, layout);
   pbut->Associate(w);

   fTrash->Add(pbut);
   fTrash->Add(layout);

   fMapOfButtons->Add(pbut, (TObject*)((Long_t)pbut->WidgetId()));

   Connect(pbut, "Pressed()" , "TGToolBar", this, "ButtonPressed()");
   Connect(pbut, "Released()", "TGToolBar", this, "ButtonReleased()");
   Connect(pbut, "Clicked()" , "TGToolBar", this, "ButtonClicked()");

   return pbut;
}

//______________________________________________________________________________
TGButton *TGToolBar::GetButton(Int_t id) const
{
   // Finds and returns a pointer to the button with the specified
   // identifier id. Returns null if the button was not found.

   TIter next(fMapOfButtons);
   register TGButton *item = 0;

   while ((item = (TGButton*)next())) {
      if ((Long_t)fMapOfButtons->GetValue(item) == id) break;   // found
   }

   return item;
}

//______________________________________________________________________________
void TGToolBar::SetId(TGButton *button, Long_t id)
{
   // changes id for button.

   TPair *a = (TPair*) fMapOfButtons->FindObject(button);
   if (a) {
      a->SetValue((TObject*)id);
   }
}

//______________________________________________________________________________
Long_t TGToolBar::GetId(TGButton *button) const
{
   // Finds and returns the id of the button.
   // Returns -1 if the button is not a member of this group.

   TPair *a = (TPair*) fMapOfButtons->FindObject(button);
   if (a)
      return Long_t(a->Value());
   else
      return Long_t(-1);
}

//______________________________________________________________________________
void TGToolBar::ChangeIcon(ToolBarData_t *button, const char *new_icon)
{
   // Change the icon of a toolbar button.

   const TGPicture *pic = fClient->GetPicture(new_icon);
   if (!pic) {
      Error("ChangeIcon", "pixmap not found: %s", new_icon);
      return;
   }
   fPictures->Add((TObject*)pic);

   ((TGPictureButton *)button->fButton)->SetPicture(pic);
}

//______________________________________________________________________________
void TGToolBar::Cleanup()
{
   // Cleanup and delete all objects contained in this composite frame.
   // This will delete all objects added via AddFrame().
   // CAUTION: all objects (frames and layout hints) must be unique, i.e.
   // cannot be shared.

   // avoid double deletion of objects in trash
   delete fTrash;
   fTrash = 0;

   TGCompositeFrame::Cleanup();
}

//______________________________________________________________________________
void TGToolBar::ButtonPressed()
{
   // This slot is activated when one of the buttons in the group emits the
   // Pressed() signal.

   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Long_t(a->Value());
      Pressed(id);
   }
}

//______________________________________________________________________________
void TGToolBar::ButtonReleased()
{
   // This slot is activated when one of the buttons in the group emits the
   // Released() signal.

   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Long_t(a->Value());
      Released(id);
   }
}

//______________________________________________________________________________
void TGToolBar::ButtonClicked()
{
   // This slot is activated when one of the buttons in the group emits the
   // Clicked() signal.

   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Long_t(a->Value());
      Clicked(id);
   }
}

//______________________________________________________________________________
void TGToolBar::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save an horizontal slider as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl;
   out << "   // tool bar" << endl;

   out << "   TGToolBar *";
   out << GetName() << " = new TGToolBar(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   char quote = '"';

   int i = 0;
   const char *picname;

   TGFrameElement *f;
   TIter next(GetList());

   while ((f = (TGFrameElement *) next())) {
      if (f->fFrame->InheritsFrom(TGPictureButton::Class())) {
         if (!gROOT->ClassSaved(TGPictureButton::Class())) {
            //  declare a structure used for pictute buttons
            out << endl << "   ToolBarData_t t;" << endl;
         }

         TGPictureButton *pb = (TGPictureButton *)f->fFrame;
         picname = pb->GetPicture()->GetName();

         out << "   t.fPixmap = " << quote
             << gSystem->ExpandPathName(gSystem->UnixPathName(picname))
             << quote << ";" << endl;
         out << "   t.fTipText = " << quote
             << pb->GetToolTip()->GetText()->GetString() << quote << ";" << endl;
         if (pb->GetState() == kButtonDown) {
            out << "   t.fStayDown = kTRUE;" << endl;
         } else {
            out << "   t.fStayDown = kFALSE;" << endl;
         }
         out << "   t.fId = " << i+1 << ";" << endl;
         out << "   t.fButton = 0;" << endl;
         out << "   " << GetName() << "->AddButton(" << GetParent()->GetName()
             << ",&t," << f->fLayout->GetPadLeft() << ");" << endl;
         if (pb->GetState() == kButtonDown) {
            out << "   TGButton *" << pb->GetName() <<  " = t.fButton;" << endl;
            out << "   " << pb->GetName() << "->SetState(kButtonDown);"  << endl;
         }
         if (pb->GetState() == kButtonDisabled) {
            out << "   TGButton *" << pb->GetName() <<  " = t.fButton;" << endl;
            out << "   " << pb->GetName() << "->SetState(kButtonDisabled);" << endl;
         }
         if (pb->GetState() == kButtonEngaged) {
            out << "   TGButton *" << pb->GetName() <<  " = t.fButton;" << endl;
            out << "   " << pb->GetName() << "->SetState(kButtonEngaged);"  << endl;
         }
         i++;
      } else {
         f->fFrame->SavePrimitive(out, option);
         out << "   " << GetName()<<"->AddFrame(" << f->fFrame->GetName();
         f->fLayout->SavePrimitive(out, option);
         out << ");"<< endl;
      }
   }
}
