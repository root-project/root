// @(#)root/gui:$Name:  $:$Id: TGToolBar.cxx,v 1.7 2003/11/05 13:08:26 rdm Exp $
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
#include "TGPicture.h"
#include "TGToolTip.h"
#include "TSystem.h"
#include "TROOT.h"
#include "Riostream.h"

ClassImp(TGToolBar)

//______________________________________________________________________________
TGToolBar::TGToolBar(const TGWindow *p, UInt_t w, UInt_t h,
                     UInt_t options, ULong_t back) :
   TGCompositeFrame(p, w, h, options, back)
{
   // Create toolbar widget.

   fPictures = new TList;
   fTrash    = new TList;
}

//______________________________________________________________________________
TGToolBar::~TGToolBar()
{
   // Delete toolbar and its buttons and layout hints.

   if (fTrash) fTrash->Delete();
   delete fTrash;

   TIter next(fPictures);
   const TGPicture *p;
   while ((p = (const TGPicture *) next()))
      fClient->FreePicture(p);

   delete fPictures;
}

//______________________________________________________________________________
void TGToolBar::AddButton(const TGWindow *w, ToolBarData_t *button, Int_t spacing)
{
   // Add button to toolbar. All buttons added via this method will be
   // deleted by the toolbar. On return the TGButton field of the
   // ToolBarData_t struct is filled in (iff fPixmap was valid).
   // Window w is the window to which the button messages will be send.

   const TGPicture *pic = fClient->GetPicture(button->fPixmap);
   if (!pic) {
      Error("AddButton", "pixmap not found: %s", button->fPixmap);
      return;
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
void TGToolBar::SavePrimitive(ofstream &out, Option_t *option)
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

   char quote = '"';

   char name[kMAXPATHLEN];
   int i = 0, len = 0;
   const char *picname, *rootname, *pos;

   rootname = gSystem->Getenv("ROOTSYS");
#ifdef R__WIN32
   TString dirname = TString(rootname);
   dirname.ReplaceAll("/","\\");
   rootname = dirname.Data();
#endif
   len = strlen(rootname);

   TGFrameElement *f;
   TIter next(GetList());

   while ((f = (TGFrameElement *) next())) {
      if (f->fFrame->InheritsFrom(TGPictureButton::Class())) {
         if (!gROOT->ClassSaved(TGPictureButton::Class())) {
            //  declare a structure used for pictute buttons
            out << endl << "   ToolBarData_t t;" << endl;
         }

         TGPictureButton *pb = (TGPictureButton *)f->fFrame;
         out << "   t.fPixmap = " << quote;

         // next write the absolute path as $ROOTSYS/path
         picname = pb->GetPicture()->GetName();
         pos = strstr(picname, rootname);
         if (pos) {
            sprintf(name,"$ROOTSYS%s",pos+len);  // if absolute path
            out << name;
         } else {
            out << picname;                      // if no path
         }
         out << quote << ";" << endl;
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
         i++;
      } else {
         f->fFrame->SavePrimitive(out, option);
		       out << "   " << GetName()<<"->AddFrame(" << f->fFrame->GetName();
         f->fLayout->SavePrimitive(out, option);
         out << ");"<< endl;
      }
   }
}
