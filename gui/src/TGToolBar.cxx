// @(#)root/gui:$Name:  $:$Id: TGToolBar.cxx,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
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


ClassImp(TGToolBar)

//______________________________________________________________________________
TGToolBar::TGToolBar(const TGWindow *p, UInt_t w, UInt_t h,
                     UInt_t options, ULong_t back) :
   TGCompositeFrame(p, w, h, options, back)
{
   // Create toolbar widget.

   fWidgets  = new TList;
   fPictures = new TList;
}

//______________________________________________________________________________
TGToolBar::~TGToolBar()
{
   // Delete toolbar and its buttons and layout hints.

   if (fWidgets) fWidgets->Delete();

   TIter next(fPictures);
   const TGPicture *p;
   while ((p = (const TGPicture *) next()))
      fClient->FreePicture(p);

   delete fWidgets;
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

   fWidgets->Add(pbut);
   fWidgets->Add(layout);
}
