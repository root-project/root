// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveWindowEditor.h"
#include "TEveWindow.h"

#include "TVirtualPad.h"
#include "TColor.h"

// Cleanup these includes:
#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"


//______________________________________________________________________________
// GUI editor for TEveWindow.
//

ClassImp(TEveWindowEditor);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveWindowEditor::TEveWindowEditor(const TGWindow *p, Int_t width, Int_t height,
             UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),
   fShowTitleBar(0)
{
   MakeTitle("TEveWindow");

   fShowTitleBar = new TGCheckButton(this, "Show title-bar");
   AddFrame(fShowTitleBar); // new TGLayoutHints());
   fShowTitleBar->Connect("Clicked()", "TEveWindowEditor", this,
                          "DoShowTitleBar()");
}

/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Set model object.

void TEveWindowEditor::SetModel(TObject* obj)
{
   fM = dynamic_cast<TEveWindow*>(obj);

   fShowTitleBar->SetState(fM->GetShowTitleBar() ? kButtonDown : kButtonUp);
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for ShowTitleBar.

void TEveWindowEditor::DoShowTitleBar()
{
   fM->SetShowTitleBar(fShowTitleBar->IsOn());
   Update();
}
