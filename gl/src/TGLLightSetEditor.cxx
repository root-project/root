// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLLightSetEditor.h"
#include <TGLLightSet.h>

#include <TVirtualPad.h>
#include <TColor.h>

#include <TGLabel.h>
#include <TGButton.h>
#include <TGNumberEntry.h>
#include <TGColorSelect.h>
#include <TGDoubleSlider.h>

//______________________________________________________________________
// TGLLightSetSubEditor
//
// Sub-editor for TGLLightSet.

ClassImp(TGLLightSetSubEditor)

TGLLightSetSubEditor::TGLLightSetSubEditor(const TGWindow *p) :
   TGVerticalFrame(p),
   fM             (0),

   fLightFrame    (0),
   fTopLight      (0),
   fRightLight    (0),
   fBottomLight   (0),
   fLeftLight     (0),
   fFrontLight    (0),
   fSpecularLight (0)
{
   // Constructor.

   fLightFrame = new TGGroupFrame(this, "Light sources:", kLHintsTop | kLHintsCenterX);

   fLightFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(fLightFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));//-

   TGMatrixLayout *ml = new TGMatrixLayout(fLightFrame, 0, 1, 10);
   fLightFrame->SetLayoutManager(ml);

   fTopLight      = MakeLampButton("Top",      TGLLightSet::kLightTop);
   fRightLight    = MakeLampButton("Right",    TGLLightSet::kLightRight);
   fBottomLight   = MakeLampButton("Bottom",   TGLLightSet::kLightBottom);
   fLeftLight     = MakeLampButton("Left",     TGLLightSet::kLightLeft);
   fFrontLight    = MakeLampButton("Front",    TGLLightSet::kLightFront);
   fSpecularLight = MakeLampButton("Specular", TGLLightSet::kLightSpecular);
}

TGButton* TGLLightSetSubEditor::MakeLampButton(const Text_t* name, Int_t wid)
{
   // Create a button for given lamp and set it up.

   TGButton* b = new TGCheckButton(fLightFrame, name,  wid);
   fLightFrame->AddFrame(b);
   b->Connect("Clicked()", "TGLLightSetSubEditor", this, "DoButton()");
   return b;
}

void TGLLightSetSubEditor::SetModel(TGLLightSet* m)
{
   // New model was set, refresh data.

   fM = m;
   UInt_t ls = fM->GetLightState();

   fTopLight   ->SetState((ls & TGLLightSet::kLightTop)    ? kButtonDown : kButtonUp);
   fRightLight ->SetState((ls & TGLLightSet::kLightRight)  ? kButtonDown : kButtonUp);
   fBottomLight->SetState((ls & TGLLightSet::kLightBottom) ? kButtonDown : kButtonUp);
   fLeftLight  ->SetState((ls & TGLLightSet::kLightLeft)   ? kButtonDown : kButtonUp);
   fFrontLight ->SetState((ls & TGLLightSet::kLightFront)  ? kButtonDown : kButtonUp);

   fSpecularLight->SetState(fM->GetUseSpecular() ? kButtonDown : kButtonUp);
}

void TGLLightSetSubEditor::Changed()
{
   // Data in sub-editor has been changed, emit "Changed()" signal.

   Emit("Changed()");
}

//______________________________________________________________________________
void TGLLightSetSubEditor::DoButton()
{
   // Lights radio button was clicked.

   TGButton* b = (TGButton*) gTQSender;
   fM->SetLight(TGLLightSet::ELight(b->WidgetId()), b->IsOn());
   Changed();
}


//______________________________________________________________________
// TGLLightSetEditor
//
// Editor for TGLLightSet.

ClassImp(TGLLightSetEditor)

TGLLightSetEditor::TGLLightSetEditor(const TGWindow *p, Int_t width, Int_t height,
                                     UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM  (0),
   fSE (0)
{
   // Constructor.

   MakeTitle("TGLLightSet");

   fSE = new TGLLightSetSubEditor(this);
   AddFrame(fSE, new TGLayoutHints(kLHintsTop, 2, 0, 2, 2));
   fSE->Connect("Changed()", "TGLLightSetEditor", this, "Update()");
}

TGLLightSetEditor::~TGLLightSetEditor()
{
   // Destructor.
}

/**************************************************************************/

void TGLLightSetEditor::SetModel(TObject* obj)
{
   // SetModel ... forward to sub-editor.

   fM = dynamic_cast<TGLLightSet*>(obj);
   fSE->SetModel(fM);
}
