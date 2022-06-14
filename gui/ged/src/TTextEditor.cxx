/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTextEditor.h"
#include "TText.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"

ClassImp(TTextEditor);


/** \class TTextEditor
\ingroup ged

Editor for changing TText's and TLatex's attributes.

*/


enum ELatexID{
   kText_Text = 0, kText_Xpos, kText_Ypos, kText_Angle, kText_Size
};


////////////////////////////////////////////////////////////////////////////////
/// TTextEditor constructor.

TTextEditor::TTextEditor(const TGWindow *p,
                  Int_t width, Int_t height,
                  UInt_t options, Pixel_t back)
                  : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   fText = 0;

   // start initializing the window components
   MakeTitle("Text String");

   fText = new TGTextEntry(this, new TGTextBuffer(50), kText_Text);
   fText->Resize(135, fText->GetDefaultHeight());
   fText->SetToolTipText("Enter the text string");
   AddFrame(fText, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl1 = new TGLabel(f1,"X Position");
   fXpos = new TGNumberEntry(f1, 4, 2, kText_Xpos, TGNumberEntry::kNESRealTwo,
   TGNumberEntry::kNEAAnyNumber);
   fXpos->Resize(50, 20);
   f1->AddFrame(lbl1, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f1->AddFrame(fXpos, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl2 = new TGLabel(f2,"Y Position");
   fYpos = new TGNumberEntry(f2, 4, 2, kText_Ypos, TGNumberEntry::kNESRealTwo,
   TGNumberEntry::kNEAAnyNumber);
   fYpos->Resize(50, 20);
   f2->AddFrame(lbl2, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f2->AddFrame(fYpos, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl3 = new TGLabel(f3,"Text Angle");
   fAngle = new TGNumberEntry(f3, 4, 2, kText_Angle, TGNumberEntry::kNESInteger,
   TGNumberEntry::kNEANonNegative);
   fAngle->Resize(50, 20);
   f3->AddFrame(lbl3, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f3->AddFrame(fAngle, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f3, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl4 = new TGLabel(f4,"Text Size");
   fSize = new TGNumberEntry(f4, 4, 2, kText_Size, TGNumberEntry::kNESRealTwo,
   TGNumberEntry::kNEANonNegative);
   fSize->Resize(50, 20);
   f4->AddFrame(lbl4, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f4->AddFrame(fSize, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f4, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
}


////////////////////////////////////////////////////////////////////////////////
/// TTextEditor destructor.

TTextEditor::~TTextEditor()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Set model.

void TTextEditor::SetModel(TObject *obj)
{
   fEditedText = (TText*) (obj);

   fAvoidSignal = kTRUE;
   fText->SetText(fEditedText->GetTitle());
   fXpos->SetNumber(fEditedText->GetX());
   fYpos->SetNumber(fEditedText->GetY());
   fAngle->SetNumber(fEditedText->GetTextAngle());
   fSize->SetNumber(fEditedText->GetTextSize());

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}


////////////////////////////////////////////////////////////////////////////////
/// Connect signals to slots.

void TTextEditor::ConnectSignals2Slots()
{
   fText->Connect("TextChanged(const char *)","TTextEditor",this,"DoText(const char *)");
   fXpos->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoXpos()");
   fYpos->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoYpos()");
   fAngle->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoAngle()");
   fSize->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoSize()");

   fInit = kFALSE;  // connect the slots to the signals only once
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for setting the text Angle.

void TTextEditor::DoAngle()
{
   if (fAvoidSignal) return;
   fEditedText->SetTextAngle(fAngle->GetNumber());
   Update();
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for setting the text Size.

void TTextEditor::DoSize()
{
   if (fAvoidSignal) return;
   fEditedText->SetTextSize(fSize->GetNumber());
   Update();
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for setting the text string.

void TTextEditor::DoText(const char *text)
{
   if (fAvoidSignal) return;
   fEditedText->SetTitle(text);
   Update();
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for setting the text X position.

void TTextEditor::DoXpos()
{
   if (fAvoidSignal) return;
   fEditedText->SetX(fXpos->GetNumber());
   Update();
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for setting the text Y position.

void TTextEditor::DoYpos()
{
   if (fAvoidSignal) return;
   fEditedText->SetY(fYpos->GetNumber());
   Update();
}
