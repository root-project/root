// @(#)root/eve:$Id$
// Authors: Alja & Matevz Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTextEditor.h"
#include "TEveText.h"
#include "TEveGValuators.h"

#include "TFTGLManager.h"

#include "TColor.h"
#include "TGLabel.h"
#include "TGColorSelect.h"
#include "TGComboBox.h"
#include "TGTextBuffer.h"
#include "TGTextEntry.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TG3DLine.h"


//______________________________________________________________________________
// GUI editor for TEveText.
//

ClassImp(TEveTextEditor);

//______________________________________________________________________________
TEveTextEditor::TEveTextEditor(const TGWindow *p, Int_t width, Int_t height,
                               UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fSize(0),
   fFile(0),
   fMode(0),
   fExtrude(0),

   fLighting(0),
   fAutoBehave(0)
{
   // Constructor.

   MakeTitle("TEveText");

   // Text entry
   fText = new TGTextEntry(this);
   fText->Resize(135, fText->GetDefaultHeight());
   AddFrame(fText, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   fText->Connect("TextChanged(const char *)", "TEveTextEditor", this, "DoText(const char *)");

   // Face Size combo
   fSize = MakeLabeledCombo("Size:");
   Int_t* fsp = &TFTGLManager::GetFontSizeArray()->front();
   Int_t  nums = TFTGLManager::GetFontSizeArray()->size();
   for(Int_t i= 0; i< nums; i++)
   {
      fSize->AddEntry(Form("%-2d", fsp[i]), fsp[i]);
   }
   fSize->Connect("Selected(Int_t)", "TEveTextEditor", this, "DoFontSize()");

   // Font File combo
   fFile = MakeLabeledCombo("File:");
   TObjArray* farr = TFTGLManager::GetFontFileArray();
   TIter next(farr);
   TObjString* os;
   Int_t cnt = 0;
   while ((os = (TObjString*) next()) != 0)
   {
      fFile->AddEntry(Form("%s", os->GetString().Data()), cnt);
      cnt++;
   }
   fFile->Connect("Selected(Int_t)", "TEveTextEditor", this, "DoFontFile()");

   // Mode combo
   fMode = MakeLabeledCombo("Mode:");
   fMode->AddEntry("Bitmap",  TFTGLManager::kBitmap);
   fMode->AddEntry("Pixmap",  TFTGLManager::kPixmap);
   fMode->AddEntry("Outline", TFTGLManager::kOutline);
   fMode->AddEntry("Polygon", TFTGLManager::kPolygon);
   fMode->AddEntry("Extrude", TFTGLManager::kExtrude);
   fMode->AddEntry("Texture", TFTGLManager::kTexture);
   fMode->Connect("Selected(Int_t)", "TEveTextEditor", this, "DoFontMode()");

   fExtrude = new TEveGValuator(this, "Depth:", 90, 0);
   fExtrude->SetLabelWidth(45);
   fExtrude->SetNELength(5);
   // fExtrude->SetShowSlider(kFALSE);
   fExtrude->Build();
   fExtrude->SetLimits(0.01, 10, 100, TGNumberFormat::kNESRealTwo);
   fExtrude->SetToolTip("Extrusion depth.");
   fExtrude->Connect("ValueSet(Double_t)", "TEveTextEditor", this, "DoExtrude()");
   AddFrame(fExtrude, new TGLayoutHints(kLHintsTop, 4, 1, 1, 1));
   
   // GLConfig
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame | kFitWidth | kFixedWidth );
   f1->AddFrame(new TGLabel(f1, "GLConfig"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 8, 0));

   fAutoBehave  = new TGCheckButton(this, "AutoBehave");
   AddFrame(fAutoBehave, new TGLayoutHints(kLHintsLeft, 1,2,0,0));
   fAutoBehave->Connect("Toggled(Bool_t)", "TEveTextEditor", this, "DoAutoBehave()");

   fLighting  = new TGCheckButton(this, "GL Lighting");
   AddFrame(fLighting, new TGLayoutHints(kLHintsLeft, 1,2,0,0));
   fLighting->Connect("Toggled(Bool_t)", "TEveTextEditor", this, "DoLighting()");
}

//______________________________________________________________________________
TGComboBox* TEveTextEditor::MakeLabeledCombo(const char* name)
{
   // Helper function. Creates TGComboBox with fixed size TGLabel. 

   Bool_t alignRight = kFALSE;
   UInt_t labelW = 45;
   UInt_t labelH = 20;

   TGHorizontalFrame* hf = new TGHorizontalFrame(this);

   // label
   TGLayoutHints *labh, *labfrh;
   if(alignRight) {
      labh   = new TGLayoutHints(kLHintsRight | kLHintsBottom);
      labfrh = new TGLayoutHints(kLHintsRight);
   } else {
      labh   = new TGLayoutHints(kLHintsLeft  | kLHintsBottom);
      labfrh = new TGLayoutHints(kLHintsLeft);
   }
   TGCompositeFrame *labfr = new TGHorizontalFrame(hf, labelW, labelH, kFixedSize);
   TGLabel* label = new TGLabel(labfr, name);
   labfr->AddFrame(label, labh);
   hf->AddFrame(labfr, labfrh);

   // combo
   TGLayoutHints*  clh =  new TGLayoutHints(kLHintsLeft, 0,0,0,0);
   TGComboBox* combo = new TGComboBox(hf);
   combo->Resize(90, 20);
   hf->AddFrame(combo, clh);

   AddFrame(hf, new TGLayoutHints(kLHintsTop, 4, 1, 1, 1));
   return combo;
}

//______________________________________________________________________________
void TEveTextEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveText*>(obj);
   if (strcmp(fM->GetText(), fText->GetText()))
      fText->SetText(fM->GetText());

   fSize->Select(fM->GetSize(), kFALSE);
   fFile->Select(fM->GetFile(), kFALSE);

   // mode
   fMode->Select(fM->GetMode(), kFALSE);

   // lightning
   fAutoBehave->SetState(fM->GetAutoBehave() ? kButtonDown : kButtonUp);
   if (fM->GetAutoBehave()) {
      fLighting->SetDisabledAndSelected(fM->GetLighting() ? kButtonDown : kButtonUp);
   } else {
      fLighting->SetEnabled();
      fLighting->SetState(fM->GetLighting() ? kButtonDown : kButtonUp);
   }

   // extrude
   if (fM->GetMode() == TFTGLManager::kExtrude)
   {
      ShowFrame(fExtrude);
      fExtrude->SetValue(fM->GetExtrude());
   }
   else
   {
      HideFrame(fExtrude);
   }
}

//______________________________________________________________________________
void TEveTextEditor::DoText(const Text_t* /*txt*/)
{
   // Slot for setting text.

   fM->SetText(fText->GetText());
   Update();
}

//______________________________________________________________________________
void TEveTextEditor::DoFontSize()
{
   // Slot for setting FTGL attributes.

   fM->SetFontSize(fSize->GetSelected(), kFALSE);
   Update();
}

//______________________________________________________________________________
void TEveTextEditor::DoFontFile()
{
   // Slot for setting FTGL attributes.

   fM->SetFontFile(fFile->GetSelected());
   Update();
}
//______________________________________________________________________________
void TEveTextEditor::DoFontMode()
{
   // Slot for setting FTGL attributes.

   fM->SetFontMode(fMode->GetSelected());
   Update();
}

//______________________________________________________________________________
void TEveTextEditor::DoExtrude()
{
   // Slot for setting an extrude depth.

   fM->SetExtrude(fExtrude->GetValue());
   Update();
}

//______________________________________________________________________________
void TEveTextEditor::DoAutoBehave()
{
   // Slot for enabling/disabling defaults.

   fM->SetAutoBehave(fAutoBehave->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveTextEditor::DoLighting()
{
    // Slot for enabling/disabling GL lighting.

   fM->SetLighting(fLighting->IsOn());
   Update();
}
