// @(#):$Name:  $:$Id: TGeoMaterialEditor.cxx,v 1.3 2006/06/23 16:00:13 brun Exp $
// Author: M.Gheata 

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoMaterialEditor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoMaterialEditor.h"
#include "TGeoTabManager.h"
#include "TGeoMaterial.h"
#include "TGeoManager.h"
#include "TVirtualGeoPainter.h"
#include "TPad.h"
#include "TView.h"
#include "TGTab.h"
#include "TGComboBox.h"
#include "TGButton.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"

ClassImp(TGeoMaterialEditor)

enum ETGeoMaterialWid {
   kMATERIAL_NAME, kMATERIAL_A, kMATERIAL_Z, kMATERIAL_RHO,
   kMATERIAL_RAD, kMATERIAL_ABS,
   kMATERIAL_APPLY, kMATERIAL_CANCEL, kMATERIAL_UNDO
};

//______________________________________________________________________________
TGeoMaterialEditor::TGeoMaterialEditor(const TGWindow *p, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGeoGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor for material editor.
   fMaterial   = 0;
   fAi = fZi = 0;
   fDensityi = 0.0;
   fNamei = "";
   fIsModified = kFALSE;
   fIsMaterialEditable = kTRUE;

   // TextEntry for material name
   MakeTitle("Name");
   fMaterialName = new TGTextEntry(this, new TGTextBuffer(50), kMATERIAL_NAME);
   fMaterialName->Resize(135, fMaterialName->GetDefaultHeight());
   fMaterialName->SetToolTipText("Enter the material name");
   fMaterialName->Associate(this);
   AddFrame(fMaterialName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Material properties");
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 118, 10, kHorizontalFrame |
                                 kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "A"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatA = new TGNumberEntry(f1, 0., 5, kMATERIAL_A);
   nef = (TGTextEntry*)fMatA->GetNumberEntry();
   nef->SetToolTipText("Enter the atomic mass");
   fMatA->Associate(this);
   f1->AddFrame(fMatA, new TGLayoutHints(kLHintsLeft , 2, 2, 4, 4));
   f1->AddFrame(new TGLabel(f1, "Z"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatZ = new TGNumberEntry(f1, 0., 5, kMATERIAL_Z);
   nef = (TGTextEntry*)fMatZ->GetNumberEntry();
   nef->SetToolTipText("Enter the atomic charge");
   fMatZ->Associate(this);
   f1->AddFrame(fMatZ, new TGLayoutHints(kLHintsLeft , 2, 2, 4, 4));
   f1->Resize(150,30);
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));
   
   
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
   // Number entry for density
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Density"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatDensity = new TGNumberEntry(f1, 0., 5, kMATERIAL_RHO);
   nef = (TGTextEntry*)fMatDensity->GetNumberEntry();
   nef->SetToolTipText("Enter material density");
   fMatDensity->Associate(this);
   f1->AddFrame(fMatDensity, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for radiation length
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "RadLen"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatRadLen = new TGNumberEntry(f1, 0., 5, kMATERIAL_RAD);
   nef = (TGTextEntry*)fMatRadLen->GetNumberEntry();
   nef->SetToolTipText("Computed radiation length");
   fMatRadLen->Associate(this);
   f1->AddFrame(fMatRadLen, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for absorbtion length
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "AbsLen"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatAbsLen = new TGNumberEntry(f1, 0., 5, kMATERIAL_ABS);
   nef = (TGTextEntry*)fMatAbsLen->GetNumberEntry();
   nef->SetToolTipText("Absorbtion length");
   fMatAbsLen->Associate(this);
   f1->AddFrame(fMatAbsLen, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
      
   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));

   // Buttons
   TGCompositeFrame *f23 = new TGCompositeFrame(this, 118, 20, kHorizontalFrame | kSunkenFrame | kDoubleBorder);
   fApply = new TGTextButton(f23, "Apply");
   f23->AddFrame(fApply, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   fApply->Associate(this);
   fCancel = new TGTextButton(f23, "Cancel");
   f23->AddFrame(fCancel, new TGLayoutHints(kLHintsCenterX, 2, 2, 4, 4));
   fCancel->Associate(this);
   fUndo = new TGTextButton(f23, " Undo ");
   f23->AddFrame(fUndo, new TGLayoutHints(kLHintsRight , 2, 2, 4, 4));
   fUndo->Associate(this);
   AddFrame(f23,  new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));  
   fUndo->SetSize(fCancel->GetSize());
   fApply->SetSize(fCancel->GetSize());
}

//______________________________________________________________________________
TGeoMaterialEditor::~TGeoMaterialEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   while ((el = (TGFrameElement *)next())) {
      if (el->fFrame->IsComposite()) 
         TGeoTabManager::Cleanup((TGCompositeFrame*)el->fFrame);
   }
   Cleanup();   
}

//______________________________________________________________________________
void TGeoMaterialEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoMaterialEditor", this, "DoApply()");
   fCancel->Connect("Clicked()", "TGeoMaterialEditor", this, "DoCancel()");
   fUndo->Connect("Clicked()", "TGeoMaterialEditor", this, "DoUndo()");
   fMaterialName->Connect("TextChanged(const char *)", "TGeoMaterialEditor", this, "DoName()");
   fMatA->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoA()");
   fMatZ->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoZ()");
   fMatDensity->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoDensity()");
   fMatRadLen->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoRadAbs()");
   fMatAbsLen->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoRadAbs()");
   fInit = kFALSE;
}

//______________________________________________________________________________
void TGeoMaterialEditor::SetModel(TObject* obj)
{
   // Connect to the selected material.
   if (obj == 0 || !(obj->InheritsFrom(TGeoMaterial::Class()))) {
      SetActive(kFALSE);
      return;                 
   } 
   fMaterial = (TGeoMaterial*)obj;
   fAi = (Int_t)fMaterial->GetA();
   fZi = (Int_t)fMaterial->GetZ();
   fDensityi = fMaterial->GetDensity();
   fNamei = fMaterial->GetName();
   fMaterialName->SetText(fMaterial->GetName());
   fMatA->SetNumber(fAi);
   fMatZ->SetNumber(fZi);
   fMatDensity->SetNumber(fDensityi);
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoName()
{
// Perform name change.
   fUndo->SetEnabled();
   fCancel->SetEnabled(kFALSE);
   fApply->SetEnabled(kTRUE);
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoA()
{
// Slot for atomic mass.
   fMatA->SetNumber(fAi);
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoZ()
{
// Slot for charge.
   fMatZ->SetNumber(fZi);
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoDensity()
{
// Slot for density.
   fMatDensity->SetNumber(fDensityi);
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoRadAbs()
{
// Slot for radiation/absorbtion length. 
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoApply()
{
// Slot for applying modifications.
   const char *name = fMaterialName->GetText();
   fMaterial->SetName(name);
   fMatA->SetNumber(fAi);
   fMatZ->SetNumber(fZi);
   fMatDensity->SetNumber(fDensityi);
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fUndo->SetEnabled();
   fCancel->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoCancel()
{
// Slot for cancelling current modifications.
   fMaterialName->SetText(fNamei.Data());
   fMatA->SetNumber(fAi);
   fMatZ->SetNumber(fZi);
   fMatDensity->SetNumber(fDensityi);
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoModified()
{
// Slot for signaling modifications.
   fApply->SetEnabled();
   if (fUndo->GetState()==kButtonDisabled) fCancel->SetEnabled();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoUndo()
{
// Slot for undoing last operation.
   DoCancel();
   fCancel->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
