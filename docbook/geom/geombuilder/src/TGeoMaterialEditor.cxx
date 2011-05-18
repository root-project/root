// @(#):$Id$
// Author: M.Gheata 

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//                                                                      
//  TGeoMaterialEditor, TGeoMixtureEditor
// =======================================
//
//   Editors for materials and mixtures.                                                                      
//______________________________________________________________________________

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
#include "TGListView.h"

ClassImp(TGeoMaterialEditor)

enum ETGeoMaterialWid {
   kMATERIAL_NAME, kMATERIAL_A, kMATERIAL_Z, kMATERIAL_RHO,
   kMATERIAL_RAD, kMATERIAL_ABS, kMATERIAL_STATE, kMATERIAL_TEMP, kMATERIAL_PRES,
   kMATERIAL_APPLY, kMATERIAL_CANCEL, kMATERIAL_UNDO
};

enum ETGeoMaterialStates {
   kMAT_UNDEFINED, kMAT_SOLID, kMAT_LIQUID, kMAT_GAS
};   

enum ETGeoMixtureWid {
   kMIX_ELEM, kMIX_CHK1, kMIX_FRAC, kMIX_CHK2, kMIX_NATOMS, kMIX_ADDELEM
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
   fMaterialName->SetDefaultSize(135, fMaterialName->GetDefaultHeight());
   fMaterialName->SetToolTipText("Enter the material name");
   fMaterialName->Associate(this);
   AddFrame(fMaterialName, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 1, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Material properties");
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 118, 10, kHorizontalFrame |
                                 kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "A"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatA = new TGNumberEntry(f1, 0., 6, kMATERIAL_A, TGNumberFormat::kNESRealThree);
   nef = (TGTextEntry*)fMatA->GetNumberEntry();
   nef->SetToolTipText("Enter the atomic mass");
   fMatA->Associate(this);
   f1->AddFrame(fMatA, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   f1->AddFrame(new TGLabel(f1, "Z"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatZ = new TGNumberEntry(f1, 0., 4, kMATERIAL_Z, TGNumberFormat::kNESInteger);
   nef = (TGTextEntry*)fMatZ->GetNumberEntry();
   nef->SetToolTipText("Enter the atomic charge");
   fMatZ->Associate(this);
   f1->AddFrame(fMatZ, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   f1->Resize(150,30);
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   
   
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
   // Combo box for material state
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "State"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatState = new TGComboBox(f1, kMATERIAL_STATE);
   fMatState->AddEntry("Undefined", TGeoMaterial::kMatStateUndefined);
   fMatState->AddEntry("Solid", TGeoMaterial::kMatStateSolid);
   fMatState->AddEntry("Liquid", TGeoMaterial::kMatStateLiquid);
   fMatState->AddEntry("Gas", TGeoMaterial::kMatStateGas);
   fMatState->Resize(90, fMaterialName->GetDefaultHeight());
   f1->AddFrame(fMatState, new TGLayoutHints(kLHintsRight , 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));

   // Number entry for density
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Density"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatDensity = new TGNumberEntry(f1, 0., 5, kMATERIAL_RHO, TGNumberFormat::kNESRealThree);
   fMatDensity->Resize(90, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fMatDensity->GetNumberEntry();
   nef->SetToolTipText("Enter material density in [g/cm3]");
   fMatDensity->Associate(this);
   f1->AddFrame(fMatDensity, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));
   
   // Number entry for temperature
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Temperature"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatTemperature = new TGNumberEntry(f1, 0., 5, kMATERIAL_TEMP, TGNumberFormat::kNESRealTwo);
   fMatTemperature->Resize(90, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fMatTemperature->GetNumberEntry();
   nef->SetToolTipText("Enter material temperature in [Kelvin]");
   fMatTemperature->Associate(this);
   f1->AddFrame(fMatTemperature, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));

   // Number entry for pressure
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Pressure"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatPressure = new TGNumberEntry(f1, 0., 5, kMATERIAL_PRES, TGNumberFormat::kNESRealThree);
   fMatPressure->Resize(90, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fMatPressure->GetNumberEntry();
   nef->SetToolTipText("Enter material pressure in [bar]");
   fMatPressure->Associate(this);
   f1->AddFrame(fMatPressure, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));

   // Number entry for radiation length
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "RadLen"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatRadLen = new TGNumberEntry(f1, 0., 5, kMATERIAL_RAD);
   fMatRadLen->Resize(90, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fMatRadLen->GetNumberEntry();
   nef->SetToolTipText("Computed radiation length");
   fMatRadLen->Associate(this);
   f1->AddFrame(fMatRadLen, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));
   
   // Number entry for absorbtion length
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "AbsLen"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatAbsLen = new TGNumberEntry(f1, 0., 5, kMATERIAL_ABS);
   fMatAbsLen->Resize(90, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fMatAbsLen->GetNumberEntry();
   nef->SetToolTipText("Absorbtion length");
   fMatAbsLen->Associate(this);
   f1->AddFrame(fMatAbsLen, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));
      
   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 0, 0, 2, 2));

   // Buttons
   f23 = new TGCompositeFrame(this, 118, 20, kHorizontalFrame | kSunkenFrame | kDoubleBorder);
   fApply = new TGTextButton(f23, "Apply");
   f23->AddFrame(fApply, new TGLayoutHints(kLHintsLeft, 2, 2, 1, 1));
   fApply->Associate(this);
   fUndo = new TGTextButton(f23, " Undo ");
   f23->AddFrame(fUndo, new TGLayoutHints(kLHintsRight , 2, 2, 1, 1));
   fUndo->Associate(this);
   AddFrame(f23,  new TGLayoutHints(kLHintsLeft, 0, 0, 4, 4));  
   fUndo->SetSize(fApply->GetSize());
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
   fUndo->Connect("Clicked()", "TGeoMaterialEditor", this, "DoUndo()");
   fMaterialName->Connect("TextChanged(const char *)", "TGeoMaterialEditor", this, "DoName()");
   fMatA->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoA()");
   fMatZ->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoZ()");
   fMatState->Connect("Selected(Int_t)", "TGeoMaterialEditor", this, "DoState(Int_t)");
   fMatDensity->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoDensity()");
   fMatTemperature->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoTemperature()");
   fMatPressure->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoPressure()");
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
   fAi = fMaterial->GetA();
   fZi = (Int_t)fMaterial->GetZ();
   fStatei = (Int_t)fMaterial->GetState();
   fDensityi = fMaterial->GetDensity();
   fTempi = fMaterial->GetTemperature();
   fPresi = fMaterial->GetPressure()/6.2415e+8;
   fNamei = fMaterial->GetName();
   fMaterialName->SetText(fMaterial->GetName());
   fMatA->SetNumber(fAi);
   fMatZ->SetNumber(fZi);
   fMatState->Select(fStatei);
   fMatDensity->SetNumber(fDensityi);
   fMatTemperature->SetNumber(fTempi);
   fMatPressure->SetNumber(fPresi);
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoName()
{
// Perform name change.
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoA()
{
// Slot for atomic mass.
   if (fMaterial->IsMixture()) {
      fMatA->SetNumber(fMaterial->GetA());
      return;
   }   
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoZ()
{
// Slot for charge.
   if (fMaterial->IsMixture()) {
      fMatZ->SetNumber(fMaterial->GetZ());
      return;
   }   
   Int_t z = (Int_t)fMatZ->GetNumber();
   TGeoElementTable *table = gGeoManager->GetElementTable();
   if (z >= table->GetNelements()) {
      z = table->GetNelements()-1;
      fMatZ->SetNumber(z);
   }   
   TGeoElement *elem = table->GetElement(z);
   if (!elem) return;
   Double_t a = elem->A();
   fMatA->SetNumber(a);
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoState(Int_t /*state*/)
{
// Slot for material state.
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoTemperature()
{
// Slot for material temperature.
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoPressure()
{
// Slot for material pressure.
   DoModified();
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoDensity()
{
// Slot for density.
//   fMatDensity->SetNumber(fDensityi);
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
   
   fMaterial->SetA(fMatA->GetNumber());
   fMaterial->SetZ(fMatZ->GetNumber());
   fMaterial->SetDensity(fMatDensity->GetNumber());
   fMaterial->SetTemperature(fMatTemperature->GetNumber());
   fMaterial->SetPressure(6.2415e+8*fMatPressure->GetNumber());
   fMaterial->SetState((TGeoMaterial::EGeoMaterialState)fMatState->GetSelected());
   fMaterial->SetRadLen(fMatRadLen->GetNumber(), fMatAbsLen->GetNumber());
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fUndo->SetEnabled();
   fApply->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoUndo()
{
// Slot for cancelling current modifications.
   fMaterialName->SetText(fNamei.Data());
   fMaterial->SetName(fNamei.Data());
   fMatA->SetNumber(fAi);
   fMaterial->SetA(fAi);
   fMatZ->SetNumber(fZi);
   fMaterial->SetZ(fZi);
   fMatState->Select(fStatei);
   fMaterial->SetState((TGeoMaterial::EGeoMaterialState)fStatei);
   fMatDensity->SetNumber(fDensityi);
   fMaterial->SetDensity(fDensityi);
   fMatTemperature->SetNumber(fTempi);
   fMaterial->SetTemperature(fTempi);
   fMatPressure->SetNumber(fPresi);
   fMaterial->SetPressure(fPresi*6.2415e+8);
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoMaterialEditor::DoModified()
{
// Slot for signaling modifications.
   fApply->SetEnabled();
}

ClassImp(TGeoMixtureEditor)

//______________________________________________________________________________
TGeoMixtureEditor::TGeoMixtureEditor(const TGWindow *p, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGeoMaterialEditor(p, width, height, options | kVerticalFrame, back)
{
   // Constructor for mixture editor.
   fMixture = 0;
   TGCompositeFrame *compxyz=0, *f1=0;
   TGTextEntry *nef;
   MakeTitle("Mixture settings");
   fNelem = new TGLabel(this, "Number of elements: 0");
   AddFrame(fNelem, new TGLayoutHints(kLHintsLeft , 6, 2, 2, 2));
   compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
   // Combo box for selecting elements
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   fMixElem = new TGComboBox(f1, kMIX_ELEM);
   TGeoElementTable *table = gGeoManager->GetElementTable();
   if (table) {
      TGeoElement *element;
      for (Int_t i=0; i<table->GetNelements(); i++) {
         element = table->GetElement(i);
         fMixElem->AddEntry(element->GetTitle(),i);
      }
   }      
   fMixElem->Select(0);
   fMixElem->Resize(90, fMaterialName->GetDefaultHeight());
   f1->AddFrame(fMixElem, new TGLayoutHints(kLHintsLeft , 2, 2, 1, 1));
   TGCompositeFrame *comp1 = new TGCompositeFrame(f1, 118, 30, kVerticalFrame);
   fAelem = new TGLabel(comp1, "A = 0");
   comp1->AddFrame(fAelem, new TGLayoutHints(kLHintsRight , 2, 2, 2, 0));
   fZelem = new TGLabel(comp1, "Z = 0");
   comp1->AddFrame(fZelem, new TGLayoutHints(kLHintsRight , 2, 2, 2, 0));
   f1->AddFrame(comp1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX| kLHintsExpandY , 2, 2, 0, 0));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 0, 0));

   // Fraction by weight
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   fChkFraction = new TGCheckButton(f1, "% weight");
   fChkFraction->SetDown(kTRUE);
   f1->AddFrame(fChkFraction, new TGLayoutHints(kLHintsLeft , 2, 2, 6, 1));
   fNEFraction = new TGNumberEntry(f1, 0., 5, kMIX_FRAC, TGNumberFormat::kNESRealThree);
   fNEFraction->SetFormat(TGNumberFormat::kNESRealThree, TGNumberFormat::kNEANonNegative);
   fNEFraction->Resize(65, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fNEFraction->GetNumberEntry();
   nef->SetToolTipText("Enter fraction by weight of this element");
   fNEFraction->SetNumber(0.);
   fNEFraction->Associate(this);
   f1->AddFrame(fNEFraction, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));
   
   // Fraction by number of atoms
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   fChkNatoms = new TGCheckButton(f1, "N. atoms");
   fChkNatoms->SetDown(kFALSE);
   f1->AddFrame(fChkNatoms, new TGLayoutHints(kLHintsLeft, 2, 2, 6, 1));
   fNENatoms = new TGNumberEntry(f1, 0., 5, kMIX_NATOMS);
   fNENatoms->SetFormat(TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative);
   fNENatoms->Resize(65, fMaterialName->GetDefaultHeight());
   nef = (TGTextEntry*)fNENatoms->GetNumberEntry();
   nef->SetToolTipText("Enter number of atoms for this element");
   fNENatoms->SetNumber(0);
   fNENatoms->Associate(this);
   f1->AddFrame(fNENatoms, new TGLayoutHints(kLHintsRight, 2, 2, 1, 1));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 1, 1));

   // Button for adding the element
   fBAddElem = new TGTextButton(compxyz, "Add component");
   fBAddElem->Associate(this);
   compxyz->AddFrame(fBAddElem, new TGLayoutHints(kLHintsRight , 2, 2, 2, 0));

   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 0, 0, 1, 1));
   
   // List view with all components
   fComps = new TGCompositeFrame(this, 150, 100, kVerticalFrame | kSunkenFrame);
   AddFrame(fComps, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 0, 2, 1, 2));

   TGeoTabManager::MoveFrame(f23, this);
}   
   
//______________________________________________________________________________
void TGeoMixtureEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoMixtureEditor", this, "DoApply1()");
   fUndo->Connect("Clicked()", "TGeoMixtureEditor", this, "DoUndo1()");
   fChkFraction->Connect("Clicked()", "TGeoMixtureEditor", this, "DoChkFraction()");
   fChkNatoms->Connect("Clicked()", "TGeoMixtureEditor", this, "DoChkNatoms()");
   fNEFraction->Connect("ValueSet(Long_t)", "TGeoMixtureEditor", this, "DoFraction()");
   fNENatoms->Connect("ValueSet(Long_t)", "TGeoMixtureEditor", this, "DoNatoms()");
   fMixElem->Connect("Selected(Int_t)", "TGeoMixtureEditor", this, "DoSelectElement(Int_t)");
   fBAddElem->Connect("Clicked()", "TGeoMixtureEditor", this, "DoAddElem()");
   fMaterialName->Connect("TextChanged(const char *)", "TGeoMaterialEditor", this, "DoName()");
   fMatA->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoA()");
   fMatZ->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoZ()");
   fMatState->Connect("Selected(Int_t)", "TGeoMaterialEditor", this, "DoState(Int_t)");
   fMatDensity->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoDensity()");
   fMatTemperature->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoTemperature()");
   fMatPressure->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoPressure()");
   fMatRadLen->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoRadAbs()");
   fMatAbsLen->Connect("ValueSet(Long_t)", "TGeoMaterialEditor", this, "DoRadAbs()");
   fInit = kFALSE;
}
      
//______________________________________________________________________________
void TGeoMixtureEditor::SetModel(TObject* obj)
{
   // Connect to the selected mixture.
   if (obj == 0 || !(obj->InheritsFrom(TGeoMixture::Class()))) {
      SetActive(kFALSE);
      return;                 
   } 
   TGeoMaterialEditor::SetModel(obj);
   fMixture = (TGeoMixture*)fMaterial;
   UpdateElements();
}

//______________________________________________________________________________
void TGeoMixtureEditor::DoChkFraction()
{
// Check button state changed for fraction.
   if (fMixture->GetNelements() && fMixture->GetNmixt()) {
      fChkFraction->SetDown(kFALSE);
      fChkNatoms->SetDown(kTRUE);
      return;
   }   
   Bool_t isDown = fChkFraction->IsDown();
   fChkNatoms->SetDown(!isDown);
}   

//______________________________________________________________________________
void TGeoMixtureEditor::DoChkNatoms()
{
// Check button state changed for natoms.
   if (fMixture->GetNelements() && !fMixture->GetNmixt()) {
      fChkFraction->SetDown(kTRUE);
      fChkNatoms->SetDown(kFALSE);
      return;
   }   
   Bool_t isDown = fChkNatoms->IsDown();
   fChkFraction->SetDown(!isDown);
}   

//______________________________________________________________________________
void TGeoMixtureEditor::DoFraction()
{
// Fraction changed.
   if (fMixture->GetNelements() && fMixture->GetNmixt()) return;
   fChkFraction->SetDown(kTRUE);
   fChkNatoms->SetDown(kFALSE);
}   

//______________________________________________________________________________
void TGeoMixtureEditor::DoNatoms()
{
// Natoms changed.
   if (fMixture->GetNelements() && !fMixture->GetNmixt()) return;
   fChkFraction->SetDown(kFALSE);
   fChkNatoms->SetDown(kTRUE);
}   

//______________________________________________________________________________
void TGeoMixtureEditor::DoSelectElement(Int_t ielem)
{
// Slot for selecting an element.
   TGeoElement *el = gGeoManager->GetElementTable()->GetElement(ielem);
   TString z = TString::Format("Z=%d",el->Z());
   TString a = TString::Format("A=%d",(Int_t)el->A());
   fAelem->SetText(a.Data());
   fZelem->SetText(z.Data());
}   

//______________________________________________________________________________
void TGeoMixtureEditor::DoAddElem()
{
// Slot for adding an element. No undo.
   Bool_t byfraction = fChkFraction->IsDown();   
   Int_t natoms = (Int_t)fNENatoms->GetNumber();
   if (!byfraction && natoms<=0) return;
   Double_t frac = fNEFraction->GetNumber();
   if (byfraction && frac<=0) return;
   TGeoElement *el = gGeoManager->GetElementTable()->GetElement(fMixElem->GetSelected());
   if (!el) return;
   if (byfraction) fMixture->AddElement(el, frac);
   else            fMixture->AddElement(el, natoms);   
   fTabMgr->GetMaterialEditor(fMixture);
}

//______________________________________________________________________________
void TGeoMixtureEditor::DoApply1()
{
// Slot for applying modifications.
   const char *name = fMaterialName->GetText();
   fMaterial->SetName(name);
   
   fMaterial->SetDensity(fMatDensity->GetNumber());
   fMaterial->SetTemperature(fMatTemperature->GetNumber());
   fMaterial->SetPressure(6.2415e+8*fMatPressure->GetNumber());
   fMaterial->SetState((TGeoMaterial::EGeoMaterialState)fMatState->GetSelected());
//   fMaterial->SetRadLen(fMatRadLen->GetNumber(), fMatAbsLen->GetNumber());
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fUndo->SetEnabled();
   fApply->SetEnabled(kFALSE);
}   

//______________________________________________________________________________
void TGeoMixtureEditor::DoUndo1()
{
// Slot for undoing all changes.
   fMaterialName->SetText(fNamei.Data());
   fMaterial->SetName(fNamei.Data());
   fMatState->Select(fStatei);
   fMaterial->SetState((TGeoMaterial::EGeoMaterialState)fStatei);
   fMatDensity->SetNumber(fDensityi);
   fMaterial->SetDensity(fDensityi);
   fMatTemperature->SetNumber(fTempi);
   fMaterial->SetTemperature(fTempi);
   fMatPressure->SetNumber(fPresi);
   fMaterial->SetPressure(fPresi*6.2415e+8);
   fMatRadLen->SetNumber(fMaterial->GetRadLen());
   fMatAbsLen->SetNumber(fMaterial->GetIntLen());
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
}   

//______________________________________________________________________________
void TGeoMixtureEditor::UpdateElements()
{
// Update the list of elements in the TGCanvas.
   fComps->RemoveAll();
   Int_t nelem = fMixture->GetNelements();
   for (Int_t i=0; i<nelem; i++) {
      TString s;
      Bool_t byfrac = (fMixture->GetNmixt())?kFALSE:kTRUE;
      if (byfrac) 
         s.TString::Format("%d-%s-%d: Wmass = %g %%", (Int_t)fMixture->GetZmixt()[i], fMixture->GetElement(i)->GetName(), 
                (Int_t)fMixture->GetAmixt()[i],fMixture->GetWmixt()[i]);
      else
         s.TString::Format("%d-%s-%d: Natoms = %d", (Int_t)fMixture->GetZmixt()[i], fMixture->GetElement(i)->GetName(), 
                (Int_t)fMixture->GetAmixt()[i],fMixture->GetNmixt()[i]);
                     
      TGLabel *label = new TGLabel(fComps, s);
      label->SetTextJustify(kTextLeft | kTextCenterY);
      fComps->AddFrame(label, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 1, 1, 0, 0));
   }   
   fComps->MapSubwindows();
}
