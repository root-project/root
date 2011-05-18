// @(#):$Id$
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
//  TGeoEltuEditor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/eltu_pic.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/eltu_ed.jpg">
*/
//End_Html

#include "TGeoEltuEditor.h"
#include "TGeoTabManager.h"
#include "TGeoEltu.h"
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

ClassImp(TGeoEltuEditor)

enum ETGeoEltuWid {
   kELTU_NAME, kELTU_A, kELTU_B,  kELTU_DZ,
   kELTU_APPLY, kELTU_UNDO
};

//______________________________________________________________________________
TGeoEltuEditor::TGeoEltuEditor(const TGWindow *p, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGeoGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor for para editor
   fShape   = 0;
   fAi = fBi = fDzi = 0.0;
   fNamei = "";
   fIsModified = kFALSE;
   fIsShapeEditable = kTRUE;

   // TextEntry for shape name
   MakeTitle("Name");
   fShapeName = new TGTextEntry(this, new TGTextBuffer(50), kELTU_NAME);
   fShapeName->Resize(135, fShapeName->GetDefaultHeight());
   fShapeName->SetToolTipText("Enter the elliptical tube name");
   fShapeName->Associate(this);
   AddFrame(fShapeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Dimensions");
   // Number entry for A
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "A"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEA = new TGNumberEntry(f1, 0., 5, kELTU_A);
   fEA->SetNumAttr(TGNumberFormat::kNEAPositive);
   fEA->Resize(100, fEA->GetDefaultHeight());
   nef = (TGTextEntry*)fEA->GetNumberEntry();
   nef->SetToolTipText("Enter the semi-axis of the ellipse along x");
   fEA->Associate(this);
   f1->AddFrame(fEA, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   
   // Number entry for B
   f1 = new TGCompositeFrame(this, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "B"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEB = new TGNumberEntry(f1, 0., 5, kELTU_B);
   fEB->SetNumAttr(TGNumberFormat::kNEAPositive);
   fEB->Resize(100, fEB->GetDefaultHeight());
   nef = (TGTextEntry*)fEB->GetNumberEntry();
   nef->SetToolTipText("Enter the semi-axis of the ellipse along y");
   fEB->Associate(this);
   f1->AddFrame(fEB, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   
   // Number entry for dz
   f1 = new TGCompositeFrame(this, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Dz"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDz = new TGNumberEntry(f1, 0., 5, kELTU_DZ);
   fEDz->SetNumAttr(TGNumberFormat::kNEAPositive);
   fEDz->Resize(100, fEDz->GetDefaultHeight());
   nef = (TGTextEntry*)fEDz->GetNumberEntry();
   nef->SetToolTipText("Enter the half-lenth in Z");
   fEDz->Associate(this);
   f1->AddFrame(fEDz, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
 
   // Delayed draw
   f1 = new TGCompositeFrame(this, 155, 10, kHorizontalFrame | kFixedWidth | kSunkenFrame);
   fDelayed = new TGCheckButton(f1, "Delayed draw");
   f1->AddFrame(fDelayed, new TGLayoutHints(kLHintsLeft , 2, 2, 4, 4));
   AddFrame(f1,  new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));  

   // Buttons
   f1 = new TGCompositeFrame(this, 155, 10, kHorizontalFrame | kFixedWidth);
   fApply = new TGTextButton(f1, "Apply");
   f1->AddFrame(fApply, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   fApply->Associate(this);
   fUndo = new TGTextButton(f1, "Undo");
   f1->AddFrame(fUndo, new TGLayoutHints(kLHintsRight , 2, 2, 4, 4));
   fUndo->Associate(this);
   AddFrame(f1,  new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));  
   fUndo->SetSize(fApply->GetSize());
}

//______________________________________________________________________________
TGeoEltuEditor::~TGeoEltuEditor()
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
void TGeoEltuEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoEltuEditor", this, "DoApply()");
   fUndo->Connect("Clicked()", "TGeoEltuEditor", this, "DoUndo()");
   fShapeName->Connect("TextChanged(const char *)", "TGeoEltuEditor", this, "DoModified()");
   fEA->Connect("ValueSet(Long_t)", "TGeoEltuEditor", this, "DoA()");
   fEB->Connect("ValueSet(Long_t)", "TGeoEltuEditor", this, "DoB()");
   fEDz->Connect("ValueSet(Long_t)", "TGeoEltuEditor", this, "DoDz()");
   fEA->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoEltuEditor", this, "DoModified()");
   fEB->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoEltuEditor", this, "DoModified()");
   fEDz->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoEltuEditor", this, "DoModified()");
   fInit = kFALSE;
}

//______________________________________________________________________________
void TGeoEltuEditor::SetModel(TObject* obj)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoEltu::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fShape = (TGeoEltu*)obj;
   fAi = fShape->GetA();
   fBi = fShape->GetB();
   fDzi = fShape->GetDz();
   const char *sname = fShape->GetName();
   if (!strcmp(sname, fShape->ClassName())) fShapeName->SetText("-no_name");
   else {
      fShapeName->SetText(sname);
      fNamei = sname;
   }   
   fEA->SetNumber(fAi);
   fEB->SetNumber(fBi);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoEltuEditor::DoName()
{
// Slot for name.
   DoModified();
}

//______________________________________________________________________________
Bool_t TGeoEltuEditor::IsDelayed() const
{
// Check if shape drawing is delayed.
   return (fDelayed->GetState() == kButtonDown);
}

//______________________________________________________________________________
void TGeoEltuEditor::DoApply()
{
// Slot for applying current settings.
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) fShape->SetName(name);
   Double_t a = fEA->GetNumber();
   Double_t b = fEB->GetNumber(); 
   Double_t z = fEDz->GetNumber();  
   Double_t param[3];
   param[0] = a;
   param[1] = b;
   param[2] = z;
   fShape->SetDimensions(param);
   fShape->ComputeBBox();
   fUndo->SetEnabled();
   fApply->SetEnabled(kFALSE);
   if (fPad) {
      if (gGeoManager && gGeoManager->GetPainter() && gGeoManager->GetPainter()->IsPaintingShape()) {
         TView *view = fPad->GetView();
         if (!view) {
            fShape->Draw();
            fPad->GetView()->ShowAxis();
         } else {
            view->SetRange(-fShape->GetDX(), -fShape->GetDY(), -fShape->GetDZ(),
                           fShape->GetDX(), fShape->GetDY(), fShape->GetDZ());
            Update();
         }                  
      } else Update();
   }   
}

//______________________________________________________________________________
void TGeoEltuEditor::DoModified()
{
// Slot for notifying modifications.
   fApply->SetEnabled();
}

//______________________________________________________________________________
void TGeoEltuEditor::DoUndo()
{
// Slot for undoing last operation.
   fEA->SetNumber(fAi);
   fEB->SetNumber(fBi);
   fEDz->SetNumber(fDzi);
   DoApply();
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoEltuEditor::DoA()
{
// Slot for A.
   Double_t a = fEA->GetNumber();
   if (a <= 0) {
      a = 0.1;
      fEA->SetNumber(a);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoEltuEditor::DoB()
{
// Slot for B.
   Double_t b = fEB->GetNumber();
   if (b <= 0) {
      b = 0.1;
      fEB->SetNumber(b);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoEltuEditor::DoDz()
{
// Slot for Z.
   Double_t z = fEDz->GetNumber();
   if (z <= 0) {
      z = 0.1;
      fEDz->SetNumber(z);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}

