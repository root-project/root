// @(#):$Name:  $:$Id: TGeoBBoxEditor.cxx,v 1.1 2006/06/13 15:27:11 brun Exp $
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
//  TGeoBBoxEditor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoBBoxEditor.h"
#include "TGeoTabManager.h"
#include "TGeoBBox.h"
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

ClassImp(TGeoBBoxEditor)

enum ETGeoBBoxWid {
   kBOX_NAME, kBOX_X, kBOX_Y, kBOX_Z,
   kBOX_OX, kBOX_OY, kBOX_OZ,
   kBOX_APPLY, kBOX_CANCEL, kBOX_UNDO
};

//______________________________________________________________________________
TGeoBBoxEditor::TGeoBBoxEditor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for volume editor
   fShape   = 0;
   fDxi = fDyi = fDzi = 0.0;
   memset(fOrigi, 0, 3*sizeof(Double_t));
   fNamei = "";
   fIsModified = kFALSE;
   fIsShapeEditable = kFALSE;

   fTabMgr = TGeoTabManager::GetMakeTabManager(gPad, fTab);
      
   // TextEntry for shape name
   MakeTitle("Name");
   fShapeName = new TGTextEntry(this, new TGTextBuffer(50), kBOX_NAME);
   fShapeName->Resize(135, fShapeName->GetDefaultHeight());
   fShapeName->SetToolTipText("Enter the box name");
   fShapeName->Associate(this);
   AddFrame(fShapeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Box half-lengths");
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
   // Number entry for dx
   TGCompositeFrame *f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "DX"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fBoxDx = new TGNumberEntry(f1, 0., 5, kBOX_X);
   fBoxDx->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fBoxDx->GetNumberEntry();
   nef->SetToolTipText("Enter the box half-lenth in X");
   fBoxDx->Associate(this);
   f1->AddFrame(fBoxDx, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dy
   TGCompositeFrame *f2 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f2->AddFrame(new TGLabel(f2, "DY"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fBoxDy = new TGNumberEntry(f2, 0., 5, kBOX_Y);
   fBoxDy->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fBoxDy->GetNumberEntry();
   nef->SetToolTipText("Enter the box half-lenth in Y");
   fBoxDy->Associate(this);
   f2->AddFrame(fBoxDy, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dx
   TGCompositeFrame *f3 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f3->AddFrame(new TGLabel(f3, "DZ"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fBoxDz = new TGNumberEntry(f3, 0., 5, kBOX_Z);
   fBoxDz->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fBoxDz->GetNumberEntry();
   nef->SetToolTipText("Enter the box half-lenth in Z");
   fBoxDz->Associate(this);
   f3->AddFrame(fBoxDz, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f3, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));
      
   MakeTitle("Box origin");
   compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
   // Number entry for dx
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "OX"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fBoxOx = new TGNumberEntry(f1, 0., 5, kBOX_OX);
   nef = (TGTextEntry*)fBoxOx->GetNumberEntry();
   nef->SetToolTipText("Enter the box origin X coordinate");
   fBoxOx->Associate(this);
   f1->AddFrame(fBoxOx, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dy
   f2 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f2->AddFrame(new TGLabel(f2, "OY"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fBoxOy = new TGNumberEntry(f2, 0., 5, kBOX_OY);
   nef = (TGTextEntry*)fBoxOy->GetNumberEntry();
   nef->SetToolTipText("Enter the box origin Y coordinate");
   fBoxOy->Associate(this);
   f2->AddFrame(fBoxOy, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dx
   f3 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                             kLHintsExpandX | kFixedWidth | kOwnBackground);
   f3->AddFrame(new TGLabel(f3, "OZ"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fBoxOz = new TGNumberEntry(f3, 0., 5, kBOX_OZ);
   nef = (TGTextEntry*)fBoxOz->GetNumberEntry();
   nef->SetToolTipText("Enter the box origin Z coordinate");
   fBoxOz->Associate(this);
   f3->AddFrame(fBoxOz, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f3, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));

   // Buttons
   TGCompositeFrame *f23 = new TGCompositeFrame(this, 118, 20, kHorizontalFrame | kSunkenFrame | kDoubleBorder);
   fApply = new TGTextButton(f23, "&Apply");
   f23->AddFrame(fApply, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
   fApply->Associate(this);
   fCancel = new TGTextButton(f23, "&Cancel");
   f23->AddFrame(fCancel, new TGLayoutHints(kLHintsCenterX, 2, 2, 4, 4));
   fCancel->Associate(this);
   fUndo = new TGTextButton(f23, " &Undo ");
   f23->AddFrame(fUndo, new TGLayoutHints(kLHintsRight , 2, 2, 4, 4));
   fUndo->Associate(this);
   AddFrame(f23,  new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));  
   fUndo->SetSize(fCancel->GetSize());
   fApply->SetSize(fCancel->GetSize());

   // Initialize layout
   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TGeoBBox::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoBBoxEditor::~TGeoBBoxEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();   
   TClass *cl = TGeoBBox::Class();
   TIter next1(cl->GetEditorList()); 
   TGedElement *ge;
   while ((ge=(TGedElement*)next1())) {
      if (ge->fGedFrame==this) {
         cl->GetEditorList()->Remove(ge);
         delete ge;
         next1.Reset();
      }
   }      
}

//______________________________________________________________________________
void TGeoBBoxEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoBBoxEditor", this, "DoApply()");
   fCancel->Connect("Clicked()", "TGeoBBoxEditor", this, "DoCancel()");
   fUndo->Connect("Clicked()", "TGeoBBoxEditor", this, "DoUndo()");
   fShapeName->Connect("TextChanged(const char *)", "TGeoBBoxEditor", this, "DoModified()");
   fBoxDx->Connect("ValueSet(Long_t)", "TGeoBBoxEditor", this, "DoDx()");
   fBoxDy->Connect("ValueSet(Long_t)", "TGeoBBoxEditor", this, "DoDy()");
   fBoxDz->Connect("ValueSet(Long_t)", "TGeoBBoxEditor", this, "DoDz()");
   fBoxDx->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoBBoxEditor", this, "DoModified()");
   fBoxDy->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoBBoxEditor", this, "DoModified()");
   fBoxDz->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoBBoxEditor", this, "DoModified()");
   fBoxOx->Connect("ValueSet(Long_t)", "TGeoBBoxEditor", this, "DoModified()");
   fBoxOy->Connect("ValueSet(Long_t)", "TGeoBBoxEditor", this, "DoModified()");
   fBoxOz->Connect("ValueSet(Long_t)", "TGeoBBoxEditor", this, "DoModified()");
   fInit = kFALSE;
}


//______________________________________________________________________________
void TGeoBBoxEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Update editor for a new selected box.
   if (obj == 0 || (obj->IsA()!=TGeoBBox::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoBBox*)fModel;
   fDxi = fShape->GetDX();
   fDyi = fShape->GetDY();
   fDzi = fShape->GetDZ();
   memcpy(fOrigi, fShape->GetOrigin(), 3*sizeof(Double_t));
   const char *sname = fShape->GetName();
   if (!strcmp(sname, fShape->ClassName())) fShapeName->SetText("-no_name");
   else {
      fShapeName->SetText(sname);
      fNamei = sname;
   }   
   fBoxDx->SetNumber(fDxi);
   fBoxDy->SetNumber(fDyi);
   fBoxDz->SetNumber(fDzi);
   fBoxOx->SetNumber(fOrigi[0]);
   fBoxOy->SetNumber(fOrigi[1]);
   fBoxOz->SetNumber(fOrigi[2]);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoName()
{
   //Slot for name.
   const char *name = fShapeName->GetText();
   if (!strcmp(name, "-no_name") || !strcmp(name, fShape->GetName())) return;
   fShape->SetName(name);
   Int_t id = gGeoManager->GetListOfShapes()->IndexOf(fShape);
   fTabMgr->UpdateShape(id);
   fUndo->SetEnabled();
   fCancel->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}

//______________________________________________________________________________
Bool_t TGeoBBoxEditor::DoBoxParameters()
{
   //Check current box parameters.
   Double_t dx = fBoxDx->GetNumber();
   Double_t dy = fBoxDy->GetNumber();
   Double_t dz = fBoxDz->GetNumber();
   Double_t orig[3];
   orig[0] = fBoxOx->GetNumber();
   orig[1] = fBoxOy->GetNumber();
   orig[2] = fBoxOz->GetNumber();
   Bool_t changed = kFALSE;
   if (dx != fShape->GetDX() || dy != fShape->GetDY() || dz != fShape->GetDZ()) changed = kTRUE;
   if (!changed) {
      if (orig[0] != fShape->GetOrigin()[0] || 
          orig[1] != fShape->GetOrigin()[1] || 
          orig[2] != fShape->GetOrigin()[2]) changed = kTRUE;
   }       
   if (!changed) return kFALSE;
   fUndo->SetEnabled();
   fShape->SetBoxDimensions(dx, dy, dz, orig);
   if (fPad) {
      if (gGeoManager && gGeoManager->GetPainter() && gGeoManager->GetPainter()->IsPaintingShape()) {
         fShape->Draw();
         fPad->GetView()->ShowAxis();
      } else {   
         fPad->Modified();
         fPad->Update();
      }   
   }   
   return kTRUE;
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoApply()
{
   //Slot for applying current parameters.
   DoName();
   if (DoBoxParameters()) {
      fUndo->SetEnabled();
      fCancel->SetEnabled(kFALSE);
      fApply->SetEnabled(kFALSE);
   }   
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoCancel()
{
   // Slot for canceling current parameters.
   if (!fNamei.Length()) fShapeName->SetText("-no_name");
   else fShapeName->SetText(fNamei.Data());
   fBoxDx->SetNumber(fDxi);
   fBoxDy->SetNumber(fDyi);
   fBoxDz->SetNumber(fDzi);
   fBoxOx->SetNumber(fOrigi[0]);
   fBoxOy->SetNumber(fOrigi[1]);
   fBoxOz->SetNumber(fOrigi[2]);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoModified()
{
   //Slot for modifying current parameters.
   fApply->SetEnabled();
   if (fUndo->GetState()==kButtonDisabled) fCancel->SetEnabled();
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoUndo()
{
   // Slot for undoing last operation.
   DoCancel();
   DoBoxParameters();
   fCancel->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoBBoxEditor::DoDx()
{
   //Slot for Dx modification.
   DoModified();
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoDy()
{
   //Slot for Dy modification.
   DoModified();
}

//______________________________________________________________________________
void TGeoBBoxEditor::DoDz()
{
   //Slot for Dz modification.
   DoModified();
}


