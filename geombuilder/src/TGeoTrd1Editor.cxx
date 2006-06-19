// @(#):$Name:  $:$Id: TGeoTrd1Editor.cxx,v 1.1 2006/06/13 15:27:11 brun Exp $
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
//  TGeoTrd1Editor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoTrd1Editor.h"
#include "TGeoTabManager.h"
#include "TGeoTrd1.h"
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

ClassImp(TGeoTrd1Editor)

enum ETGeoTrd1Wid {
   kTRD1_NAME, kTRD1_X1, kTRD1_X2,  kTRD1_Y, kTRD1_Z,
   kTRD1_APPLY, kTRD1_CANCEL, kTRD1_UNDO
};

//______________________________________________________________________________
TGeoTrd1Editor::TGeoTrd1Editor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for trd1 editor
   fShape   = 0;
   fDxi1 = fDxi2 = fDyi = fDzi = 0.0;
   fNamei = "";
   fIsModified = kFALSE;
   fIsShapeEditable = kFALSE;

   fTabMgr = TGeoTabManager::GetMakeTabManager(gPad, fTab);
      
   // TextEntry for shape name
   MakeTitle("Name");
   fShapeName = new TGTextEntry(this, new TGTextBuffer(50), kTRD1_NAME);
   fShapeName->Resize(135, fShapeName->GetDefaultHeight());
   fShapeName->SetToolTipText("Enter the box name");
   fShapeName->Associate(this);
   AddFrame(fShapeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Trd1 dimensions");
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
  
   // Number entry for dx1
   TGCompositeFrame *f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "DX1"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDx1 = new TGNumberEntry(f1, 0., 5, kTRD1_X1);
   fEDx1->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDx1->GetNumberEntry();
   nef->SetToolTipText("Enter the half-lenth in X1");
   fEDx1->Associate(this);
   f1->AddFrame(fEDx1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dx2
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "DX2"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDx2 = new TGNumberEntry(f1, 0., 5, kTRD1_X2);
   fEDx2->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDx2->GetNumberEntry();
   nef->SetToolTipText("Enter the  half-lenth in X2");
   fEDx2->Associate(this);
   f1->AddFrame(fEDx2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));

   // Number entry for dy
   TGCompositeFrame *f2 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f2->AddFrame(new TGLabel(f2, "DY"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDy = new TGNumberEntry(f2, 0., 5, kTRD1_Y);
   fEDy->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDy->GetNumberEntry();
   nef->SetToolTipText("Enter the half-lenth in Y");
   fEDy->Associate(this);
   f2->AddFrame(fEDy, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dz
   TGCompositeFrame *f3 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f3->AddFrame(new TGLabel(f3, "DZ"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDz = new TGNumberEntry(f3, 0., 5, kTRD1_Z);
   fEDz->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDz->GetNumberEntry();
   nef->SetToolTipText("Enter the  half-lenth in Z");
   fEDz->Associate(this);
   f3->AddFrame(fEDz, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
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

   TClass *cl = TGeoTrd1::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoTrd1Editor::~TGeoTrd1Editor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();   
   TClass *cl = TGeoTrd1::Class();
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
void TGeoTrd1Editor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoTrd1Editor", this, "DoApply()");
   fCancel->Connect("Clicked()", "TGeoTrd1Editor", this, "DoCancel()");
   fUndo->Connect("Clicked()", "TGeoTrd1Editor", this, "DoUndo()");
   fShapeName->Connect("TextChanged(const char *)", "TGeoTrd1Editor", this, "DoModified()");
   fEDx1->Connect("ValueSet(Long_t)", "TGeoTrd1Editor", this, "DoDx1()");
   fEDx2->Connect("ValueSet(Long_t)", "TGeoTrd1Editor", this, "DoDx2()");
   fEDy->Connect("ValueSet(Long_t)", "TGeoTrd1Editor", this, "DoDy()");
   fEDz->Connect("ValueSet(Long_t)", "TGeoTrd1Editor", this, "DoDz()");
   fEDx1->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd1Editor", this, "DoModified()");
   fEDx2->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd1Editor", this, "DoModified()");
   fEDy->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd1Editor", this, "DoModified()");
   fEDz->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd1Editor", this, "DoModified()");
   fInit = kFALSE;
}


//______________________________________________________________________________
void TGeoTrd1Editor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoTrd1::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoTrd1*)fModel;
   fDxi1 = fShape->GetDx1();
   fDxi2 = fShape->GetDx2();
   fDyi = fShape->GetDy();
   fDzi = fShape->GetDz();
   const char *sname = fShape->GetName();
   if (!strcmp(sname, fShape->ClassName())) fShapeName->SetText("-no_name");
   else {
      fShapeName->SetText(sname);
      fNamei = sname;
   }   
   fEDx1->SetNumber(fDxi1);
   fEDx2->SetNumber(fDxi2);
   fEDy->SetNumber(fDyi);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoName()
{
// Perform name change.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoApply()
{
// Slot for applying modifications.
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) {
      fShape->SetName(name);
      Int_t id = gGeoManager->GetListOfShapes()->IndexOf(fShape);
      fTabMgr->UpdateShape(id);
   }   
   Double_t dx1 = fEDx1->GetNumber();
   Double_t dx2 = fEDx2->GetNumber();
   Double_t dy = fEDy->GetNumber(); 
   Double_t dz = fEDz->GetNumber();
   Double_t param[4];
   param[0] = dx1;
   param[1] = dx2;
   param[2] = dy;
   param[3] = dz;
   fShape->SetDimensions(param);
   fShape->ComputeBBox();
   fUndo->SetEnabled();
   fCancel->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
   if (fPad) {
      if (gGeoManager && gGeoManager->GetPainter() && gGeoManager->GetPainter()->IsPaintingShape()) {
         fShape->Draw();
         fPad->GetView()->ShowAxis();
      } else {   
         fPad->Modified();
         fPad->Update();
      }   
   }   
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoCancel()
{
// Slot for cancelling current modifications.
   fShapeName->SetText(fNamei.Data());
   fEDx1->SetNumber(fDxi1);
   fEDx2->SetNumber(fDxi2);
   fEDy->SetNumber(fDyi);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoModified()
{
// Slot for signaling modifications.
   fApply->SetEnabled();
   if (fUndo->GetState()==kButtonDisabled) fCancel->SetEnabled();
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoUndo()
{
// Slot for undoing last operation.
   DoCancel();
   DoApply();
   fCancel->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoTrd1Editor::DoDx1()
{
// Slot for dx1.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoDx2()
{
// Slot for dx2.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoDy()
{
// Slot for dy.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd1Editor::DoDz()
{
// Slot for dz.
   DoModified();
}


