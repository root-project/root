// @(#):$Name:  $:$Id: TGeoTrd2Editor.cxx,v 1.1 2006/06/13 15:27:11 brun Exp $
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
//  TGeoTrd2Editor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoTrd2Editor.h"
#include "TGeoTabManager.h"
#include "TGeoTrd2.h"
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

ClassImp(TGeoTrd2Editor)

enum ETGeoTrd2Wid {
   kTRD2_NAME, kTRD2_X1, kTRD2_X2,  kTRD2_Y1, kTRD2_Y2, kTRD2_Z,
   kTRD2_APPLY, kTRD2_CANCEL, kTRD2_UNDO
};

//______________________________________________________________________________
TGeoTrd2Editor::TGeoTrd2Editor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for trd2 editor
   fShape   = 0;
   fDxi1 = fDxi2 = fDyi1 = fDyi2 = fDzi = 0.0;
   fNamei = "";
   fIsModified = kFALSE;
   fIsShapeEditable = kFALSE;

   fTabMgr = TGeoTabManager::GetMakeTabManager(gPad, fTab);
      
   // TextEntry for shape name
   MakeTitle("Name");
   fShapeName = new TGTextEntry(this, new TGTextBuffer(50), kTRD2_NAME);
   fShapeName->Resize(135, fShapeName->GetDefaultHeight());
   fShapeName->SetToolTipText("Enter the box name");
   fShapeName->Associate(this);
   AddFrame(fShapeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Trd2 dimensions");
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame | kDoubleBorder);
  
   // Number entry for dx1
   TGCompositeFrame *f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "DX1"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDx1 = new TGNumberEntry(f1, 0., 5, kTRD2_X1);
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
   fEDx2 = new TGNumberEntry(f1, 0., 5, kTRD2_X2);
   fEDx2->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDx2->GetNumberEntry();
   nef->SetToolTipText("Enter the  half-lenth in X2");
   fEDx2->Associate(this);
   f1->AddFrame(fEDx2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));

   // Number entry for dy1
   TGCompositeFrame *f2 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f2->AddFrame(new TGLabel(f2, "DY1"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDy1 = new TGNumberEntry(f2, 0., 5, kTRD2_Y1);
   fEDy1->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDy1->GetNumberEntry();
   nef->SetToolTipText("Enter the half-lenth in Y1");
   fEDy1->Associate(this);
   f2->AddFrame(fEDy1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dy2
   f2 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f2->AddFrame(new TGLabel(f2, "DY2"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDy2 = new TGNumberEntry(f2, 0., 5, kTRD2_Y2);
   fEDy2->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDy2->GetNumberEntry();
   nef->SetToolTipText("Enter the half-lenth in Y2");
   fEDy2->Associate(this);
   f2->AddFrame(fEDy2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   compxyz->AddFrame(f2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dz
   TGCompositeFrame *f3 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kFixedWidth | kOwnBackground);
   f3->AddFrame(new TGLabel(f3, "DZ"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDz = new TGNumberEntry(f3, 0., 5, kTRD2_Z);
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

   TClass *cl = TGeoTrd2::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoTrd2Editor::~TGeoTrd2Editor()
{
// Destructor.
   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();   
   TClass *cl = TGeoTrd2::Class();
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
void TGeoTrd2Editor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoTrd2Editor", this, "DoApply()");
   fCancel->Connect("Clicked()", "TGeoTrd2Editor", this, "DoCancel()");
   fUndo->Connect("Clicked()", "TGeoTrd2Editor", this, "DoUndo()");
   fShapeName->Connect("TextChanged(const char *)", "TGeoTrd2Editor", this, "DoModified()");
   fEDx1->Connect("ValueSet(Long_t)", "TGeoTrd2Editor", this, "DoDx1()");
   fEDx2->Connect("ValueSet(Long_t)", "TGeoTrd2Editor", this, "DoDx2()");
   fEDy1->Connect("ValueSet(Long_t)", "TGeoTrd2Editor", this, "DoDy1()");
   fEDy2->Connect("ValueSet(Long_t)", "TGeoTrd2Editor", this, "DoDy2()");
   fEDz->Connect("ValueSet(Long_t)", "TGeoTrd2Editor", this, "DoDz()");
   fEDx1->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd2Editor", this, "DoModified()");
   fEDx2->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd2Editor", this, "DoModified()");
   fEDy1->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd2Editor", this, "DoModified()");
   fEDy2->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd2Editor", this, "DoModified()");
   fEDz->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTrd2Editor", this, "DoModified()");
   fInit = kFALSE;
}


//______________________________________________________________________________
void TGeoTrd2Editor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoTrd2::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoTrd2*)fModel;
   fDxi1 = fShape->GetDx1();
   fDxi2 = fShape->GetDx2();
   fDyi1 = fShape->GetDy1();
   fDyi2 = fShape->GetDy2();
   fDzi = fShape->GetDz();
   const char *sname = fShape->GetName();
   if (!strcmp(sname, fShape->ClassName())) fShapeName->SetText("-no_name");
   else {
      fShapeName->SetText(sname);
      fNamei = sname;
   }   
   fEDx1->SetNumber(fDxi1);
   fEDx2->SetNumber(fDxi2);
   fEDy1->SetNumber(fDyi1);
   fEDy2->SetNumber(fDyi2);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoName()
{
// Perform name change.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoApply()
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
   Double_t dy1 = fEDy1->GetNumber(); 
   Double_t dy2 = fEDy2->GetNumber(); 
   Double_t dz = fEDz->GetNumber();
   Double_t param[5];
   param[0] = dx1;
   param[1] = dx2;
   param[2] = dy1;
   param[3] = dy2;
   param[4] = dz;
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
void TGeoTrd2Editor::DoCancel()
{
// Slot for cancelling current modifications.
   fShapeName->SetText(fNamei.Data());
   fEDx1->SetNumber(fDxi1);
   fEDx2->SetNumber(fDxi2);
   fEDy1->SetNumber(fDyi1);
   fEDy2->SetNumber(fDyi2);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoModified()
{
// Slot for signaling modifications.
   fApply->SetEnabled();
   if (fUndo->GetState()==kButtonDisabled) fCancel->SetEnabled();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoUndo()
{
// Slot for undoing last operation.
   DoCancel();
   DoApply();
   fCancel->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoTrd2Editor::DoDx1()
{
// Slot for dx1.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoDx2()
{
// Slot for dx2.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoDy1()
{
// Slot for dy1.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoDy2()
{
// Slot for dy2.
   DoModified();
}

//______________________________________________________________________________
void TGeoTrd2Editor::DoDz()
{
// Slot for dz.
   DoModified();
}


