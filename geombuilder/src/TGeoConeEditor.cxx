// @(#):$Name:  $:$Id: TGeoConeEditor.cxx,v 1.1 2006/06/13 15:27:11 brun Exp $
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
//  TGeoConeEditor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoConeEditor.h"
#include "TGeoTabManager.h"
#include "TGeoCone.h"
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
#include "TGDoubleSlider.h"

ClassImp(TGeoConeEditor)

enum ETGeoConeWid {
   kCONE_NAME, kCONE_RMIN1, kCONE_RMIN2, kCONE_RMAX1, kCONE_RMAX2, kCONE_Z,
   kCONE_APPLY, kCONE_CANCEL, kCONE_UNDO
};

//______________________________________________________________________________
TGeoConeEditor::TGeoConeEditor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for volume editor
   fShape   = 0;
   fRmini1 = fRmaxi1 = fRmini2 = fRmaxi2 = fDzi = 0.0;
   fNamei = "";
   fIsModified = kFALSE;
   fIsShapeEditable = kTRUE;

   fTabMgr = TGeoTabManager::GetMakeTabManager(gPad, fTab);
      
   // TextEntry for shape name
   MakeTitle("Name");
   fShapeName = new TGTextEntry(this, new TGTextBuffer(50), kCONE_NAME);
   fShapeName->Resize(135, fShapeName->GetDefaultHeight());
   fShapeName->SetToolTipText("Enter the cone name");
   fShapeName->Associate(this);
   AddFrame(fShapeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Cone dimensions");
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame);
   
   // Number entry for Rmin1
   TGCompositeFrame *f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Rmin1"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fERmin1 = new TGNumberEntry(f1, 0., 5, kCONE_RMIN1);
   fERmin1->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fERmin1->GetNumberEntry();
   nef->SetToolTipText("Enter the inner radius");
   fERmin1->Associate(this);
   fERmin1->Resize(100, fERmin1->GetDefaultHeight());
   f1->AddFrame(fERmin1, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
  // Number entry for Rmax1
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Rmax1"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fERmax1 = new TGNumberEntry(f1, 0., 5, kCONE_RMAX1);
   fERmax1->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fERmax1->GetNumberEntry();
   nef->SetToolTipText("Enter the outer radius");
   fERmax1->Associate(this);
   fERmax1->Resize(100, fERmax1->GetDefaultHeight());
   f1->AddFrame(fERmax1, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
    
   // Number entry for Rmin2
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Rmin2"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fERmin2 = new TGNumberEntry(f1, 0., 5, kCONE_RMIN2);
   fERmin2->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fERmin2->GetNumberEntry();
   nef->SetToolTipText("Enter the inner radius");
   fERmin2->Associate(this);
   fERmin2->Resize(100, fERmin2->GetDefaultHeight());
   f1->AddFrame(fERmin2, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4)); 
    
   // Number entry for Rmax2
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Rmax2"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fERmax2 = new TGNumberEntry(f1, 0., 5, kCONE_RMAX2);
   fERmax2->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fERmax1->GetNumberEntry();
   nef->SetToolTipText("Enter the outer radius");
   fERmax2->Associate(this);
   fERmax2->Resize(100, fERmax2->GetDefaultHeight());
   f1->AddFrame(fERmax2, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dz
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "DZ"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDz = new TGNumberEntry(f1, 0., 5, kCONE_Z);
   fEDz->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDz->GetNumberEntry();
   nef->SetToolTipText("Enter the cone half-lenth in Z");
   fEDz->Associate(this);
   fEDz->Resize(100, fEDz->GetDefaultHeight());
   f1->AddFrame(fEDz, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));
      
   // Buttons
   TGCompositeFrame *f23 = new TGCompositeFrame(this, 118, 20, kHorizontalFrame | kSunkenFrame);
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

   // Initialize layout
   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TGeoCone::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoConeEditor::~TGeoConeEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();   
   TClass *cl = TGeoCone::Class();
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
void TGeoConeEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoConeEditor", this, "DoApply()");
   fCancel->Connect("Clicked()", "TGeoConeEditor", this, "DoCancel()");
   fUndo->Connect("Clicked()", "TGeoConeEditor", this, "DoUndo()");
   fShapeName->Connect("TextChanged(const char *)", "TGeoConeEditor", this, "DoModified()");
   fERmin1->Connect("ValueSet(Long_t)", "TGeoConeEditor", this, "DoRmin1()");
   fERmin2->Connect("ValueSet(Long_t)", "TGeoConeEditor", this, "DoRmin2()");
   fERmax1->Connect("ValueSet(Long_t)", "TGeoConeEditor", this, "DoRmax1()");
   fERmax2->Connect("ValueSet(Long_t)", "TGeoConeEditor", this, "DoRmax2()");
   fEDz->Connect("ValueSet(Long_t)", "TGeoConeEditor", this, "DoDz()");
   fERmin1->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoConeEditor", this, "DoRmin1()");
   fERmin2->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoConeEditor", this, "DoRmin2()");
   fERmax1->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoConeEditor", this, "DoRmax1()");
   fERmax2->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoConeEditor", this, "DoRmax2()");
   fEDz->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoConeEditor", this, "DoDz()");
   fInit = kFALSE;
}


//______________________________________________________________________________
void TGeoConeEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoCone::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoCone*)fModel;
   fRmini1 = fShape->GetRmin1();
   fRmini2 = fShape->GetRmin2();
   fRmaxi1 = fShape->GetRmax1();
   fRmaxi2 = fShape->GetRmax2();
   fDzi = fShape->GetDz();
   fNamei = fShape->GetName();
   fShapeName->SetText(fShape->GetName());
   fERmin1->SetNumber(fRmini1);
   fERmin2->SetNumber(fRmini2);
   fERmax1->SetNumber(fRmaxi1);
   fERmax2->SetNumber(fRmaxi2);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoConeEditor::DoName()
{
   // Slot for name.
   DoModified();
}

//______________________________________________________________________________
void TGeoConeEditor::DoApply()
{
   //Slot for applying current parameters.
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) {
      fShape->SetName(name);
      Int_t id = gGeoManager->GetListOfShapes()->IndexOf(fShape);
      fTabMgr->UpdateShape(id);
   }   
   Double_t rmin1 = fERmin1->GetNumber();
   Double_t rmin2 = fERmin2->GetNumber();
   Double_t rmax1 = fERmax1->GetNumber();
   Double_t rmax2 = fERmax2->GetNumber();
   Double_t dz = fEDz->GetNumber();
   fShape->SetConeDimensions(dz, rmin1, rmax1, rmin2, rmax2);
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
void TGeoConeEditor::DoCancel()
{
   //Slot for changing current parameters.
   fShapeName->SetText(fNamei.Data());
   fERmin1->SetNumber(fRmini1);
   fERmin2->SetNumber(fRmini2);
   fERmax1->SetNumber(fRmaxi1);
   fERmax2->SetNumber(fRmaxi2);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoConeEditor::DoModified()
{
   //Slot for modifing current parameters.
   fApply->SetEnabled();
   if (fUndo->GetState()==kButtonDisabled) fCancel->SetEnabled();
}

//______________________________________________________________________________
void TGeoConeEditor::DoUndo()
{
   // Slot for undoing current operation.
   DoCancel();
   DoApply();
   fCancel->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoConeEditor::DoRmin1()
{
   // Slot for Rmin1
   Double_t rmin1 = fERmin1->GetNumber();
   Double_t rmax1 = fERmax1->GetNumber();
   if (rmax1<rmin1+1.e-10) {
      rmax1 = rmin1 + 0.1;
      fERmax1->SetNumber(rmax1);
   }   
   DoModified();
}

//______________________________________________________________________________
void TGeoConeEditor::DoRmax1()
{
   // Slot for Rmax1
   Double_t rmin1 = fERmin1->GetNumber();
   Double_t rmax1 = fERmax1->GetNumber();
   if (rmax1<rmin1+1.e-10) {
      rmin1 = rmax1 - 0.1;
      if (rmin1 < 0.) rmin1 = 0.;
      fERmin1->SetNumber(rmin1);
   }   
   DoModified();
}

//______________________________________________________________________________
void TGeoConeEditor::DoRmin2()
{
   // Slot for Rmin2
   Double_t rmin2 = fERmin2->GetNumber();
   Double_t rmax2 = fERmax2->GetNumber();
   if (rmax2<rmin2+1.e-10) {
      rmax2 = rmin2 + 0.1;
      fERmax2->SetNumber(rmax2);
   }   
   DoModified();
}

//______________________________________________________________________________
void TGeoConeEditor::DoRmax2()
{
   // Slot for  Rmax2
   Double_t rmin2 = fERmin2->GetNumber();
   Double_t rmax2 = fERmax2->GetNumber();
   if (rmax2<rmin2+1.e-10) {
      rmin2 = rmax2 - 0.1;
      if (rmin2 < 0.) rmin2 = 0.;
      fERmin2->SetNumber(rmin2);
   }   
   DoModified();
}

//______________________________________________________________________________
void TGeoConeEditor::DoDz()
{
   // Slot for Dz
   DoModified();
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoConeSegEditor                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGeoConeSegEditor)

enum ETGeoConeSegWid {
   kCONESEG_PHI1, kCONESEG_PHI2, kCONESEG_PHI
};

//______________________________________________________________________________
TGeoConeSegEditor::TGeoConeSegEditor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
                 : TGeoConeEditor(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for cone segment editor
   fLock = kFALSE;
   MakeTitle("Phi range");
   TGTextEntry *nef;
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 155, 200, kHorizontalFrame | kRaisedFrame);
   // Vertical slider
   fSPhi = new TGDoubleVSlider(compxyz,140);
   fSPhi->SetRange(0.,720.);
   compxyz->AddFrame(fSPhi, new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 2, 2, 4, 4)); 
   TGCompositeFrame *f1 = new TGCompositeFrame(compxyz, 100, 200, kVerticalFrame);
   f1->AddFrame(new TGLabel(f1, "Phi min."), new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 2, 2));
   fEPhi1 = new TGNumberEntry(f1, 0., 5, kCONESEG_PHI1);
   fEPhi1->Resize(100, fEPhi1->GetDefaultHeight());
   fEPhi1->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fEPhi1->GetNumberEntry();
   nef->SetToolTipText("Enter the phi1 value");
   fEPhi1->Associate(this);
   f1->AddFrame(fEPhi1, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));

   fEPhi2 = new TGNumberEntry(f1, 0., 5, kCONESEG_PHI2);
   fEPhi2->Resize(100, fEPhi2->GetDefaultHeight());
   fEPhi2->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fEPhi2->GetNumberEntry();
   nef->SetToolTipText("Enter the phi2 value");
   fEPhi2->Associate(this);
   f1->AddFrame(fEPhi2, new TGLayoutHints(kLHintsBottom | kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));
   f1->AddFrame(new TGLabel(f1, "Phi max."), new TGLayoutHints(kLHintsBottom | kLHintsLeft, 2, 2, 2, 2));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 2, 2, 2, 2));
   
   compxyz->Resize(150,150);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));
   
   // Initialize layout
   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TGeoConeSeg::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoConeSegEditor::~TGeoConeSegEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();   
   TClass *cl = TGeoConeSeg::Class();
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
void TGeoConeSegEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   TGeoConeEditor::ConnectSignals2Slots();
   Disconnect(fApply, "Clicked()",(TGeoConeEditor*)this, "DoApply()");
   Disconnect(fUndo, "Clicked()",(TGeoConeEditor*)this, "DoUndo()");
   Disconnect(fCancel, "Clicked()",(TGeoConeEditor*)this, "DoCancel()");
   fApply->Connect("Clicked()", "TGeoConeSegEditor", this, "DoApply()");
   fUndo->Connect("Clicked()", "TGeoConeSegEditor", this, "DoUndo()");
   fCancel->Connect("Clicked()", "TGeoConeSegEditor", this, "DoCancel()");
   fEPhi1->Connect("ValueSet(Long_t)", "TGeoConeSegEditor", this, "DoPhi1()");
   fEPhi2->Connect("ValueSet(Long_t)", "TGeoConeSegEditor", this, "DoPhi2()");
//   fEPhi1->GetNumberEntry()->Connect("TextChanged(const char *)","TGeoConeSegEditor", this, "DoPhi1()");
//   fEPhi2->GetNumberEntry()->Connect("TextChanged(const char *)","TGeoConeSegEditor", this, "DoPhi2()");
   fSPhi->Connect("PositionChanged()","TGeoConeSegEditor", this, "DoPhi()");
}

//______________________________________________________________________________
void TGeoConeSegEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoConeSeg::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoCone*)fModel;
   fRmini1 = fShape->GetRmin1();
   fRmaxi1 = fShape->GetRmax1();
   fRmini2 = fShape->GetRmin2();
   fRmaxi2 = fShape->GetRmax2();
   fDzi = fShape->GetDz();
   fNamei = fShape->GetName();
   fPmini = ((TGeoConeSeg*)fShape)->GetPhi1();
   fPmaxi = ((TGeoConeSeg*)fShape)->GetPhi2();
   fShapeName->SetText(fShape->GetName());
   fEPhi1->SetNumber(fPmini);
   fEPhi2->SetNumber(fPmaxi);
   fSPhi->SetPosition(fPmini,fPmaxi);
   fERmin1->SetNumber(fRmini1);
   fERmax1->SetNumber(fRmaxi1);
   fERmin2->SetNumber(fRmini2);
   fERmax2->SetNumber(fRmaxi2);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoConeSegEditor::DoPhi1()
{
   //Slot for Phi1
   Double_t phi1 = fEPhi1->GetNumber();
   Double_t phi2 = fEPhi2->GetNumber();
   if (phi1 > 360-1.e-10) {
      phi1 = 0.;
      fEPhi1->SetNumber(phi1);
   }   
   if (phi2<phi1+1.e-10) {
      phi1 = phi2 - 0.1;
      fEPhi1->SetNumber(phi1);
   }   
   if (!fLock) {
      DoModified();
      fLock = kTRUE;
      fSPhi->SetPosition(phi1,phi2);
   } else fLock = kFALSE;
}

//______________________________________________________________________________
void TGeoConeSegEditor::DoPhi2()
{
   // Slot for Phi2
   Double_t phi1 = fEPhi1->GetNumber();
   Double_t phi2 = fEPhi2->GetNumber();
   if (phi2-phi1 > 360.) {
      phi2 -= 360.;
      fEPhi2->SetNumber(phi2);
   }   
   if (phi2<phi1+1.e-10) {
      phi2 = phi1 + 0.1;
      fEPhi2->SetNumber(phi2);
   }   
   if (!fLock) {
      DoModified();
      fLock = kTRUE;
      fSPhi->SetPosition(phi1,phi2);
   } else fLock = kFALSE;
}

//______________________________________________________________________________
void TGeoConeSegEditor::DoPhi()
{
   // Slot for Phi
   if (!fLock) {
      DoModified();
      fLock = kTRUE;
      fEPhi1->SetNumber(fSPhi->GetMinPosition());
      fLock = kTRUE;
      fEPhi2->SetNumber(fSPhi->GetMaxPosition());
   } else fLock = kFALSE;   
}

//______________________________________________________________________________
void TGeoConeSegEditor::DoApply()
{
   // Slot for applying current parameters.
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) {
      fShape->SetName(name);
      Int_t id = gGeoManager->GetListOfShapes()->IndexOf(fShape);
      fTabMgr->UpdateShape(id);
   }   
   Double_t rmin1 = fERmin1->GetNumber();
   Double_t rmax1 = fERmax1->GetNumber();
   Double_t rmin2 = fERmin2->GetNumber();
   Double_t rmax2 = fERmax2->GetNumber();
   Double_t dz = fEDz->GetNumber();
   Double_t phi1 = fEPhi1->GetNumber();
   Double_t phi2 = fEPhi2->GetNumber();
   if ((phi2-phi1) > 360.) {
      phi1 = 0.;
      phi2 = 360.;
      fEPhi1->SetNumber(phi1);
      fEPhi2->SetNumber(phi2);
      fLock = kTRUE;
      fSPhi->SetPosition(phi1,phi2);
      fLock = kFALSE;
   }   
   ((TGeoConeSeg*)fShape)->SetConsDimensions(dz, rmin1, rmax1, rmin2,rmax2, phi1, phi2);
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
void TGeoConeSegEditor::DoUndo()
{
   // Slot for undoing last operation.
   DoCancel();
   DoApply();
   fCancel->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGeoConeSegEditor::DoCancel()
{
   // Slot for cancel last operation.
   fEPhi1->SetNumber(fPmini);
   fEPhi2->SetNumber(fPmaxi);
   fSPhi->SetPosition(fPmini,fPmaxi);
   TGeoConeEditor::DoCancel();
}


   
