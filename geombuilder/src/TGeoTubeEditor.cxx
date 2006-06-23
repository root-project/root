// @(#):$Name:  $:$Id: TGeoTubeEditor.cxx,v 1.3 2006/06/20 06:33:20 brun Exp $
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
//  TGeoTubeEditor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoTubeEditor.h"
#include "TGeoTabManager.h"
#include "TGeoTube.h"
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

ClassImp(TGeoTubeEditor)

enum ETGeoTubeWid {
   kTUBE_NAME, kTUBE_RMIN, kTUBE_RMAX, kTUBE_Z,
   kTUBE_APPLY, kTUBE_UNDO
};

//______________________________________________________________________________
TGeoTubeEditor::TGeoTubeEditor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for tube editor
   fShape   = 0;
   fRmini = fRmaxi = fDzi = 0.0;
   fNamei = "";
   fIsModified = kFALSE;
   fIsShapeEditable = kTRUE;

   fTabMgr = TGeoTabManager::GetMakeTabManager(gPad, fTab);
      
   // TextEntry for shape name
   MakeTitle("Name");
   fShapeName = new TGTextEntry(this, new TGTextBuffer(50), kTUBE_NAME);
   fShapeName->Resize(135, fShapeName->GetDefaultHeight());
   fShapeName->SetToolTipText("Enter the box name");
   fShapeName->Associate(this);
   AddFrame(fShapeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGTextEntry *nef;
   MakeTitle("Tube dimensions");
   TGCompositeFrame *compxyz = new TGCompositeFrame(this, 118, 30, kVerticalFrame | kRaisedFrame);
   // Number entry for rmin
   TGCompositeFrame *f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Rmin"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fERmin = new TGNumberEntry(f1, 0., 5, kTUBE_RMIN);
   fERmin->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fERmin->GetNumberEntry();
   nef->SetToolTipText("Enter the inner radius");
   fERmin->Associate(this);
   fERmin->Resize(100,fERmin->GetDefaultHeight());
   f1->AddFrame(fERmin, new TGLayoutHints(kLHintsRight , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for Rmax
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "Rmax"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fERmax = new TGNumberEntry(f1, 0., 5, kTUBE_RMAX);
   fERmax->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fERmax->GetNumberEntry();
   nef->SetToolTipText("Enter the outer radius");
   fERmax->Associate(this);
   fERmax->Resize(100,fERmax->GetDefaultHeight());
   f1->AddFrame(fERmax, new TGLayoutHints(kLHintsRight , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   // Number entry for dz
   f1 = new TGCompositeFrame(compxyz, 118, 10, kHorizontalFrame |
                                 kLHintsExpandX | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, "DZ"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEDz = new TGNumberEntry(f1, 0., 5, kTUBE_Z);
   fEDz->SetNumAttr(TGNumberFormat::kNEAPositive);
   nef = (TGTextEntry*)fEDz->GetNumberEntry();
   nef->SetToolTipText("Enter the tube half-lenth in Z");
   fEDz->Associate(this);
   fEDz->Resize(100,fEDz->GetDefaultHeight());
   f1->AddFrame(fEDz, new TGLayoutHints(kLHintsRight , 2, 2, 4, 4));
   compxyz->AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   
   compxyz->Resize(150,30);
   AddFrame(compxyz, new TGLayoutHints(kLHintsLeft, 6, 6, 4, 4));
      
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
   
   // Initialize layout
   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TGeoTube::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoTubeEditor::~TGeoTubeEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   while ((el = (TGFrameElement *)next())) {
      if (el->fFrame->IsComposite()) 
         TGeoTabManager::Cleanup((TGCompositeFrame*)el->fFrame);
   }
   Cleanup();   

   TClass *cl = TGeoTube::Class();
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
void TGeoTubeEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoTubeEditor", this, "DoApply()");
   fUndo->Connect("Clicked()", "TGeoTubeEditor", this, "DoUndo()");
   fShapeName->Connect("TextChanged(const char *)", "TGeoTubeEditor", this, "DoModified()");
   fERmin->Connect("ValueSet(Long_t)", "TGeoTubeEditor", this, "DoRmin()");
   fERmax->Connect("ValueSet(Long_t)", "TGeoTubeEditor", this, "DoRmax()");
   fEDz->Connect("ValueSet(Long_t)", "TGeoTubeEditor", this, "DoDz()");
   fERmin->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTubeEditor", this, "DoRmin()");
   fERmax->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTubeEditor", this, "DoRmax()");
   fEDz->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoTubeEditor", this, "DoDz()");
   fInit = kFALSE;
}


//______________________________________________________________________________
void TGeoTubeEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoTube::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoTube*)fModel;
   fRmini = fShape->GetRmin();
   fRmaxi = fShape->GetRmax();
   fDzi = fShape->GetDz();
   fNamei = fShape->GetName();
   fShapeName->SetText(fShape->GetName());
   fERmin->SetNumber(fRmini);
   fERmax->SetNumber(fRmaxi);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
Bool_t TGeoTubeEditor::IsDelayed() const
{
// Check if shape drawing is delayed.
   return (fDelayed->GetState() == kButtonDown);
}

//______________________________________________________________________________
void TGeoTubeEditor::DoName()
{
// Perform name change.
   DoModified();
}

//______________________________________________________________________________
void TGeoTubeEditor::DoApply()
{
// Slot for applying modifications.
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) fShape->SetName(name);
   Double_t rmin = fERmin->GetNumber();
   Double_t rmax = fERmax->GetNumber();
   Double_t dz = fEDz->GetNumber();
   fShape->SetTubeDimensions(rmin, rmax, dz);
   fShape->ComputeBBox();
   fUndo->SetEnabled();
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
void TGeoTubeEditor::DoModified()
{
// Slot for signaling modifications.
   fApply->SetEnabled();
}

//______________________________________________________________________________
void TGeoTubeEditor::DoUndo()
{
// Slot for undoing last operation.
   fERmin->SetNumber(fRmini);
   fERmax->SetNumber(fRmaxi);
   fEDz->SetNumber(fDzi);
   DoApply();
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoTubeEditor::DoRmin()
{
// Slot for rmin.
   Double_t rmin = fERmin->GetNumber();
   Double_t rmax = fERmax->GetNumber();
   if (rmax<rmin+1.e-10) {
      rmin = rmax - 0.1;
      fERmin->SetNumber(rmin);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoTubeEditor::DoRmax()
{
// Slot for rmax.
   Double_t rmin = fERmin->GetNumber();
   Double_t rmax = fERmax->GetNumber();
   if (rmax <= 0.) {
       rmax = 0.1;
       fERmax->SetNumber(rmax);
   }     
   if (rmax<rmin+1.e-10) {
      rmax = rmin + 0.1;
      fERmax->SetNumber(rmax);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoTubeEditor::DoDz()
{
// Slot for dz.
   Double_t dz = fEDz->GetNumber();
   if (dz<=0) {
      dz = 0.1;
      fEDz->SetNumber(dz);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTubeSegEditor                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGeoTubeSegEditor)

enum ETGeoTubeSegWid {
   kTUBESEG_PHI1, kTUBESEG_PHI2, kTUBESEG_PHI
};

//______________________________________________________________________________
TGeoTubeSegEditor::TGeoTubeSegEditor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
                 : TGeoTubeEditor(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for tube segment editor
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
   fEPhi1 = new TGNumberEntry(f1, 0., 5, kTUBESEG_PHI1);
   fEPhi1->Resize(100, fEPhi1->GetDefaultHeight());
   fEPhi1->SetNumAttr(TGNumberFormat::kNEANonNegative);
   nef = (TGTextEntry*)fEPhi1->GetNumberEntry();
   nef->SetToolTipText("Enter the phi1 value");
   fEPhi1->Associate(this);
   f1->AddFrame(fEPhi1, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));

   fEPhi2 = new TGNumberEntry(f1, 0., 5, kTUBESEG_PHI2);
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

   TClass *cl = TGeoTubeSeg::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoTubeSegEditor::~TGeoTubeSegEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   while ((el = (TGFrameElement *)next())) {
      if (el->fFrame->IsComposite()) 
         TGeoTabManager::Cleanup((TGCompositeFrame*)el->fFrame);
   }
   Cleanup();   

   TClass *cl = TGeoTubeSeg::Class();
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
void TGeoTubeSegEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   TGeoTubeEditor::ConnectSignals2Slots();
   Disconnect(fApply, "Clicked()",(TGeoTubeEditor*)this, "DoApply()");
   Disconnect(fUndo, "Clicked()",(TGeoTubeEditor*)this, "DoUndo()");
   fApply->Connect("Clicked()", "TGeoTubeSegEditor", this, "DoApply()");
   fUndo->Connect("Clicked()", "TGeoTubeSegEditor", this, "DoUndo()");
   fEPhi1->Connect("ValueSet(Long_t)", "TGeoTubeSegEditor", this, "DoPhi1()");
   fEPhi2->Connect("ValueSet(Long_t)", "TGeoTubeSegEditor", this, "DoPhi2()");
//   fEPhi1->GetNumberEntry()->Connect("TextChanged(const char *)","TGeoTubeSegEditor", this, "DoPhi1()");
//   fEPhi2->GetNumberEntry()->Connect("TextChanged(const char *)","TGeoTubeSegEditor", this, "DoPhi2()");
   fSPhi->Connect("PositionChanged()","TGeoTubeSegEditor", this, "DoPhi()");
}

//______________________________________________________________________________
void TGeoTubeSegEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Connect to the selected object.
   if (obj == 0 || (obj->IsA()!=TGeoTubeSeg::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fShape = (TGeoTube*)fModel;
   fRmini = fShape->GetRmin();
   fRmaxi = fShape->GetRmax();
   fDzi = fShape->GetDz();
   fNamei = fShape->GetName();
   fPmini = ((TGeoTubeSeg*)fShape)->GetPhi1();
   fPmaxi = ((TGeoTubeSeg*)fShape)->GetPhi2();
   fShapeName->SetText(fShape->GetName());
   fEPhi1->SetNumber(fPmini);
   fEPhi2->SetNumber(fPmaxi);
   fSPhi->SetPosition(fPmini,fPmaxi);
   fERmin->SetNumber(fRmini);
   fERmax->SetNumber(fRmaxi);
   fEDz->SetNumber(fDzi);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoTubeSegEditor::DoPhi1()
{
// Slot for phi1.
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
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoTubeSegEditor::DoPhi2()
{
// Slot for phi2.
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
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoTubeSegEditor::DoPhi()
{
// Slot for phi slider.
   if (!fLock) {
      DoModified();
      fLock = kTRUE;
      fEPhi1->SetNumber(fSPhi->GetMinPosition());
      fLock = kTRUE;
      fEPhi2->SetNumber(fSPhi->GetMaxPosition());
   } else fLock = kFALSE;   
   if (!IsDelayed()) DoApply();
}

//______________________________________________________________________________
void TGeoTubeSegEditor::DoApply()
{
// Slot for applying modifications.
   fApply->SetEnabled(kFALSE);
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) fShape->SetName(name);
   Double_t rmin = fERmin->GetNumber();
   Double_t rmax = fERmax->GetNumber();
   if (rmin<0 || rmax<rmin) return;
   Double_t dz = fEDz->GetNumber();
   Double_t phi1 = fEPhi1->GetNumber();
   Double_t phi2 = fEPhi2->GetNumber();
   if ((phi2-phi1) > 360.001) {
      phi1 = 0.;
      phi2 = 360.;
      fEPhi1->SetNumber(phi1);
      fEPhi2->SetNumber(phi2);
      fLock = kTRUE;
      fSPhi->SetPosition(phi1,phi2);
      fLock = kFALSE;
   }   
   ((TGeoTubeSeg*)fShape)->SetTubsDimensions(rmin, rmax, dz, phi1, phi2);
   fShape->ComputeBBox();
   fUndo->SetEnabled();
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
void TGeoTubeSegEditor::DoUndo()
{
// Slot for undoing last operation.
   fEPhi1->SetNumber(fPmini);
   fEPhi2->SetNumber(fPmaxi);
   fSPhi->SetPosition(fPmini,fPmaxi);
   DoApply();
   fUndo->SetEnabled(kFALSE);
   fApply->SetEnabled(kFALSE);
}



   
