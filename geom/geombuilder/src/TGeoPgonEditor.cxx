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
//  TGeoPgonEditor                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/pgon_pic.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/pgon_ed.jpg">
*/
//End_Html

#include "TGeoPgonEditor.h"
#include "TGeoTabManager.h"
#include "TGeoPgon.h"
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

ClassImp(TGeoPgonEditor)

enum ETGeoPgonWid {
   kPGON_NEDGES
};

//______________________________________________________________________________
TGeoPgonEditor::TGeoPgonEditor(const TGWindow *p, Int_t width,
                               Int_t height, UInt_t options, Pixel_t back)
   : TGeoPconEditor(p, width, height, options | kVerticalFrame, back)
{
   // Constructor for polycone editor
   fNedgesi = 0;
   CreateEdges();
   TGeoTabManager::MoveFrame(fDFrame, this);
   TGeoTabManager::MoveFrame(fBFrame, this);
   fENedges->Connect("ValueSet(Long_t)", "TGeoPgonEditor", this, "DoNedges()");
   fENedges->GetNumberEntry()->Connect("TextChanged(const char *)", "TGeoPgonEditor", this, "DoModified()");
}

//______________________________________________________________________________
TGeoPgonEditor::~TGeoPgonEditor()
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
void TGeoPgonEditor::SetModel(TObject* obj)
{
   // Connect to a given pcon.
   if (obj == 0 || (obj->IsA()!=TGeoPgon::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fShape = (TGeoPcon*)obj;
   const char *sname = fShape->GetName();
   if (!strcmp(sname, fShape->ClassName())) fShapeName->SetText("-no_name");
   else fShapeName->SetText(sname);

   Int_t nsections = fShape->GetNz();
   fNsecti = nsections;
   fNedgesi = ((TGeoPgon*)fShape)->GetNedges();
   fENz->SetNumber(nsections);
   fENedges->SetNumber(fNedgesi);
   fEPhi1->SetNumber(fShape->GetPhi1());
   fPhi1i = fShape->GetPhi1();
   fEDPhi->SetNumber(fShape->GetDphi());
   fDPhii = fShape->GetDphi();
   CreateSections(nsections);
   UpdateSections();
   
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoPgonEditor::DoApply()
{
// Slot for applying modifications.
   TGeoPgon *shape = (TGeoPgon*)fShape;
   const char *name = fShapeName->GetText();
   if (strcmp(name,fShape->GetName())) fShape->SetName(name);
   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled();
   if (!CheckSections()) return;
   // check if number of sections changed
   Bool_t recreate = kFALSE;
   Int_t nz = fENz->GetIntNumber();
   Int_t nedges = fENedges->GetIntNumber();
   Double_t phi1 = fEPhi1->GetNumber();
   Double_t dphi = fEDPhi->GetNumber();
   if (nz != fShape->GetNz()) recreate = kTRUE;
   TGeoPconSection *sect;
   Int_t isect;
   if (recreate) {
      Double_t *array = new Double_t[3*(nz+1)+1];
      array[0] = phi1;
      array[1] = dphi;
      array[2] = nedges;
      array[3] = nz;
      for (isect=0; isect<nz; isect++) {
         sect = (TGeoPconSection*)fSections->At(isect);
         array[4+3*isect] = sect->GetZ();
         array[5+3*isect] = sect->GetRmin();
         array[6+3*isect] = sect->GetRmax();
      }
      shape->SetDimensions(array);
      delete [] array;   
      if (fPad) {
         if (gGeoManager && gGeoManager->GetPainter() && gGeoManager->GetPainter()->IsPaintingShape()) {
            TView *view = fPad->GetView();
            if (!view) {
               fShape->Draw();
               fPad->GetView()->ShowAxis();
            } else {
               const Double_t *orig = fShape->GetOrigin();
               view->SetRange(orig[0]-fShape->GetDX(), orig[1]-fShape->GetDY(), orig[2]-fShape->GetDZ(),
                              orig[0]+fShape->GetDX(), orig[1]+fShape->GetDY(), orig[2]+fShape->GetDZ());
               Update();
            }                  
         } else Update();
      }
      return;
   }           
   // No need to call SetDimensions   
   if (TMath::Abs(phi1-fShape->GetPhi1())>1.e-6) fShape->Phi1() = phi1;
   if (TMath::Abs(dphi-fShape->GetDphi())>1.e-6) fShape->Dphi() = dphi;
   if (nedges != shape->GetNedges())             shape->SetNedges(nedges);
   for (isect=0; isect<fNsections; isect++) {
      sect = (TGeoPconSection*)fSections->At(isect);
      fShape->Z(isect) = sect->GetZ();
      fShape->Rmin(isect) = sect->GetRmin();
      fShape->Rmax(isect) = sect->GetRmax();
   }   
   shape->ComputeBBox();
   if (fPad) {
      if (gGeoManager && gGeoManager->GetPainter() && gGeoManager->GetPainter()->IsPaintingShape()) {
         TView *view = fPad->GetView();
         if (!view) {
            shape->Draw();
            fPad->GetView()->ShowAxis();
         } else {
            const Double_t *orig = fShape->GetOrigin();
            view->SetRange(orig[0]-fShape->GetDX(), orig[1]-fShape->GetDY(), orig[2]-fShape->GetDZ(),
                           orig[0]+fShape->GetDX(), orig[1]+fShape->GetDY(), orig[2]+fShape->GetDZ());
            Update();
         }                  
      } else Update();
   }   
}

//______________________________________________________________________________
void TGeoPgonEditor::DoUndo()
{
// Slot for undoing last operation.
   fENedges->SetNumber(fNedgesi);
   TGeoPconEditor::DoUndo();
}   

//______________________________________________________________________________
void TGeoPgonEditor::CreateEdges()
{
// Create number entry for Nedges.
   TGTextEntry *nef;
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Nedges"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fENedges = new TGNumberEntry(f1, 0., 5, kPGON_NEDGES);
   fENedges->SetNumAttr(TGNumberFormat::kNEAPositive);
   fENedges->SetNumStyle(TGNumberFormat::kNESInteger);
   fENedges->Resize(100,fENedges->GetDefaultHeight());
   nef = (TGTextEntry*)fENedges->GetNumberEntry();
   nef->SetToolTipText("Enter the  number of edges of the polygon");
   fENedges->Associate(this);
   f1->AddFrame(fENedges, new TGLayoutHints(kLHintsRight, 2, 2, 2, 2));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 4));
}

//______________________________________________________________________________
void TGeoPgonEditor::DoNedges()
{
// Change number of edges.
   Int_t nedges = fENedges->GetIntNumber();
   if (nedges < 3) {
      nedges = 3;
      fENedges->SetNumber(nedges);
   }   
   DoModified();
   if (!IsDelayed()) DoApply();
}   

