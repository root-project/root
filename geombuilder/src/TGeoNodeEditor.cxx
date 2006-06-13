// @(#):$Name:  $:$Id: Exp $
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
//  TGeoNodeEditor                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoNodeEditor.h"
#include "TGedEditor.h"
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TPad.h"
#include "TGTab.h"
#include "TGComboBox.h"
#include "TGButton.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"
#include "TGeoTabManager.h"

ClassImp(TGeoNodeEditor)

enum ETGeoNodeWid {
   kNODE_NAME, kNODE_ID, kNODE_VOLSEL, kNODE_MVOLSEL,
   kNODE_MATRIX, kNODE_EDIT_VOL, kNODE_EDIT_MATRIX,
   kNODE_APPLY, kNODE_CANCEL, kNODE_UNDO
};

//______________________________________________________________________________
TGeoNodeEditor::TGeoNodeEditor(const TGWindow *p, Int_t id, Int_t width,
                               Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor for volume editor
   
   fNode   = 0;
   fTabMgr = TGeoTabManager::GetMakeTabManager(gPad, fTab);
   fIsEditable = kTRUE;
      
   // TextEntry for medium name
   TGTextEntry *nef;
   MakeTitle("Name");
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 140, 30, kHorizontalFrame | kRaisedFrame);
   fNodeName = new TGTextEntry(f1, new TGTextBuffer(50), kNODE_NAME);
   fNodeName->Resize(100, fNodeName->GetDefaultHeight());
   fNodeName->SetToolTipText("Enter the node name");
   fNodeName->Associate(this);
   f1->AddFrame(fNodeName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   f1->AddFrame(new TGLabel(f1, "ID"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));   
   fNodeNumber = new TGNumberEntry(f1, 0., 1, kNODE_ID);
   nef = (TGTextEntry*)fNodeNumber->GetNumberEntry();
   nef->SetToolTipText("Enter the node copy number");
   fNodeNumber->Associate(this);
   f1->AddFrame(fNodeNumber, new TGLayoutHints(kLHintsLeft | kLHintsExpandX , 2, 2, 4, 4));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 3, 3, 2, 5));


// Combo box for mother volume selection
   MakeTitle("Mother volume");
   f1 = new TGCompositeFrame(this, 140, 30, kHorizontalFrame | kRaisedFrame);
   fMotherVolList = new TGComboBox(f1, kNODE_MVOLSEL);
//   fTabMgr->AddComboVolume(fMotherVolList);
   fMotherVolList->Resize(100, fNodeName->GetDefaultHeight());
   fMotherVolList->Associate(this);
   f1->AddFrame(fMotherVolList, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 5));
   fEditMother = new TGTextButton(f1, "Edit");
   fEditMother->Associate(this);
   f1->AddFrame(fEditMother, new TGLayoutHints(kLHintsRight, 2, 2, 2, 5));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 5));
   
// Combo box for volume selection
   MakeTitle("Volume");
   f1 = new TGCompositeFrame(this, 140, 30, kHorizontalFrame | kRaisedFrame);
   fVolList = new TGComboBox(f1, kNODE_VOLSEL);
//   fTabMgr->AddComboVolume(fVolList);
   fVolList->Resize(100, fNodeName->GetDefaultHeight());
   fVolList->Associate(this);
   f1->AddFrame(fVolList, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   fEditVolume = new TGTextButton(f1, "Edit");
   fEditVolume->Associate(this);
   f1->AddFrame(fEditVolume, new TGLayoutHints(kLHintsRight, 2, 2, 2, 5));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 5));

// Combo box for matrix selection
   MakeTitle("Matrix");
   f1 = new TGCompositeFrame(this, 140, 30, kHorizontalFrame | kRaisedFrame);
   fMatrixList = new TGComboBox(f1, kNODE_MATRIX);
   fTabMgr->AddComboMatrix(fMatrixList);
   fMatrixList->Resize(100, fNodeName->GetDefaultHeight());
   fMatrixList->Associate(this);
   f1->AddFrame(fMatrixList, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   fEditMatrix = new TGTextButton(f1, "Edit");
   fEditMatrix->Associate(this);
   f1->AddFrame(fEditMatrix, new TGLayoutHints(kLHintsRight, 2, 2, 2, 5));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 5));
   
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
   AddFrame(f23,  new TGLayoutHints(kLHintsLeft, 2, 2, 6, 6));  
   fUndo->SetSize(fCancel->GetSize());
   fApply->SetSize(fCancel->GetSize());

   // Initialize layout
   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TGeoNode::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TGeoNodeEditor::~TGeoNodeEditor()
{
// Destructor
   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();   
   TClass *cl = TGeoNode::Class();
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
void TGeoNodeEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fApply->Connect("Clicked()", "TGeoNodeEditor", this, "DoApply()");
   fCancel->Connect("Clicked()", "TGeoNodeEditor", this, "DoCancel()");
   fUndo->Connect("Clicked()", "TGeoNodeEditor", this, "DoUndo()");
   fEditMother->Connect("Clicked()", "TGeoNodeEditor", this, "DoEditMother()");
   fEditVolume->Connect("Clicked()", "TGeoNodeEditor", this, "DoEditVolume()");
   fEditMatrix->Connect("Clicked()", "TGeoNodeEditor", this, "DoEditMatrix()");
   fNodeName->Connect("TextChanged(const char *)", "TGeoNodeEditor", this, "DoNodeName()");
   fInit = kFALSE;
}


//______________________________________________________________________________
void TGeoNodeEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t /*event*/)
{
   // Connect to the picked volume.
   if (obj == 0 || !obj->InheritsFrom(TGeoNode::Class())) {
      SetActive(kFALSE);
      return;                 
   } 
   fModel = obj;
   fPad = pad;
   fNode = (TGeoNode*)fModel;
   const char *sname = fNode->GetName();
   fNodeName->SetText(sname);

   fNodeNumber->SetNumber(fNode->GetNumber());

   TGeoVolume *vol;
   TObjArray *list = gGeoManager->GetListOfVolumes();
   Int_t nobj = list->GetEntriesFast();
   Int_t i, icrt1=0, icrt2=0;
   for (i=0; i<nobj; i++) {
      vol = (TGeoVolume*)list->At(i);
      if (fNode->GetMotherVolume() == vol) icrt1 = i;
      if (fNode->GetVolume() == vol) icrt2 = i;
   }
   fMotherVolList->Select(icrt1);   
   fVolList->Select(icrt2); 
        
   list = gGeoManager->GetListOfMatrices();
   nobj = list->GetEntriesFast();
   TGeoMatrix *matrix;
   icrt1 = 1;
   for (i=0; i<nobj; i++) {
      matrix = (TGeoMatrix*)list->At(i);
      if (fNode->GetMatrix() == matrix) icrt1 = i;
   }   
   fMatrixList->Select(icrt1);   

   fApply->SetEnabled(kFALSE);
   fUndo->SetEnabled(kFALSE);
   fCancel->SetEnabled(kFALSE);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TGeoNodeEditor::DoEditMother()
{
// Edit the mother volume.
   fTabMgr->SetEnabled(TGeoTabManager::kTabVolume);
   fTabMgr->GetVolumeEditor(fNode->GetMotherVolume());
   fTabMgr->SetTab(TGeoTabManager::kTabVolume);
}

//______________________________________________________________________________
void TGeoNodeEditor::DoEditVolume()
{
// Edit selected volume.
   fTabMgr->SetEnabled(TGeoTabManager::kTabVolume);
   fTabMgr->GetVolumeEditor(fNode->GetVolume());
   fTabMgr->SetTab(TGeoTabManager::kTabVolume);
}

//______________________________________________________________________________
void TGeoNodeEditor::DoEditMatrix()
{
// Edit selected material.
//   fTabMgr->SetEnabled(TGeoTabManager::kTabMatrix);
   fTabMgr->GetMatrixEditor(fNode->GetMatrix());
//   fTabMgr->SetTab(TGeoTabManager::kTabMatrix);
}

//______________________________________________________________________________
void TGeoNodeEditor::DoNodeName()
{
// Change node name.
   const char *name = fNodeName->GetText();
   if (!strlen(name) || !strcmp(name, fNode->GetName())) return;
   fNode->SetName(name);
}

//______________________________________________________________________________
void TGeoNodeEditor::DoNodeNumber()
{
// Change node copy number
   
}

//______________________________________________________________________________
void TGeoNodeEditor::DoVolumeSelect()
{
}

//______________________________________________________________________________
void TGeoNodeEditor::DoMotherVolumeSelect()
{
}

//______________________________________________________________________________
void TGeoNodeEditor::DoMatrixSelect()
{
}

//______________________________________________________________________________
void TGeoNodeEditor::DoApply()
{
}

//______________________________________________________________________________
void TGeoNodeEditor::DoCancel()
{
}

//______________________________________________________________________________
void TGeoNodeEditor::DoUndo()
{
}
   
