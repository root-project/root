// @(#)root/ged:$Name:  $:$Id: TArrowEditor.cxx,v 1.12 2006/09/25 13:35:58 rdm Exp $
// Author: Ilka Antcheva   20/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TArrowEditor                                                        //
//                                                                      //
//  Implements GUI for editing arrow attributes: shape, size, angle.    //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TArrowEditor.gif">
*/
//End_Html


#include "TArrowEditor.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TArrow.h"

ClassImp(TArrowEditor)

enum EArrowWid {
   kARROW_ANG,
   kARROW_OPT,
   kARROW_SIZ
};


//______________________________________________________________________________
TArrowEditor::TArrowEditor(const TGWindow *p, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of arrow GUI.

   fArrow = 0;

   MakeTitle("Arrow");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGLabel *fShapeLabel = new TGLabel(f2, "Shape:");
   f2->AddFrame(fShapeLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fOptionCombo = BuildOptionComboBox(f2, kARROW_OPT);
   fOptionCombo->Resize(80, 20);
   f2->AddFrame(fOptionCombo, new TGLayoutHints(kLHintsLeft, 13, 1, 1, 1));
   fOptionCombo->Associate(this);

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fAngleLabel = new TGLabel(f3, "Angle:");
   f3->AddFrame(fAngleLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fAngleEntry = new TGNumberEntry(f3, 30, 8, kARROW_ANG,
                             TGNumberFormat::kNESInteger,
                             TGNumberFormat::kNEANonNegative,
                             TGNumberFormat::kNELLimitMinMax,0, 180);
   fAngleEntry->GetNumberEntry()->SetToolTipText("Set the arrow opening angle in degrees.");
   f3->AddFrame(fAngleEntry, new TGLayoutHints(kLHintsLeft, 16, 1, 1, 1));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fSizeLabel = new TGLabel(f4, "Size: ");
   f4->AddFrame(fSizeLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fSizeEntry = new TGNumberEntry(f4, 0.03, 8, kARROW_SIZ,
                                  TGNumberFormat::kNESRealTwo,
                                  TGNumberFormat::kNEANonNegative,
                                  TGNumberFormat::kNELLimitMinMax, 0.01, 0.30);
   fSizeEntry->GetNumberEntry()->SetToolTipText("Set the size of arrow.");
   f4->AddFrame(fSizeEntry, new TGLayoutHints(kLHintsLeft, 21, 1, 1, 1));

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));
}

//______________________________________________________________________________
TArrowEditor::~TArrowEditor()
{
   // Destructor of arrow editor.

   TGFrameElement *el;
   TIter next(GetList());

   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________
void TArrowEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fOptionCombo->Connect("Selected(Int_t)", "TArrowEditor", this, "DoOption(Int_t)");
   fAngleEntry->Connect("ValueSet(Long_t)", "TArrowEditor", this, "DoAngle()");
   (fAngleEntry->GetNumberEntry())->Connect("ReturnPressed()", "TArrowEditor", this, "DoAngle()");
   fSizeEntry->Connect("ValueSet(Long_t)", "TArrowEditor", this, "DoSize()");
   (fSizeEntry->GetNumberEntry())->Connect("ReturnPressed()", "TArrowEditor", this, "DoSize()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TArrowEditor::SetModel(TObject* obj)
{
   // Pick up the used arrow attributes.

   fArrow = (TArrow *)obj;
   fAvoidSignal = kTRUE;

   Int_t id = GetShapeEntry(fArrow->GetDrawOption());
   if (id != fOptionCombo->GetSelected())
      fOptionCombo->Select(id);

   Float_t sz = fArrow->GetArrowSize();
   fSizeEntry->SetNumber(sz);

   Int_t deg = (Int_t)fArrow->GetAngle();
   fAngleEntry->SetNumber(deg);

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
void TArrowEditor::DoAngle()
{
   // Slot connected to the arrow opening angle setting.

   if (fAvoidSignal) return;
   fArrow->SetAngle((Float_t)fAngleEntry->GetNumber());
   fArrow->Paint(fArrow->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TArrowEditor::DoOption(Int_t id)
{
   // Slot connected to the arrow shape setting.

   if (fAvoidSignal) return;
   const char* opt=0;
   switch (id) {
      case 1:
         opt = "|>";
         break;
      case 2:
         opt = "<|";
         break;
      case 3:
         opt = ">";
         break;
      case 4:
         opt = "<";
         break;
      case 5:
         opt = "->-";
         break;
      case 6:
         opt = "-<-";
         break;
      case 7:
         opt = "-|>-";
         break;
      case 8:
         opt = "-<|-";
         break;
      case 9:
         opt = "<>";
         break;
      case 10:
         opt = "<|>";
         break;
   }
   fArrow->SetDrawOption(opt);
   fArrow->Paint(fArrow->GetDrawOption());
   Update();
}


//______________________________________________________________________________
void TArrowEditor::DoSize()
{
   // Slot connected to the arrow size.

   if (fAvoidSignal) return;
   fArrow->SetArrowSize(fSizeEntry->GetNumber());
   fArrow->Paint(fArrow->GetDrawOption());
   Update();
}

//______________________________________________________________________________
TGComboBox* TArrowEditor::BuildOptionComboBox(TGFrame* parent, Int_t id)
{
   // Arrow shape combobox.

   TGComboBox *cb = new TGComboBox(parent, id);

   cb->AddEntry(" -------|>",1);
   cb->AddEntry(" <|-------",2);
   cb->AddEntry(" -------->",3);
   cb->AddEntry(" <--------",4);
   cb->AddEntry(" ---->----",5);
   cb->AddEntry(" ----<----",6);
   cb->AddEntry(" ----|>---",7);
   cb->AddEntry(" ---<|----",8);
   cb->AddEntry(" <------>", 9);
   cb->AddEntry(" <|-----|>",10);
   (cb->GetListBox())->Resize((cb->GetListBox())->GetWidth(), 136);
   cb->Select(1);
   return cb;
}

//______________________________________________________________________________
Int_t TArrowEditor::GetShapeEntry(Option_t *option)
{
   // Return shape entry according to the arrow draw option.

   TString opt = option;
   opt.ToLower();
   Int_t id = 0;

   if (opt == "|>")   id = 1;
   if (opt == "<|")   id = 2;
   if (opt == ">")    id = 3;
   if (opt == "<")    id = 4;
   if (opt == "->-")  id = 5;
   if (opt == "-<-")  id = 6;
   if (opt == "-|>-") id = 7;
   if (opt == "-<|-") id = 8;
   if (opt == "<>")   id = 9;
   if (opt == "<|>")  id = 10;
   return id;
}
