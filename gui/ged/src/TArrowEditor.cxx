// @(#)root/ged:$Id$
// Author: Ilka Antcheva   20/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TArrowEditor.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TArrow.h"

ClassImp(TArrowEditor);

enum EArrowWid {
   kARROW_ANG,
   kARROW_OPT,
   kARROW_SIZ
};

/** \class TArrowEditor
    \ingroup ged

Implements user interface for editing of arrow attributes:
shape, size, angle.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor of arrow GUI.

TArrowEditor::TArrowEditor(const TGWindow *p, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   fArrow = 0;

   MakeTitle("Arrow");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f2a = new TGCompositeFrame(f2, 80, 20);
   f2->AddFrame(f2a, new TGLayoutHints(kLHintsTop, 10, 0, 0, 0));

   TGLabel *fShapeLabel = new TGLabel(f2a, "Shape:");
   f2a->AddFrame(fShapeLabel, new TGLayoutHints(kLHintsNormal, 0, 0, 1, 5));

   TGLabel *fAngleLabel = new TGLabel(f2a, "Angle:");
   f2a->AddFrame(fAngleLabel, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));

   TGLabel *fSizeLabel = new TGLabel(f2a, "Size: ");
   f2a->AddFrame(fSizeLabel, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 1));

   TGCompositeFrame *f2b = new TGCompositeFrame(f2, 80, 20, kFixedWidth);
   f2->AddFrame(f2b, new TGLayoutHints(kLHintsTop, 10, 0, 0, 0));

   fOptionCombo = BuildOptionComboBox(f2b, kARROW_OPT);
   fOptionCombo->Resize(80, 20);
   f2b->AddFrame(fOptionCombo, new TGLayoutHints(kLHintsExpandX, 1, 1, 1, 1));
   fOptionCombo->Associate(this);

   fAngleEntry = new TGNumberEntry(f2b, 30, 8, kARROW_ANG,
                             TGNumberFormat::kNESInteger,
                             TGNumberFormat::kNEANonNegative,
                             TGNumberFormat::kNELLimitMinMax,0, 180);
   fAngleEntry->GetNumberEntry()->SetToolTipText("Set the arrow opening angle in degrees.");
   f2b->AddFrame(fAngleEntry, new TGLayoutHints(kLHintsExpandX, 1, 1, 3, 1));

   fSizeEntry = new TGNumberEntry(f2b, 0.03, 8, kARROW_SIZ,
                                  TGNumberFormat::kNESRealTwo,
                                  TGNumberFormat::kNEANonNegative,
                                  TGNumberFormat::kNELLimitMinMax, 0.01, 0.30);
   fSizeEntry->GetNumberEntry()->SetToolTipText("Set the size of arrow.");
   f2b->AddFrame(fSizeEntry, new TGLayoutHints(kLHintsExpandX, 1, 1, 3, 1));

}

////////////////////////////////////////////////////////////////////////////////
/// Destructor of arrow editor.

TArrowEditor::~TArrowEditor()
{
   TGFrameElement *el;
   TIter next(GetList());

   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

////////////////////////////////////////////////////////////////////////////////
/// Connect signals to slots.

void TArrowEditor::ConnectSignals2Slots()
{
   fOptionCombo->Connect("Selected(Int_t)", "TArrowEditor", this, "DoOption(Int_t)");
   fAngleEntry->Connect("ValueSet(Long_t)", "TArrowEditor", this, "DoAngle()");
   (fAngleEntry->GetNumberEntry())->Connect("ReturnPressed()", "TArrowEditor", this, "DoAngle()");
   fSizeEntry->Connect("ValueSet(Long_t)", "TArrowEditor", this, "DoSize()");
   (fSizeEntry->GetNumberEntry())->Connect("ReturnPressed()", "TArrowEditor", this, "DoSize()");

   fInit = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Pick up the used arrow attributes.

void TArrowEditor::SetModel(TObject* obj)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the arrow opening angle setting.

void TArrowEditor::DoAngle()
{
   if (fAvoidSignal) return;
   fArrow->SetAngle((Float_t)fAngleEntry->GetNumber());
   fArrow->Paint(fArrow->GetDrawOption());
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the arrow shape setting.

void TArrowEditor::DoOption(Int_t id)
{
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


////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the arrow size.

void TArrowEditor::DoSize()
{
   if (fAvoidSignal) return;
   fArrow->SetArrowSize(fSizeEntry->GetNumber());
   fArrow->Paint(fArrow->GetDrawOption());
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Arrow shape combobox.

TGComboBox* TArrowEditor::BuildOptionComboBox(TGFrame* parent, Int_t id)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Return shape entry according to the arrow draw option.

Int_t TArrowEditor::GetShapeEntry(Option_t *option)
{
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
