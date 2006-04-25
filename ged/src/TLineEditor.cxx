// @(#)root/ged:$Name:  $:$Id: TLineEditor.cxx,v 1.1 2006/04/25 08:14:20 antcheva Exp $
// Author: Ilka  Antcheva 24/04/06

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TLineEditor                                                        //
//                                                                      //
//  Implements GUI for editing line attributes: shape, size, angle.    //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TLineEditor.gif">
*/
//End_Html


#include "TLineEditor.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TLine.h"
#include "TVirtualPad.h"
#include "TClass.h"

ClassImp(TLineEditor)

enum ELineWid {
   kLine_STAX,
   kLine_STAY,
   kLine_ENDX,
   kLine_ENDY,
   kLine_VERTICAL,
   kLine_HORIZONTAL
};


//______________________________________________________________________________
TLineEditor::TLineEditor(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of line GUI.

   fLine = 0;

   MakeTitle("Points");

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fStartPointXLabel = new TGLabel(f3, "Start X:");
   f3->AddFrame(fStartPointXLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fStartPointX = new TGNumberEntry(f3, 0.0, 8, kLine_STAX,
                                      TGNumberFormat::kNESRealThree,
                                      TGNumberFormat::kNEAAnyNumber,
                                      TGNumberFormat::kNELNoLimits);
   fStartPointX->GetNumberEntry()->SetToolTipText("Set start point X coordinate of Line.");
   f3->AddFrame(fStartPointX, new TGLayoutHints(kLHintsLeft, 11, 1, 1, 1));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fStartPointYLabel = new TGLabel(f4, "Y:");
   f4->AddFrame(fStartPointYLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 37, 0, 1, 1));
   fStartPointY = new TGNumberEntry(f4, 0.0, 8, kLine_STAY,
                                      TGNumberFormat::kNESRealThree,
                                      TGNumberFormat::kNEAAnyNumber,
                                      TGNumberFormat::kNELNoLimits);
   fStartPointY->GetNumberEntry()->SetToolTipText("Set start point Y coordinate of Line.");
   f4->AddFrame(fStartPointY, new TGLayoutHints(kLHintsLeft, 10, 1, 1, 1));

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fEndPointXLabel = new TGLabel(f5, "End  X:");
   f5->AddFrame(fEndPointXLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 7, 0, 1, 1));
   fEndPointX = new TGNumberEntry(f5, 0.0, 8, kLine_ENDX,
                                    TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEAAnyNumber,
                                    TGNumberFormat::kNELNoLimits);
   fEndPointX->GetNumberEntry()->SetToolTipText("Set end point X xoordinate of Line.");
   f5->AddFrame(fEndPointX, new TGLayoutHints(kLHintsLeft, 11, 1, 1, 1));

   TGCompositeFrame *f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fEndPointYLabel = new TGLabel(f6, "Y:");
   f6->AddFrame(fEndPointYLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 37, 0, 1, 1));
   fEndPointY = new TGNumberEntry(f6, 0.0, 8, kLine_ENDY,
                                    TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEAAnyNumber,
                                    TGNumberFormat::kNELNoLimits);
   fEndPointY->GetNumberEntry()->SetToolTipText("Set end point Y coordinate of Line.");
   f6->AddFrame(fEndPointY, new TGLayoutHints(kLHintsLeft, 11, 1, 1, 1));

   TGCompositeFrame *f7 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f7, new TGLayoutHints(kLHintsTop, 1, 1, 5, 0));

   fVertical = new TGCheckButton(f7,"Vertical",kLine_VERTICAL);
   fVertical->SetToolTipText("Set vertical");
   f7->AddFrame(fVertical, new TGLayoutHints(kLHintsTop, 5, 1, 0, 0));

   fHorizontal = new TGCheckButton(f7,"Horizontal",kLine_HORIZONTAL);
   fHorizontal->SetToolTipText("Set horizontal");
   f7->AddFrame(fHorizontal, new TGLayoutHints(kLHintsTop, 5, 1, 0, 0));

   TClass *cl = TLine::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TLineEditor::~TLineEditor()
{
   // Destructor of line editor.

   TGFrameElement *el;
   TIter next(GetList());

   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________
void TLineEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fStartPointX->Connect("ValueSet(Long_t)", "TLineEditor", this, "DoStartPoint()");
   (fStartPointX->GetNumberEntry())->Connect("ReturnPressed()", "TLineEditor", this, "DoStartPoint()");
   fStartPointY->Connect("ValueSet(Long_t)", "TLineEditor", this, "DoStartPoint()");
   (fStartPointY->GetNumberEntry())->Connect("ReturnPressed()", "TLineEditor", this, "DoStartPoint()");
   fEndPointX->Connect("ValueSet(Long_t)", "TLineEditor", this, "DoEndPoint()");
   (fEndPointX->GetNumberEntry())->Connect("ReturnPressed()", "TLineEditor", this, "DoEndPoint()");
   fEndPointY->Connect("ValueSet(Long_t)", "TLineEditor", this, "DoEndPoint()");
   (fEndPointY->GetNumberEntry())->Connect("ReturnPressed()", "TLineEditor", this, "DoEndPoint()");
   fVertical->Connect("Clicked()","TLineEditor",this,"DoLineVertical()");
   fHorizontal->Connect("Clicked()","TLineEditor",this,"DoLineHorizontal()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TLineEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used line attributes.

   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom(TLine::Class())) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   fLine = (TLine *)fModel;

   Float_t val = fLine->GetX1();
   fStartPointX->SetNumber(val);

   val = fLine->GetX2();
   fEndPointX->SetNumber(val);

   val = fLine->GetY1();
   fStartPointY->SetNumber(val);

   val = fLine->GetY2();
   fEndPointY->SetNumber(val);

   if (fLine->IsHorizontal()) fHorizontal->SetState(kButtonDown, kFALSE);
   else fHorizontal->SetState(kButtonUp, kFALSE);

   if (fLine->IsVertical()) fVertical->SetState(kButtonDown, kFALSE);
   else fVertical->SetState(kButtonUp, kFALSE);

   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TLineEditor::DoStartPoint()
{
   // Slot connected to the line start point.

   fLine->SetX1((Double_t)fStartPointX->GetNumber());
   fLine->SetY1((Double_t)fStartPointY->GetNumber());
   fLine->Paint(fLine->GetDrawOption());
   Update();
}
//______________________________________________________________________________
void TLineEditor::DoEndPoint()
{
   // Slot connected to the line EndPoint.

   fLine->SetX2((Double_t)fEndPointX->GetNumber());
   fLine->SetY2((Double_t)fEndPointY->GetNumber());
   fLine->Paint(fLine->GetDrawOption());
   Update();
}

//______________________________________________________________________________                                                                                
void TLineEditor::DoLineVertical()
{
   // Slot so set the line vertical

   if (fVertical->GetState() == kButtonDown) {
      fLine->SetVertical();
      fHorizontal->SetState(kButtonUp, kFALSE);
   } else {
      fLine->SetVertical(kFALSE);
   }
   Update();
}

//______________________________________________________________________________                                                                                
void TLineEditor::DoLineHorizontal()
{
   // Slot so set the line horizontal

   if (fHorizontal->GetState() == kButtonDown) {
      fLine->SetHorizontal();
      fVertical->SetState(kButtonUp, kFALSE);
   } else {
      fLine->SetHorizontal(kFALSE);
   }
   Update();
}
