// @(#)root/ged:$Id$
// Author: Ilka Antcheva   08/05/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TFunctionParametersDialog                                           //
//                                                                      //
//  This class is used for function parameter settings.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFunctionParametersDialog.h"
#include "TTimer.h"
#include "TList.h"
#include "TF1.h"
#include "TGButton.h"
#include "TGFrame.h"
#include "TGLabel.h"
#include "TGLayout.h"
#include "TGTextEntry.h"
#include "TGMsgBox.h"
#include "TGNumberEntry.h"
#include "TGTripleSlider.h"
#include "TVirtualPad.h"


enum EParametersDialogWid {
   kNAME,
   kFIX = 10,
   kVAL = 20,
   kMIN = 30,
   kMAX = 40,
   kSLD = 50,
   kUPDATE = 8888,
   kRESET,
   kAPPLY,
   kOK,
   kCANCEL
};

ClassImp(TFunctionParametersDialog)

//______________________________________________________________________________
TFunctionParametersDialog::TFunctionParametersDialog(const TGWindow *p,
                                                     const TGWindow *main,
                                                     TF1 *func,
                                                     TVirtualPad *pad,
                                                     Double_t rx, Double_t ry) :
   TGTransientFrame(p, main, 10, 10, kVerticalFrame)
{
   // Create the parameters' dialog of currently selected function 'func'.

   fFunc = func;
   fFpad = pad;
   fRXmin = rx;
   fRXmax = ry;
   fFunc->GetRange(fRangexmin, fRangexmax);
   fNP = fFunc->GetNpar();
   fHasChanges = kFALSE;
   fPmin = new Double_t[fNP];
   fPmax = new Double_t[fNP];
   fPval = new Double_t[fNP];
   fPerr = new Double_t[fNP];

   for (Int_t i = 0; i < fNP; i++) {
      fFunc->GetParLimits(i, fPmin[i], fPmax[i]);
      fPval[i] = fFunc->GetParameter(i);
      fPerr[i] = fFunc->GetParError(i);
   }
   fParNam = new TGTextEntry*[fNP];
   fParFix = new TGCheckButton*[fNP];
   fParVal = new TGNumberEntry*[fNP];
   fParMin = new TGNumberEntryField*[fNP];
   fParMax = new TGNumberEntryField*[fNP];
   fParSld = new TGTripleHSlider*[fNP];

   memset(fParNam, 0, sizeof(TGTextEntry*)*fNP);
   memset(fParFix, 0, sizeof(TGCheckButton*)*fNP);
   memset(fParVal, 0, sizeof(TGNumberEntry*)*fNP);
   memset(fParMin, 0, sizeof(TGNumberEntryField*)*fNP);
   memset(fParMax, 0, sizeof(TGNumberEntryField*)*fNP);
   memset(fParMax, 0, sizeof(TGTripleHSlider*)*fNP);

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f1, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   // column 'Name'
   fContNam = new TGCompositeFrame(f1, 80, 20, kVerticalFrame | kFixedWidth);
   fContNam->AddFrame(new TGLabel(fContNam,"Name"),
                      new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
   for (Int_t i = 0; i < fNP; i++ ) {
      fParNam[i] = new TGTextEntry(fContNam, new TGTextBuffer(80), kNAME+i);
      fParNam[i]->SetText(Form("%s", fFunc->GetParName(i)));
      fParNam[i]->SetEnabled(kFALSE);
      fContNam->AddFrame(fParNam[i],
                         new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 5));
   }
   f1->AddFrame(fContNam, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));

   // column 'Fix'
   fContFix = new TGCompositeFrame(f1, 20, 20, kVerticalFrame | kFixedWidth);
   fContFix->AddFrame(new TGLabel(fContFix,"Fix"),
                      new TGLayoutHints(kLHintsTop, 2, 0, 0, 0));
   for (Int_t i = 0; i < fNP; i++ ) {
      fParFix[i] = new TGCheckButton(fContFix, "", kFIX*fNP+i);
      fParFix[i]->SetToolTipText(Form("Set %s to fixed", fFunc->GetParName(i)));
      fContFix->AddFrame(fParFix[i], new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
                                                       5, 5, 10, 7));
      if ((fPmin[i] == fPmax[i]) && (fPmin[i] || fPmax[i]))
         fParFix[i]->SetState(kButtonDown);
      else
         fParFix[i]->SetState(kButtonUp);
      fParFix[i]->Connect("Toggled(Bool_t)", "TFunctionParametersDialog", this, "DoFix(Bool_t)");
   }
   f1->AddFrame(fContFix, new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5));

   // column 'Value'
   fContVal = new TGCompositeFrame(f1, 80, 20, kVerticalFrame | kFixedWidth);
   fContVal->AddFrame(new TGLabel(fContVal,"Value"),
                      new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
   for (Int_t i = 0; i < fNP; i++ ) {
      fParVal[i] = new TGNumberEntry(fContVal, 1.2E-12, 15, kVAL*fNP+i,
                                     TGNumberFormat::kNESReal);
      fParVal[i]->SetNumber(fPval[i]);
      fParVal[i]->SetFormat(TGNumberFormat::kNESReal, TGNumberFormat::kNEAAnyNumber); //tbs
      fContVal->AddFrame(fParVal[i], new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 5));
      (fParVal[i]->GetNumberEntry())->SetToolTipText(Form("%s", fFunc->GetParName(i)));
      (fParVal[i]->GetNumberEntry())->Connect("ReturnPressed()", "TFunctionParametersDialog",
                                              this, "DoParValue()");
      fParVal[i]->Connect("ValueSet(Long_t)", "TFunctionParametersDialog", this, "DoParValue()");
   }
   f1->AddFrame(fContVal, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));

   // column 'Min'
   fContMin = new TGCompositeFrame(f1, 80, 20, kVerticalFrame | kFixedWidth);
   fContMin->AddFrame(new TGLabel(fContMin,"Min"),
                      new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
   for (Int_t i = 0; i < fNP; i++ ) {
      fParMin[i] = new TGNumberEntryField(fContMin, kMIN*fNP+i, 0.0,
                                          TGNumberFormat::kNESReal,
                                          TGNumberFormat::kNEAAnyNumber);
      ((TGTextEntry*)fParMin[i])->SetToolTipText(Form("Lower limit of %s",
                                                 fFunc->GetParName(i)));
      fContMin->AddFrame(fParMin[i], new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 5));
      if (fPmin[i])
         fParMin[i]->SetNumber(fPmin[i]);
      else if (fPerr[i])
         fParMin[i]->SetNumber(fPval[i]-3*fPerr[i]);
      else if (fPval[i])
         fParMin[i]->SetNumber(fPval[i]-0.1*fPval[i]);
      else
         fParMin[i]->SetNumber(1.0);
      fParMin[i]->Connect("ReturnPressed()", "TFunctionParametersDialog", this, "DoParMinLimit()");
   }
   f1->AddFrame(fContMin, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));

   // column 'Set Range'
   fContSld = new TGCompositeFrame(f1, 120, 20, kVerticalFrame | kFixedWidth);
   fContSld->AddFrame(new TGLabel(fContSld,"Set Range"),
                      new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
   for (Int_t i = 0; i < fNP; i++ ) {
      fParSld[i] = new TGTripleHSlider(fContSld, 100, kDoubleScaleBoth, kSLD*fNP+i,
                                       kHorizontalFrame, GetDefaultFrameBackground(),
                                       kFALSE, kFALSE, kFALSE, kFALSE);
      fContSld->AddFrame(fParSld[i], new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));
      fParSld[i]->SetConstrained(kTRUE);
   }
   f1->AddFrame(fContSld,  new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));

   // column 'Max'
   fContMax = new TGCompositeFrame(f1, 80, 20, kVerticalFrame);
   fContMax->AddFrame(new TGLabel(fContMax,"Max"),
                      new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
   for (Int_t i = 0; i < fNP; i++ ) {
      fParMax[i] = new TGNumberEntryField(fContMax, kMAX*fNP+i, 0.0,
                                          TGNumberFormat::kNESReal,
                                          TGNumberFormat::kNEAAnyNumber);
      ((TGTextEntry*)fParMax[i])->SetToolTipText(Form("Upper limit of %s",
                                                 fFunc->GetParName(i)));
      fContMax->AddFrame(fParMax[i], new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 5));
      if (fPmax[i])
         fParMax[i]->SetNumber(fPmax[i]);
      else if (fPerr[i])
         fParMax[i]->SetNumber(fPval[i]+3*fPerr[i]);
      else if (fPval[i])
         fParMax[i]->SetNumber(fPval[i]+0.1*fPval[i]);
      else
         fParMax[i]->SetNumber(1.0);
      if (fParMax[i]->GetNumber() < fParMin[i]->GetNumber()){
         Double_t temp;
         temp = fParMax[i]->GetNumber();
         fParMax[i]->SetNumber(fParMin[i]->GetNumber());
         fParMin[i]->SetNumber(temp);
      }
      fParMax[i]->Connect("ReturnPressed()", "TFunctionParametersDialog", this, "DoParMaxLimit()");
   }
   f1->AddFrame(fContMax, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));


   fUpdate = new TGCheckButton(this, "&Immediate preview", kUPDATE);
   fUpdate->SetToolTipText("Immediate function redrawing");
   AddFrame(fUpdate, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 5, 5, 5, 5));
   fUpdate->Connect("Toggled(Bool_t)", "TFunctionParametersDialog", this, "HandleButtons(Bool_t)");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 270, 20, kHorizontalFrame | kFixedWidth);
   AddFrame(f2, new TGLayoutHints(kLHintsRight, 20, 20, 5, 1));

   fReset = new TGTextButton(f2, "&Reset", kRESET);
   f2->AddFrame(fReset, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,5,5,5,5));
   fReset->SetToolTipText("Reset the parameter settings");
   fReset->SetState(kButtonDisabled);
   fReset->Connect("Clicked()", "TFunctionParametersDialog", this, "DoReset()");

   fApply = new TGTextButton(f2, "&Apply", kAPPLY);
   f2->AddFrame(fApply, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,5,5,5,5));
   fApply->SetState(kButtonDisabled);
   fApply->Connect("Clicked()", "TFunctionParametersDialog", this, "DoApply()");
   fApply->SetToolTipText("Apply parameter settings and redraw the function");

   fOK = new TGTextButton(f2, "&OK", kOK);
   f2->AddFrame(fOK, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,5,5,5,5));
   fOK->SetToolTipText("Apply parameter settings, redraw function and close this dialog");
   fOK->Connect("Clicked()", "TFunctionParametersDialog", this, "DoOK()");

   fCancel = new TGTextButton(f2, "&Cancel", kCANCEL);
   f2->AddFrame(fCancel, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,5,5,5,5));
   fCancel->SetToolTipText("Close this dialog with no parameter changes");
   fCancel->Connect("Clicked()", "TFunctionParametersDialog", this, "DoCancel()");

   MapSubwindows();
   Resize(GetDefaultSize());
   CenterOnParent(kFALSE, kBottomLeft);
   SetWindowName(Form("Set Parameters of %s", fFunc->GetTitle()));
   MapWindow();

   for (Int_t i = 0; i < fNP; i++ ) {
      if (fParFix[i]->GetState() == kButtonDown) {
         fParVal[i]->SetState(kFALSE);
         fParMin[i]->SetEnabled(kFALSE);
         fParMax[i]->SetEnabled(kFALSE);
         fParSld[i]->UnmapWindow();
      } else {
         fParSld[i]->SetRange(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
         fParSld[i]->SetPosition(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
         fParSld[i]->SetPointerPosition(fPval[i]);
         fParSld[i]->Connect("PointerPositionChanged()", "TFunctionParametersDialog",
                             this, "DoSlider()");
         fParSld[i]->Connect("PositionChanged()", "TFunctionParametersDialog",
                             this, "DoSlider()");
      }
   }

   gClient->WaitFor(this);
}

//______________________________________________________________________________
TFunctionParametersDialog::~TFunctionParametersDialog()
{
   // Destructor.

   TGFrameElement *el;
   TIter next(GetList());

   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame")) {
         TGFrameElement *el1;
         TIter next1(((TGCompositeFrame *)el->fFrame)->GetList());
         while ((el1 = (TGFrameElement *)next1())) {
            if (!strcmp(el1->fFrame->ClassName(), "TGCompositeFrame"))
               ((TGCompositeFrame *)el1->fFrame)->Cleanup();
         }
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
      }
   }
   Cleanup();
   delete [] fPval;
   delete [] fPmin;
   delete [] fPmax;
   delete [] fPerr;
}

//______________________________________________________________________________
void TFunctionParametersDialog::CloseWindow()
{
   // Close parameters' dialog.

   if (fHasChanges) {
      Int_t ret;
      const char *txt;
      txt = "Do you want to apply last parameters' setting?";
      new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                   "Parameters Have Been Changed", txt, kMBIconExclamation,
                   kMBYes | kMBNo | kMBCancel, &ret);
      if (ret == kMBYes) {
         DoOK();
         return;
      } else if (ret == kMBNo) {
         DoReset();
      } else return;
   }
   DeleteWindow();
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoCancel()
{
   // Slot related to the Cancel button.

   if (fHasChanges)
      DoReset();
   TTimer::SingleShot(50, "TFunctionParametersDialog", this, "CloseWindow()");
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoFix(Bool_t on)
{
   // Slot related to the Fix check button.

   fReset->SetState(kButtonUp);
   TGButton *bt = (TGButton *) gTQSender;
   Int_t id = bt->WidgetId();
   fHasChanges = kTRUE;
   for (Int_t i = 0; i < fNP; i++ ) {
      if (id == kFIX*fNP+i) {
         if (on) {
            if (fParVal[i]->GetNumber() != 0) {
               fParMin[i]->SetNumber(fParVal[i]->GetNumber());
               fParMin[i]->SetEnabled(kFALSE);
               fParMax[i]->SetNumber(fParVal[i]->GetNumber());
               fParMax[i]->SetEnabled(kFALSE);
            } else {
               fParMin[i]->SetNumber(1.);
               fParMin[i]->SetEnabled(kFALSE);
               fParMax[i]->SetNumber(1.);
               fParMax[i]->SetEnabled(kFALSE);
            }
            fParVal[i]->SetState(kFALSE);
            fParSld[i]->Disconnect("PointerPositionChanged()");
            fParSld[i]->Disconnect("PositionChanged()");
            fParSld[i]->UnmapWindow();
            fFunc->FixParameter(i, fParVal[i]->GetNumber());
         } else if (!fParMin[i]->IsEnabled()) {
            if (fPmin[i] != fPmax[i]) {
               if (fPmin[i])
                  fParMin[i]->SetNumber(fPmin[i]);
               else if (fPerr[i])
                  fParMin[i]->SetNumber(fPval[i]-3*fPerr[i]);
               else if (fPval[i])
                  fParMin[i]->SetNumber(fPval[i]-0.1*fPval[i]);
               else
                  fParMin[i]->SetNumber(1.0);
               if (fPmax[i])
                  fParMax[i]->SetNumber(fPmax[i]);
               else if (fPerr[i])
                  fParMax[i]->SetNumber(fPval[i]+3*fPerr[i]);
               else if (fPval[i])
                  fParMax[i]->SetNumber(fPval[i]+0.1*fPval[i]);
               else
                  fParMax[i]->SetNumber(1.0);
            } else if (fPval[i]) {
               fParMin[i]->SetNumber(fPval[i]-0.1*fPval[i]);
               fParMax[i]->SetNumber(fPval[i]+0.1*fPval[i]);
            } else {
               fParMin[i]->SetNumber(1.0);
               fParMax[i]->SetNumber(1.0);
            }
            if (fParMax[i]->GetNumber() < fParMin[i]->GetNumber()){
               Double_t temp;
               temp = fParMax[i]->GetNumber();
               fParMax[i]->SetNumber(fParMin[i]->GetNumber());
               fParMin[i]->SetNumber(temp);
            }
            fParMax[i]->SetEnabled(kTRUE);
            fParMin[i]->SetEnabled(kTRUE);
            fParSld[i]->MapWindow();
            fParVal[i]->SetState(kTRUE);
            fParSld[i]->SetRange(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
            fParSld[i]->SetPosition(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
            fParSld[i]->SetPointerPosition(fPval[i]);
            fParSld[i]->Connect("PointerPositionChanged()", "TFunctionParametersDialog",
                                this, "DoSlider()");
            fParSld[i]->Connect("PositionChanged()", "TFunctionParametersDialog",
                                this, "DoSlider()");
            fFunc->SetParLimits(i, fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
         }
      }
   }
   if (fUpdate->GetState() == kButtonDown)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges)
      fApply->SetState(kButtonUp);
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoOK()
{
   // Slot related to the OK button.

   if (fHasChanges)
      RedrawFunction();
   fFunc->SetRange(fRangexmin, fRangexmax);
   TTimer::SingleShot(50, "TFunctionParametersDialog", this, "CloseWindow()");
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoApply()
{
   // Slot related to the Preview button.

   RedrawFunction();
   fApply->SetState(kButtonDisabled);
   if (fReset->GetState() == kButtonDisabled)
      fReset->SetState(kButtonUp);
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoReset()
{
   // Slot related to the Reset button.

   fHasChanges = kTRUE;
   Int_t k = fNP;
   for (Int_t i = 0; i < fNP; i++) {
      if (fParVal[i]->GetNumber() == fPval[i])
         k--;
      else
         break;
   }

   if (!k) {
      if (fReset->GetState() == kButtonUp)
         fReset->SetState(kButtonDisabled);
      fHasChanges = kFALSE;
      return;
   }
   for (Int_t i = 0; i < fNP; i++) {
      fFunc->SetParameter(i, fPval[i]);
      fFunc->SetParLimits(i, fPmin[i], fPmax[i]);
      fFunc->SetParError(i, fPerr[i]);

      if (fPmin[i])
         fParMin[i]->SetNumber(fPmin[i]);
      else if (fPerr[i])
         fParMin[i]->SetNumber(fPval[i]-3*fPerr[i]);
      else if (fPval[i])
         fParMin[i]->SetNumber(fPval[i]-0.1*fPval[i]);
      else
         fParMin[i]->SetNumber(1.0);

      if (fPmax[i])
         fParMax[i]->SetNumber(fPmax[i]);
      else if (fPerr[i])
         fParMax[i]->SetNumber(fPval[i]+3*fPerr[i]);
      else if (fPval[i])
         fParMax[i]->SetNumber(fPval[i]+0.1*fPval[i]);
      else
         fParMax[i]->SetNumber(1.0);
      if (fParMax[i]->GetNumber() < fParMin[i]->GetNumber()){
         Double_t temp;
         temp = fParMax[i]->GetNumber();
         fParMax[i]->SetNumber(fParMin[i]->GetNumber());
         fParMin[i]->SetNumber(temp);
      }
      if (fParMin[i]->GetNumber() == fParMax[i]->GetNumber()) {
         fParVal[i]->SetState(kFALSE);
         fParMin[i]->SetEnabled(kFALSE);
         fParMax[i]->SetEnabled(kFALSE);
         fParSld[i]->Disconnect("PointerPositionChanged()");
         fParSld[i]->Disconnect("PositionChanged()");
         fParSld[i]->UnmapWindow();
         fFunc->FixParameter(i, fParVal[i]->GetNumber());
         fParFix[i]->SetState(kButtonDown);
      } else {
         fParFix[i]->SetState(kButtonUp);
         if (!fParMax[i]->IsEnabled()) {
            fParMax[i]->SetEnabled(kTRUE);
            fParMin[i]->SetEnabled(kTRUE);
            fParVal[i]->SetState(kTRUE);
            fParSld[i]->SetRange(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
            fParSld[i]->SetPosition(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
            fParSld[i]->SetPointerPosition(fPval[i]);
            fParSld[i]->MapWindow();
            fParSld[i]->Connect("PointerPositionChanged()", "TFunctionParametersDialog",
                                this, "DoSlider()");
            fParSld[i]->Connect("PositionChanged()", "TFunctionParametersDialog",
                                this, "DoSlider()");
         }
      }
      fParVal[i]->SetNumber(fPval[i]);

      fParSld[i]->SetRange(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
      fParSld[i]->SetPosition(fParMin[i]->GetNumber(), fParMax[i]->GetNumber());
      fParSld[i]->SetPointerPosition(fPval[i]);
   }

   if (fUpdate->GetState() == kButtonDown)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges)
      fApply->SetState(kButtonUp);
   fHasChanges = kFALSE;
   fReset->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoSlider()
{
   // Slot related to the parameters' value settings.

   TGTripleHSlider *sl = (TGTripleHSlider *) gTQSender;
   Int_t id = sl->WidgetId();

   fHasChanges = kTRUE;
   for (Int_t i = 0; i < fNP; i++ ) {
      if (id == kSLD*fNP+i) {
         fFunc->SetParameter(i,fParSld[i]->GetPointerPosition());
         fFunc->SetParLimits(i,fParSld[i]->GetMinPosition(),
                               fParSld[i]->GetMaxPosition());
         fParMin[i]->SetNumber(fParSld[i]->GetMinPosition());
         fParMax[i]->SetNumber(fParSld[i]->GetMaxPosition());
         fParVal[i]->SetNumber(fParSld[i]->GetPointerPosition());
      }
   }
   if (fUpdate->GetState() == kButtonDown)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges)
      fApply->SetState(kButtonUp);
   if (fReset->GetState() == kButtonDisabled)
      fReset->SetState(kButtonUp);
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoParValue()
{
   // Slot related to the parameter value settings.

   TGNumberEntry *ne = (TGNumberEntry *) gTQSender;
   Int_t id = ne->WidgetId();

   for (Int_t i = 0; i < fNP; i++ ) {
      if (id == kVAL*fNP+i)  {
         fParSld[i]->SetPointerPosition(fParVal[i]->GetNumber());
         if (fParVal[i]->GetNumber() < fParMin[i]->GetNumber()) {
            fParMin[i]->SetNumber(fParVal[i]->GetNumber());
            fClient->NeedRedraw(fParMin[i]);
            fParSld[i]->SetRange(fParMin[i]->GetNumber(),
                                 fParMax[i]->GetNumber());
            fParSld[i]->SetPosition(fParMin[i]->GetNumber(),
                                    fParMax[i]->GetNumber());
         }
         if (fParVal[i]->GetNumber() > fParMax[i]->GetNumber()) {
            fParMax[i]->SetNumber(fParVal[i]->GetNumber());
            fClient->NeedRedraw(fParMax[i]);
            fParSld[i]->SetRange(fParMin[i]->GetNumber(),
                                 fParMax[i]->GetNumber());
            fParSld[i]->SetPosition(fParMin[i]->GetNumber(),
                                    fParMax[i]->GetNumber());
         }
         fClient->NeedRedraw(fParSld[i]);
         fFunc->SetParameter(i,fParSld[i]->GetPointerPosition());
         fFunc->SetParLimits(i,fParSld[i]->GetMinPosition(),
                               fParSld[i]->GetMaxPosition());
      }
   }
   fHasChanges = kTRUE;
   if (fUpdate->GetState() == kButtonDown)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges)
      fApply->SetState(kButtonUp);
   if (fReset->GetState() == kButtonDisabled)
      fReset->SetState(kButtonUp);
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoParMinLimit()
{
   // Slot related to the minumum parameter limit settings.

   TGNumberEntryField *ne = (TGNumberEntryField *) gTQSender;
   Int_t id = ne->WidgetId();

   for (Int_t i = 0; i < fNP; i++ ) {
      if (id == kMIN*fNP+i) {
         if (fParMin[i]->GetNumber() > fParMax[i]->GetNumber()) {
            Int_t ret;
            const char *txt;
            txt = "The lower parameter bound cannot be bigger then the upper one.";
            new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                         "Parameter Limits", txt, kMBIconExclamation,kMBOk,&ret);
            fParMin[i]->SetNumber(fParVal[i]->GetNumber());
            return;
         }
         fParSld[i]->SetRange(fParMin[i]->GetNumber(),
                              fParMax[i]->GetNumber());
         fParSld[i]->SetPosition(fParMin[i]->GetNumber(),
                                 fParMax[i]->GetNumber());
         fParSld[i]->SetPointerPosition(fParVal[i]->GetNumber());
         fClient->NeedRedraw(fParSld[i]);
      }
   }
   fHasChanges = kTRUE;
   if (fUpdate->GetState() == kButtonDown)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges)
      fApply->SetState(kButtonUp);
   if (fReset->GetState() == kButtonDisabled)
      fReset->SetState(kButtonUp);
}

//______________________________________________________________________________
void TFunctionParametersDialog::DoParMaxLimit()
{
   // Slot related to the maximum parameter limit settings.

   TGNumberEntryField *ne = (TGNumberEntryField *) gTQSender;
   Int_t id = ne->WidgetId();

   for (Int_t i = 0; i < fNP; i++ ) {
      if (id == kMAX*fNP+i) {
         if (fParMin[i]->GetNumber() > fParMax[i]->GetNumber()) {
            Int_t ret;
            const char *txt;
            txt = "The lower parameter bound cannot be bigger then the upper one.";
            new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                         "Parameter Limits", txt, kMBIconExclamation,kMBOk,&ret);
            fParMax[i]->SetNumber(fParVal[i]->GetNumber());
            return;
         }
         fParSld[i]->SetRange(fParMin[i]->GetNumber(),
                              fParMax[i]->GetNumber());
         fParSld[i]->SetPosition(fParMin[i]->GetNumber(),
                                 fParMax[i]->GetNumber());
         fParSld[i]->SetPointerPosition(fParVal[i]->GetNumber());
         fClient->NeedRedraw(fParSld[i]);
      }
   }
   fHasChanges = kTRUE;
   if (fUpdate->GetState() == kButtonDown)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges)
      fApply->SetState(kButtonUp);
   if (fReset->GetState() == kButtonDisabled)
      fReset->SetState(kButtonUp);
}

//______________________________________________________________________________
void TFunctionParametersDialog::RedrawFunction()
{
   // Redraw function graphics.

   TString opt = fFunc->GetDrawOption();
   opt.ToUpper();
   if (!opt.Contains("SAME"))
      opt += "SAME";
   fFunc->SetRange(fRXmin, fRXmax);
   fFunc->Draw(opt);
   fFpad->Modified();
   fFpad->Update();
   fHasChanges = kFALSE;
}

//______________________________________________________________________________
void TFunctionParametersDialog::HandleButtons(Bool_t update)
{
   // Handle the button dependent states in this dialog.

   if (update && fHasChanges)
      RedrawFunction();
   else if ((fApply->GetState() == kButtonDisabled) && fHasChanges) {
      fApply->SetState(kButtonUp);
   }
}
