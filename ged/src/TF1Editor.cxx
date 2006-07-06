// @(#)root/ged:$Name:  $:$Id:
// Author: Ilka Antcheva 21/03/06

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TF1Editor                                                           //
//                                                                      //
//  GUI for TF1 attributes and parameters.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TF1Editor.h"
#include "TH1.h"
#include "TGedFrame.h"
#include "TGTextEntry.h"
#include "TGToolTip.h"
#include "TGLabel.h"
#include "TGDoubleSlider.h"
#include "TGClient.h"
#include "TVirtualPad.h"
#include "TString.h"
#include "TGNumberEntry.h"
#include "TG3DLine.h"
#include "TFunctionParametersDialog.h"
#include "TCanvas.h"


ClassImp(TF1Editor)

enum ETF1Wid {
   kTF1_TIT,  kTF1_NPX,
   kTF1_XSLD, kTF1_XMIN, kTF1_XMAX,
   kTF1_PAR
};

//______________________________________________________________________________
TF1Editor::TF1Editor(const TGWindow *p, Int_t id, Int_t width, Int_t height, 
                     UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of TF1 editor.
 
   MakeTitle("Function");
   
   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kTF1_TIT);
   fTitle->Resize(137, fTitle->GetDefaultHeight());
   fTitle->SetEnabled(kFALSE);
   fTitle->SetToolTipText(Form("Function expression or predefined name"));
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft,3, 2, 2, 3));
   
   TGCompositeFrame *f3 = new TGCompositeFrame(this, 137, 20, kHorizontalFrame | kFixedWidth);
   fSetPars = new TGTextButton(f3, "Set Parameters...", kTF1_PAR);
   f3->AddFrame(fSetPars, new TGLayoutHints(kLHintsRight | kLHintsTop | kLHintsExpandX, 
                                            0, 1, 5, 0));
   fSetPars->SetToolTipText("Open a dialog for parameter(s) settings");
   AddFrame(f3, new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 2, 2, 3));
   
   MakeTitle("X-Range");
   
   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGCompositeFrame *f4a = new TGCompositeFrame(f4, 66, 20, kVerticalFrame | kFixedWidth);
   TGLabel *fNpxLabel = new TGLabel(f4a, "Points: ");
   f4a->AddFrame(fNpxLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 0, 5, 1));
   f4->AddFrame(f4a, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 1, 0, 0));
 
   TGCompositeFrame *f4b = new TGCompositeFrame(f4, 40, 20, kVerticalFrame);
   fNXpoints = new TGNumberEntry(f4b, 100, 7, kTF1_NPX, 
                                 TGNumberFormat::kNESInteger,
                                 TGNumberFormat::kNEANonNegative, 
                                 TGNumberFormat::kNELLimitMinMax,4,100000);
   fNXpoints->GetNumberEntry()->SetToolTipText("Points along x-axis (4-100 000)");
   f4b->AddFrame(fNXpoints, new TGLayoutHints(kLHintsLeft, 0, 0, 1, 0));
   f4->AddFrame(f4b, new TGLayoutHints(kLHintsTop | kLHintsRight, 0, 1, 0, 0));
   AddFrame(f4, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 1, 0, 0));

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fSliderX = new TGDoubleHSlider(f5, 1, 2);
   fSliderX->Resize(137,20);
   f5->AddFrame(fSliderX, new TGLayoutHints(kLHintsLeft));
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 3, 7, 4, 1));
   
   TGCompositeFrame *f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fSldMinX = new TGNumberEntryField(f6, kTF1_XMIN, 0.0,  
                                     TGNumberFormat::kNESRealFour,
                                     TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldMinX)->SetToolTipText("Lower bound along x-axis");
   fSldMinX->Resize(65,20);
   fSldMinX->SetState(kFALSE);
   f6->AddFrame(fSldMinX, new TGLayoutHints(kLHintsLeft));
   fSldMaxX = new TGNumberEntryField(f6, kTF1_XMAX, 0.0,  
                                     TGNumberFormat::kNESRealFour,
                                     TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldMaxX)->SetToolTipText("Upper bound along x-axis");
   fSldMaxX->SetState(kFALSE);
   fSldMaxX->Resize(65,20);
   f6->AddFrame(fSldMaxX, new TGLayoutHints(kLHintsLeft, 4, 0, 0, 0));
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 3, 3, 5, 0));

   TClass *cl = TF1::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TF1Editor::~TF1Editor()
{
   // Destructor of TF1 editor.
   
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
}

//______________________________________________________________________________
void TF1Editor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fNXpoints->Connect("ValueSet(Long_t)", "TF1Editor", this, "DoXPoints()");
   (fNXpoints->GetNumberEntry())->Connect("ReturnPressed()", "TF1Editor", this, "DoXPoints()");
   fSetPars->Connect("Clicked()", "TF1Editor", this, "DoParameterSettings()");
   fSliderX->Connect("PositionChanged()","TF1Editor", this,"DoSliderXMoved()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TF1Editor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the function parameters and options.

   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom(TF1::Class()) || 
       obj->InheritsFrom("TF2") || obj->InheritsFrom("TF3")) {
      SetActive(kFALSE);
      return;
   }
   
   fModel = obj;
   fPad = pad;
   TF1 *fF1 = (TF1*)fModel;

   const char *text = fF1->GetTitle();
   fTitle->SetText(text);
   
   fNP = fF1->GetNpar();
   fNXpoints->SetNumber(fF1->GetNpx());

   if (!fNP)
      fSetPars->SetState(kButtonDisabled, kFALSE);
   else
      fSetPars->SetState(kButtonUp, kFALSE);
   
   TAxis *x = fF1->GetHistogram()->GetXaxis();
   Int_t nx = x->GetNbins();
   Int_t nxbinmin = x->GetFirst();
   Int_t nxbinmax = x->GetLast();
   fSliderX->SetRange(1,nx);
   fSliderX->SetPosition((Double_t)nxbinmin,(Double_t)nxbinmax);
   fSldMinX->SetNumber(x->GetBinLowEdge(nxbinmin));
   fSldMaxX->SetNumber(x->GetBinUpEdge(nxbinmax));

   if (fInit) ConnectSignals2Slots();
   SetActive(kTRUE);
}

//______________________________________________________________________________
void TF1Editor::DoParameterSettings()
{
   // Slot connected to the function parameter(s) settings.
   
   TF1 *fF1 = (TF1*)fModel;
   TGMainFrame *main =  (TGMainFrame *)GetMainFrame();
   Double_t rmin = fSldMinX->GetNumber();
   Double_t rmax = fSldMaxX->GetNumber();
   new TFunctionParametersDialog(gClient->GetDefaultRoot(), main, 
                                 fF1, fPad, rmin, rmax);

}

//______________________________________________________________________________
void TF1Editor::DoXPoints()
{
   // Slot connected to the number of points setting.

   TF1 *fF1 = (TF1*)fModel;
   Double_t rmin, rmax;
   fF1->GetRange(rmin, rmax);
   fF1->SetRange(fSldMinX->GetNumber(), fSldMaxX->GetNumber());
   fF1->SetNpx((Int_t)fNXpoints->GetNumber());
   fF1->GetHistogram()->GetXaxis()->Set((Int_t)fNXpoints->GetNumber(),
                                          fSldMinX->GetNumber(),
                                          fSldMaxX->GetNumber());
   Update();
   fF1->SetRange(rmin, rmax);
}

//______________________________________________________________________________
void TF1Editor::DoSliderXMoved()
{
   // Slot connected to the x-Slider range for function redrawing.

   TF1 *fF1 = (TF1*)fModel;
   fF1->SetNpx((Int_t)fNXpoints->GetNumber());
   TAxis *x = fF1->GetHistogram()->GetXaxis();
   
   fPad->cd();
   TString opt = fF1->GetDrawOption();
   opt.ToUpper();
   if (!opt.Contains("SAME"))
      opt += "SAME";
   fF1->Draw(opt);

   x->SetRange((Int_t)((fSliderX->GetMinPosition())+0.5),
               (Int_t)((fSliderX->GetMaxPosition())+0.5));
   fSldMinX->SetNumber(x->GetBinLowEdge(x->GetFirst()));
   fSldMaxX->SetNumber(x->GetBinUpEdge(x->GetLast())); 
   fClient->NeedRedraw(fSliderX,kTRUE);  
   fClient->NeedRedraw(fSldMinX,kTRUE);
   fClient->NeedRedraw(fSldMaxX,kTRUE);
   Update();   
}

//______________________________________________________________________________
void TF1Editor::DoXRange()
{
   // Slot connected to the Min/Max x-axis settings of the slider range.

   TF1 *fF1 = (TF1*)fModel;
   TAxis *x = fF1->GetHistogram()->GetXaxis();
   Int_t nx = x->GetNbins();
   Double_t width = x->GetBinWidth(1);
   Double_t lowLimit = x->GetBinLowEdge(1);
   Double_t upLimit = x->GetBinUpEdge(nx);
   if ((fSldMinX->GetNumber()+width/2) < (lowLimit)) 
      fSldMinX->SetNumber(lowLimit); 
   if ((fSldMaxX->GetNumber()-width/2) > (upLimit)) 
      fSldMaxX->SetNumber(upLimit); 
   x->SetRangeUser(fSldMinX->GetNumber()+width/2, 
                   fSldMaxX->GetNumber()-width/2);
   Int_t nxbinmin = x->GetFirst();
   Int_t nxbinmax = x->GetLast();
   fSliderX->SetPosition((Double_t)(nxbinmin),(Double_t)(nxbinmax));
   Update();
}

