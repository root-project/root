// @(#)root/ged:$Id$
// Author: Ilka Antcheva

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TPaveStatsEditor                                                    //
//                                                                      //
//  Implements GUI for editing attributes of TPaveStats objects.        //                                             //
//      all text attributes                                             //
//      The following statistics option settings can be set:            //
//      name, mean, RMS, overflow, underflow, integral of bins,         //
//      Fit parameters that can be set are: Values/Names, Probability,  //
//      Errors, Chisquare                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TPaveStatsEditor.gif">
*/
//End_Html

#include "TPaveStatsEditor.h"
#include "TGButton.h"
#include "TPaveStats.h"

ClassImp(TPaveStatsEditor)

enum EPaveStatsWid {
   kSTAT_NAME,
   kSTAT_ENTRIES,
   kSTAT_MEAN,
   kSTAT_RMS,
   kSTAT_UNDER,
   kSTAT_OVER,
   kSTAT_INTEGRAL,
   kSTAT_SKEWNESS,
   kSTAT_KURTOSIS,
   kSTAT_ERR,
   kFIT_NAME,
   kFIT_ERR,
   kFIT_CHI,
   kFIT_PROB
};


//______________________________________________________________________________
TPaveStatsEditor::TPaveStatsEditor(const TGWindow *p, Int_t width, Int_t height,
   UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of TPaveStats GUI.

   fPaveStats = 0;

   MakeTitle("Stat Options");

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGCompositeFrame *f2 = new TGCompositeFrame(f1, 40, 20, kVerticalFrame);
   f1->AddFrame(f2, new TGLayoutHints(kLHintsTop, 0, 1, 0, 0));

   fHistoName = new TGCheckButton(f2, "Name", kSTAT_NAME);
   fHistoName->SetToolTipText("Print the histogram name");
   f2->AddFrame(fHistoName, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fOverflow = new TGCheckButton(f2, "Overflow", kSTAT_OVER);
   fOverflow->SetToolTipText("Print the number of overflows");
   f2->AddFrame(fOverflow, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fUnderflow = new TGCheckButton(f2, "Underflow", kSTAT_UNDER);
   fUnderflow->SetToolTipText("Print the number of underflows");
   f2->AddFrame(fUnderflow, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fSkewness = new TGCheckButton(f2, "Skewness", kSTAT_SKEWNESS);
   fSkewness->SetToolTipText("Print the skewness");
   f2->AddFrame(fSkewness, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fKurtosis = new TGCheckButton(f2, "Kurtosis", kSTAT_KURTOSIS);
   fKurtosis->SetToolTipText("Print the kurtosis");
   f2->AddFrame(fKurtosis, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f3 = new TGCompositeFrame(f1, 40, 20, kVerticalFrame);
   fEntries = new TGCheckButton(f3, "Entries", kSTAT_ENTRIES);
   fEntries->SetToolTipText("Print the number of entries");
   f3->AddFrame(fEntries, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fMean = new TGCheckButton(f3, "Mean", kSTAT_MEAN);
   fMean->SetToolTipText("Print the mean value");
   f3->AddFrame(fMean, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fRMS = new TGCheckButton(f3, "RMS", kSTAT_RMS);
   fRMS->SetToolTipText("Print root-mean-square (RMS)");
   f3->AddFrame(fRMS, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fIntegral = new TGCheckButton(f3, "Integral", kSTAT_INTEGRAL);
   fIntegral->SetToolTipText("Print the integral of bins");
   f3->AddFrame(fIntegral, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fStatsErrors = new TGCheckButton(f3, "Errors", kSTAT_ERR);
   fStatsErrors->SetToolTipText("Print the errors");
   f3->AddFrame(fStatsErrors, new TGLayoutHints(kLHintsTop, 1, 1, 0, 5));
   f1->AddFrame(f3, new TGLayoutHints(kLHintsTop, 0, 1, 0, 0));

   AddFrame(f1, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   MakeTitle("Fit Options");

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGCompositeFrame *f5 = new TGCompositeFrame(f4, 40, 20, kVerticalFrame);
   f4->AddFrame(f5, new TGLayoutHints(kLHintsTop, 0, 1, 0, 0));

   fNameValues = new TGCheckButton(f5, "Values", kFIT_NAME);
   fNameValues->SetToolTipText("Print the parameter name and value");
   f5->AddFrame(fNameValues, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fProbability = new TGCheckButton(f5, "Probability", kFIT_PROB);
   fProbability->SetToolTipText("Print probability");
   f5->AddFrame(fProbability, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f6 = new TGCompositeFrame(f4, 40, 20, kVerticalFrame);
   fErrors = new TGCheckButton(f6, "Errors", kFIT_ERR);
   fErrors->SetToolTipText("Print the errors");
   f6->AddFrame(fErrors, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fChisquare = new TGCheckButton(f6, "Chi", kFIT_CHI);
   fChisquare->SetToolTipText("Print Chisquare");
   f6->AddFrame(fChisquare, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   f4->AddFrame(f6, new TGLayoutHints(kLHintsTop, 0, 1, 0, 0));

   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}

//______________________________________________________________________________
TPaveStatsEditor::~TPaveStatsEditor()
{
  // Destructor of fill editor.
}

//______________________________________________________________________________
void TPaveStatsEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   // about stat options
   fHistoName->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fEntries->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fOverflow->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fMean->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fUnderflow->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fRMS->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fIntegral->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fSkewness->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fKurtosis->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");
   fStatsErrors->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoStatOptions()");

   // about fit options
   fNameValues->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoFitOptions()");
   fErrors->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoFitOptions()");
   fErrors->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"SetValuesON(Bool_t");
   fProbability->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoFitOptions()");
   fChisquare->Connect("Toggled(Bool_t)","TPaveStatsEditor",this,"DoFitOptions()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TPaveStatsEditor::SetModel(TObject* obj)
{
   // Set GUI widgets according to the used TPaveStats attributes.

   fPaveStats = (TPaveStats *)obj;
   fAvoidSignal = kTRUE;

   Int_t stat = fPaveStats->GetOptStat();

   if (stat % 10)  fHistoName->SetState(kButtonDown);
   else fHistoName->SetState(kButtonUp);

   if (stat/10 % 10) fEntries->SetState(kButtonDown);
   else fEntries->SetState(kButtonUp);

   if (stat/100 % 10) fMean->SetState(kButtonDown);
   else fMean->SetState(kButtonUp);

   if (stat/1000 % 10) fRMS->SetState(kButtonDown);
   else fRMS->SetState(kButtonUp);

   if (stat/10000 % 10) fUnderflow->SetState(kButtonDown);
   else fUnderflow->SetState(kButtonUp);

   if (stat/100000 % 10) fOverflow->SetState(kButtonDown);
   else fOverflow->SetState(kButtonUp);

   if (stat/1000000 % 10) fIntegral->SetState(kButtonDown);
   else fIntegral->SetState(kButtonUp);

   if (stat/10000000 % 10) fSkewness->SetState(kButtonDown);
   else fSkewness->SetState(kButtonUp);

   if (stat/100000000 % 10) fKurtosis->SetState(kButtonDown);
   else fKurtosis->SetState(kButtonUp);

   Int_t fit = fPaveStats->GetOptFit();
   if (fit % 10)  fNameValues->SetState(kButtonDown);
   else fNameValues->SetState(kButtonUp);

   if (fit/10 % 10) {
      fErrors->SetState(kButtonDown);
      fNameValues->SetState(kButtonDown);
   } else {
      fErrors->SetState(kButtonUp);
   }

   if (fit/100 % 10) fChisquare->SetState(kButtonDown);
   else fChisquare->SetState(kButtonUp);

   if (fit/1000 % 10) fProbability->SetState(kButtonDown);
   else fProbability->SetState(kButtonUp);

   if (fInit) ConnectSignals2Slots();

   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
void TPaveStatsEditor::DoStatOptions()
{
   // Slot connected to the stat options.

   if (fAvoidSignal) return;
   Int_t stat = 0;
   if (fHistoName->GetState()   == kButtonDown) stat +=1;
   if (fEntries->GetState()     == kButtonDown) stat +=10;
   if (fMean->GetState()        == kButtonDown) stat +=100;
   if (fRMS->GetState()         == kButtonDown) stat +=1000;
   if (fUnderflow->GetState()   == kButtonDown) stat +=10000;
   if (fOverflow->GetState()    == kButtonDown) stat +=100000;
   if (fIntegral->GetState()    == kButtonDown) stat +=1000000;
   if (fSkewness->GetState()    == kButtonDown) stat +=10000000;
   if (fKurtosis->GetState()    == kButtonDown) stat +=100000000;
   if (fStatsErrors->GetState() == kButtonDown) {
      if (fMean->GetState()     == kButtonDown) stat +=100;
      if (fRMS->GetState()      == kButtonDown) stat +=1000;
      if (fSkewness->GetState() == kButtonDown) stat +=10000000;
      if (fKurtosis->GetState() == kButtonDown) stat +=100000000;
   }

   if (!stat) {
      stat = 1;
      fHistoName->SetState(kButtonDown);
   }
   if (stat == 1) stat = 1000000001;
   fPaveStats->SetOptStat(stat);
   Update();
}

//______________________________________________________________________________
void TPaveStatsEditor::DoFitOptions()
{
   // Slot connected to the fit options.

   if (fAvoidSignal) return;
   Int_t fit = 0;
   if (fNameValues->GetState()  == kButtonDown) fit +=1;
   if (fErrors->GetState()      == kButtonDown) fit +=10;
   if (fChisquare->GetState()   == kButtonDown) fit +=100;
   if (fProbability->GetState() == kButtonDown) fit +=1000;

   if (fit == 1) fit = 10001;
   fPaveStats->SetOptFit(fit);
   Update();
}

//______________________________________________________________________________
void TPaveStatsEditor::SetValuesON(Bool_t on)
{
   // Slot connected to the selection of the button 'Errors':
   // check button Values should be selected if Errors is selected.

   if (fAvoidSignal) return;
   if (on == kTRUE) fNameValues->SetState(kButtonDown);
}
