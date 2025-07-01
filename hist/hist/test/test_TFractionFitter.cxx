#include "gtest/gtest.h"
#include "ROOT/TestSupport.hxx"

#include "TH1F.h"
#include "TF1.h"
#include "TFractionFitter.h"
#include "TRandom.h"

// https://its.cern.ch/jira/browse/ROOT-9330
TEST(TFractionFitter, FitExample)
{
   // pointers to the data
   TH1F *data;                              // data histogram
   TH1F *mc0;                               // first MC histogram
   TH1F *mc1;                               // second MC histogram
   TH1F *mc2;                               // third MC histogram

   // parameters and functions to generate the data
   Int_t Ndata = 1000;
   Int_t N0 = 1000;
   Int_t N1 = 1000;
   Int_t N2 = 1000;

   Int_t nBins = 40;

   Double_t trueP0 = .01;
   Double_t trueP1 = .3;
   // Double_t trueP2 = 1. - trueP0 - trueP1;

   // contribution 0
   TF1 *f0 = new TF1("f0", "[0]*(1-cos(x))/TMath::Pi()", 0., TMath::Pi());
   f0->SetParameter(0, 1.);
   f0->SetLineColor(2);
   // Double_t int0 = f0->Integral(0., TMath::Pi());

   // contribution 1
   TF1 *f1 = new TF1("f1", "[0]*(1-cos(x)*cos(x))*2./TMath::Pi()", 0., TMath::Pi());
   f1->SetParameter(0, 1.);
   f1->SetLineColor(3);
   // Double_t int1 = f1->Integral(0., TMath::Pi());

   // contribution 2
   TF1 *f2 = new TF1("f2", "[0]*(1+cos(x))/TMath::Pi()", 0., TMath::Pi());
   f2->SetParameter(0, 1.);
   f2->SetLineColor(4);
   // Double_t int2 = f2->Integral(0., TMath::Pi());

   // generate data
   data = new TH1F("data", "Data angle distribution", nBins, 0, TMath::Pi());
   data->SetXTitle("x");
   data->SetMarkerStyle(20);
   data->SetMarkerSize(.7);
   data->SetMinimum(0);
   TH1F *htruemc0 = new TH1F(*data);
   htruemc0->SetLineColor(2);
   TH1F *htruemc1 = new TH1F(*data);
   htruemc1->SetLineColor(3);
   TH1F *htruemc2 = new TH1F(*data);
   htruemc2->SetLineColor(4);
   Double_t p, x;
   for (Int_t i = 0; i < Ndata; i++) {
      p = gRandom->Uniform();
      if (p < trueP0) {
         x = f0->GetRandom();
         htruemc0->Fill(x);
      }
      else if (p < trueP0 + trueP1) {
         x = f1->GetRandom();
         htruemc1->Fill(x);
      }
      else {
         x = f2->GetRandom();
         htruemc2->Fill(x);
      }
      data->Fill(x);
   }

   // generate MC samples
   mc0 = new TH1F("mc0", "MC sample 0 angle distribution", nBins, 0, TMath::Pi());
   mc0->SetXTitle("x");
   mc0->SetLineColor(2);
   mc0->SetMarkerColor(2);
   mc0->SetMarkerStyle(24);
   mc0->SetMarkerSize(.7);
   for (Int_t i = 0; i < N0; i++) {
      mc0->Fill(f0->GetRandom());
   }

   mc1 = new TH1F("mc1", "MC sample 1 angle distribution", nBins, 0, TMath::Pi());
   mc1->SetXTitle("x");
   mc1->SetLineColor(3);
   mc1->SetMarkerColor(3);
   mc1->SetMarkerStyle(24);
   mc1->SetMarkerSize(.7);
   for (Int_t i = 0; i < N1; i++) {
      mc1->Fill(f1->GetRandom());
   }

   mc2 = new TH1F("mc2", "MC sample 2 angle distribution", nBins, 0, TMath::Pi());
   mc2->SetXTitle("x");
   mc2->SetLineColor(4);
   mc2->SetMarkerColor(4);
   mc2->SetMarkerStyle(24);
   mc2->SetMarkerSize(.7);
   for (Int_t i = 0; i < N2; i++) {
      mc2->Fill(f2->GetRandom());
   }

   // FractionFitter
   TObjArray *mc = new TObjArray(3);          // MC histograms are put in this array
   mc->Add(mc0);
   mc->Add(mc1);
   mc->Add(mc2);
   TFractionFitter *fit = new TFractionFitter(data, mc); // initialise
   fit->Constrain(0, 0.0, 1.0);               // constrain fraction 1 to be between 0 and 1
   fit->Constrain(1, 0.0, 1.0);               // constrain fraction 1 to be between 0 and 1
   fit->Constrain(2, 0.0, 1.0);               // constrain fraction 1 to be between 0 and 1
   // fit->SetRangeX(1,15);                    // use only the first 15 bins in the fit
   ROOT::TestSupport::CheckDiagsRAII diags;
   Int_t status = fit->Fit();                 // perform the fit
   EXPECT_EQ(status, 0);

   // Cleanup
   delete fit;
   delete data;
   delete mc;
   delete mc0;
   delete mc1;
   delete mc2;
   delete htruemc0;
   delete htruemc1;
   delete htruemc2;
   delete f0;
   delete f1;
   delete f2;
}
