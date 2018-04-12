/// \file
/// \ingroup tutorial_hist
/// Fill multiple histograms with different functions and automatic binning.
/// Illustrates merging with the power-of-two autobin algorithm
///
/// \macro_output
/// \macro_code
///
/// \date November 2017
/// \author Gerardo Ganis

#include "TF1.h"
#include "TH1D.h"
#include "TMath.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TRandom3.h"
#include "TStatistic.h"
#include "TFile.h"
#include "TStyle.h"

TF1 *gam = new TF1("gam", "1/(1+0.1*x*0.1*x)", -100., 100.);
TF1 *gam1 = new TF1("gam", "1/(1+0.1*x*0.1*x)", -1., .25);
TF1 *iga = new TF1("inv gam", "1.-1/(1+0.1*x*0.1*x)", -100., 100.);
TF1 *iga1 = new TF1("inv gam", "1.-1/(1+0.1*x*0.1*x)", -.5, 1.);

void fillhistosauto2p(unsigned opt = 1, unsigned n = 1001)
{

   UInt_t nh = 10;
   UInt_t bsize = 1000;

   TRandom3 rndm((Long64_t)time(0));

   // Standard autobinning reference
   auto href = new TH1D("myhref", "current", 50, 0., -1.);
   href->SetBuffer(bsize);

   // New autobinning 1-histo reference
   auto href2 = new TH1D("myhref", "Auto P2, sequential", 50, 0., -1.);
   href2->SetBit(TH1::kAutoBinPTwo);
   href2->SetBuffer(bsize);

   TList *hlist = new TList;

   Int_t nbins = 50;

   TStatistic x("min"), y("max"), d("dif"), a("mean"), r("rms");
   for (UInt_t j = 0; j < nh; ++j) {
      Double_t xmi = 1e15, xma = -1e15;
      TStatistic xw("work");
      TString hname = TString::Format("myh%d", j);
      auto hw = new TH1D(hname.Data(), "Auto P2, merged", nbins, 0., -1.);
      hw->SetBit(TH1::kAutoBinPTwo);
      hw->SetBuffer(bsize);

      Double_t xhma, xhmi, ovf, unf;
      Bool_t emptied = kFALSE, tofill = kTRUE;
      Bool_t buffering = kTRUE;
      for (UInt_t i = 0; i < n; ++i) {

         Double_t xx;
         switch (opt) {
         case 1: xx = rndm.Gaus(3, 1); break;
         case 2: xx = rndm.Rndm() * 100. - 50.; break;
         case 3: xx = gam->GetRandom(); break;
         case 4: xx = gam1->GetRandom(); break;
         case 5: xx = iga->GetRandom(); break;
         case 6: xx = iga1->GetRandom(); break;
         default: xx = rndm.Gaus(0, 1);
         }

         if (buffering) {
            if (xx > xma)
               xma = xx;
            if (xx < xmi)
               xmi = xx;
            xw.Fill(xx);
         }
         hw->Fill(xx);
         href->Fill(xx);
         href2->Fill(xx);
         if (!hw->GetBuffer()) {
            // Not buffering anymore
            buffering = kFALSE;
         }
      }
      x.Fill(xmi);
      y.Fill(xma);
      d.Fill(xma - xmi);
      a.Fill(xw.GetMean());
      r.Fill(xw.GetRMS());

      hlist->Add(hw);
   }

   x.Print();
   y.Print();
   d.Print();
   a.Print();
   r.Print();

   TH1D *h0 = (TH1D *)hlist->First();
   hlist->Remove(h0);
   if (!h0->Merge(hlist))
      return;

   gStyle->SetOptStat(111110);

   if (gROOT->GetListOfCanvases()->FindObject("c3"))
      delete gROOT->GetListOfCanvases()->FindObject("c3");
   TCanvas *c3 = new TCanvas("c3", "c3", 800, 800);
   c3->Divide(1, 3);
   c3->cd(1);
   h0->StatOverflows();
   h0->DrawClone("HIST");

   c3->cd(2);
   href2->StatOverflows();
   href2->DrawClone();

   c3->cd(3);
   href->StatOverflows();
   href->DrawClone();
   c3->Update();
   std::cout << " ent: " << h0->GetEntries() << "\n";
   h0->Print();
   href->Print();

   hlist->SetOwner(kTRUE);
   delete hlist;
   delete href;
   delete href2;
   delete h0;
}
