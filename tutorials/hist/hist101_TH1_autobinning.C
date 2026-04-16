/// \file
/// \ingroup tutorial_hist
/// \preview Fill multiple histograms with different functions and automatic binning.
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

enum class EHist101_Func {
   kGaus = 1,
   kLinear = 2,
   kGamma = 3,
   kGamma1 = 4,
   kInvGamma = 5,
   kInvGamma1 = 6
};

void hist101_TH1_autobinning(EHist101_Func function = EHist101_Func::kGaus, unsigned nEntriesPerHisto = 1001)
{
   const Int_t nbins = 50;

   TRandom3 rndm((Long64_t)time(0));

   TH1D::SetDefaultBufferSize(1000);

   // Create a histogram with `nbins` bins and range [0, -1].
   // When a histogram is created with an upper limit lower or equal to its lower limit, it will automatically
   // compute the axis limits.
   // The binning is decided as soon as the histogram's internal entry buffer is filled, i.e. when you fill in a number
   // of entries equal to the buffer size.
   // The default buffer size is determined by TH1::SetDefaultBufferSize, or you can customize each individual histogram's
   // buffer by calling TH1D::SetBuffer(int bufferSize).
   auto href = std::make_unique<TH1D>("myhref", "current", nbins, 0., -1.);

   auto href2 = std::make_unique<TH1D>("myhref", "Auto P2, sequential", nbins, 0., -1.);
   // If you want to enable power-of-2 auto binning, call this:
   href2->SetBit(TH1::kAutoBinPTwo);

   // list to hold all histograms we're going to create
   TList histoList;
   // tell the list it should delete its elements upon destruction.
   histoList.SetOwner(true);

   TStatistic x("min"), y("max"), d("dif"), a("mean"), r("rms");

   // Fill a bunch of histograms with the selected function
   for (int j = 0; j < 10; ++j) {
      double xmi = 1e15, xma = -1e15;
      TStatistic xw("work");
      const std::string hname = "myh" + std::to_string(j);

      // Create more auto-binning histograms and add them to the list
      auto hw = new TH1D(hname.c_str(), "Auto P2, merged", nbins, 0., -1.);
      hw->SetBit(TH1::kAutoBinPTwo);

      bool buffering = true;
      for (UInt_t i = 0; i < nEntriesPerHisto; ++i) {
         double xx = 0;
         // clang-format off
         switch (function) {
         case EHist101_Func::kGaus:      xx = rndm.Gaus(3, 1);          break;
         case EHist101_Func::kLinear:    xx = rndm.Rndm() * 100. - 50.; break;
         case EHist101_Func::kGamma:     xx = gam->GetRandom();         break;
         case EHist101_Func::kGamma1:    xx = gam1->GetRandom();        break;
         case EHist101_Func::kInvGamma:  xx = iga->GetRandom();         break;
         case EHist101_Func::kInvGamma1: xx = iga1->GetRandom();        break;
         default:                        xx = rndm.Gaus(0, 1);
         }
         // clang-format on

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
            // We exhausted the histogram's buffer
            buffering = false;
         }
      }
      x.Fill(xmi);
      y.Fill(xma);
      d.Fill(xma - xmi);
      a.Fill(xw.GetMean());
      r.Fill(xw.GetRMS());

      histoList.Add(hw);
   }

   x.Print();
   y.Print();
   d.Print();
   a.Print();
   r.Print();

   // Merge all histograms into one
   auto h0 = std::unique_ptr<TH1D>(static_cast<TH1D *>(histoList.First()));
   histoList.Remove(h0.get());
   if (!h0->Merge(&histoList))
      return;

   // Set what we want to display in the histogram stat box
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
}
