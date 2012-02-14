//*********************************************************************
//+ Evaluate the performance of THnSparse vs TH1/2/3/nF
//  for different numbers of dimensions and bins per dimension.
// 
//  The script calculates the bandwidth for filling and retrieving
//  bin contents (in million entries per second) for these two
//  histogramming techniques, where "seconds" is CPU and real time.
//
//  The first line of the plots contains the bandwidth based on the 
//  CPU time (THnSpase, TH1/2/3/nF*, ratio), the second line shows
//  the plots for real time, and the third line shows the fraction of
//  filled bins and memory used by THnSparse vs. TH1/2/3/nF.
// 
//  The timing depends on the distribution and the amount of entries
//  in the histograms; here, a Gaussian distribution (center is
//  contained in the histograms) is used to fill each histogram with
//  1000 entries. The filling and reading is repeated until enough
//  statistics have been collected.
// 
//  tutorials/tree/drawsparse.C shows an example for visualizing a
//  THnSparse. It creates a TTree which is then drawn using
//  TParallelCoord.
// 
//  This macro should be run in compiled mode due to the many nested
//  loops that force CINT to disable its optimization. If run
//  interpreted one would not benchmark THnSparse but CINT.
// 
//  Run as
//    .L $ROOTSYS/tutorials/hist/sparsehist.C+
// 
//  Axel.Naumann@cern.ch (2007-09-14)
// *********************************************************************

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"
#include "TStopwatch.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"
#include "TSystem.h"

class TTimeHists {
public:
   enum EHist { kHist, kSparse, kNumHist };
   enum ETime { kReal, kCPU, kNumTime };
   TTimeHists(Int_t dim, Int_t bins, Long_t num):
      fValue(0), fDim(dim), fBins(bins), fNum(num),
      fSparse(0), fHist(0), fHn(0) {}
   ~TTimeHists();
   bool Run();
   Double_t GetTime(EHist hist, ETime time) const {
      if (time == kReal) return fTime[hist][0];
      return fTime[hist][1]; }
   static void SetDebug(Int_t lvl) { fgDebug = lvl; }
   THnSparse* GetSparse() const { return fSparse; }

protected:
   void Fill(EHist hist);
   Double_t Check(EHist hist);
   void SetupHist(EHist hist);
   void NextValues();
   void SetupValues();

private:
   Double_t* fValue;
   Int_t fDim;
   Int_t fBins;
   Long_t fNum;
   Double_t fTime[2][2];
   THnSparse* fSparse;
   TH1*       fHist;
   THn*       fHn;
   static Int_t fgDebug;
};

Int_t TTimeHists::fgDebug = 0;

TTimeHists::~TTimeHists()
{
   delete [] fValue;
   delete fSparse;
   delete fHist;
   delete fHn;
}

bool TTimeHists::Run()
{
   // run all tests with current settings, and check for identity of content.

   Double_t check[2];
   Long64_t rep[2];
   for (int h = 0; h < 2; ++h) {
      rep[h] = 0;
      SetupValues();
      try {
         TStopwatch w;
         w.Start();
         SetupHist((EHist) h);
         w.Stop();
         do {
            w.Start(kFALSE);
            Fill((EHist) h);
            check[h] = Check((EHist) h);
            w.Stop();
            ++rep[h];
         } while ((!h && w.RealTime() < 0.1)
            || (h && rep[0] > 0 && rep[1] < rep[0]));

         fTime[h][0] = (1.* fNum * rep[h]) / w.RealTime() / 1E6;
         fTime[h][1] = (1.* fNum * rep[h]) / w.CpuTime() / 1E6;

         if (h == 1 && (fTime[h][0] > 1E20 || fTime[h][1] > 1E20)) {
            do {
               // some more cycles:
               w.Start(kFALSE);
               Fill((EHist) h);
               Check((EHist) h);
               w.Stop();
               ++rep[h];
            } while (w.RealTime() < 0.1);

            fTime[h][0] = (1.* fNum * rep[h]) / w.RealTime() / 1E6;
            fTime[h][1] = (1.* fNum * rep[h]) / w.CpuTime() / 1E6;
         }

         if (fTime[h][0] > 1E20) fTime[h][0] = 1E20;
         if (fTime[h][1] > 1E20) fTime[h][1] = 1E20;
      }
      catch (std::exception&) {
         fTime[h][0] = fTime[h][1] = -1.;
         check[h] = -1.; // can never be < 1 without exception
         rep[h] = -1;
      }
   }
   if (check[0] != check[1])
      if (check[0] != -1.)
         printf("ERROR: mismatch of histogram (%g) and sparse histogram (%g) for dim=%d, bins=%d!\n",
                check[0], check[1], fDim, fBins);
      // else
      //   printf("ERROR: cannot allocate histogram for dim=%d, bins=%d - out of memory!\n",
      //          fDim, fBins);
   return (check[0] == check[1]);
}

void TTimeHists::NextValues()
{
   for (Int_t d = 0; d < fDim; ++d)
      fValue[d] = gRandom->Gaus() / 4.;
}

void TTimeHists::SetupValues()
{
   // define fValue
   if (!fValue) fValue = new Double_t[fDim];
   gRandom->SetSeed(42);
}

void TTimeHists::Fill(EHist hist)
{
   for (Long_t n = 0; n < fNum; ++n) {
      NextValues();
      if (fgDebug > 1) {
         printf("%ld: fill %s", n, hist == kHist? (fDim < 4 ? "hist" : "arr") : "sparse");
         for (Int_t d = 0; d < fDim; ++d)
            printf("[%g]", fValue[d]);
         printf("\n");
      }
      if (hist == kHist) {
         switch (fDim) {
         case 1: fHist->Fill(fValue[0]); break;
         case 2: ((TH2F*)fHist)->Fill(fValue[0], fValue[1]); break;
         case 3: ((TH3F*)fHist)->Fill(fValue[0], fValue[1], fValue[2]); break;
         default: fHn->Fill(fValue); break;
         }
      } else {
         fSparse->Fill(fValue);
      }
   }
}

void TTimeHists::SetupHist(EHist hist)
{
   if (hist == kHist) {
      switch (fDim) {
      case 1: fHist = new TH1F("h1", "h1", fBins, -1., 1.); break;
      case 2: fHist = new TH2F("h2", "h2", fBins, -1., 1., fBins, -1., 1.); break;
      case 3: fHist = new TH3F("h3", "h3", fBins, -1., 1., fBins, -1., 1., fBins, -1., 1.); break;
      default:
         {
            MemInfo_t meminfo;
            gSystem->GetMemInfo(&meminfo);
            Int_t size = 1;
            for (Int_t d = 0; d < fDim; ++d) {
               if ((Int_t)(size * sizeof(Float_t)) > INT_MAX / (fBins + 2)
                  || (meminfo.fMemFree > 0 
                  && meminfo.fMemFree / 2 < (Int_t) (size * sizeof(Float_t)/1000/1000)))
                  throw std::bad_alloc();
               size *= (fBins + 2);
            }
            if (meminfo.fMemFree > 0 
               && meminfo.fMemFree / 2 < (Int_t) (size * sizeof(Float_t)/1000/1000))
               throw std::bad_alloc();
            Int_t* bins = new Int_t[fDim];
            Double_t *xmin = new Double_t[fDim];
            Double_t *xmax = new Double_t[fDim];
            for (Int_t d = 0; d < fDim; ++d) {
               bins[d] = fBins;
               xmin[d] = -1.;
               xmax[d] =  1.;
            }
            fHn = new THnF("hn", "hn", fDim, bins, xmin, xmax);
         }
      }
   } else {
      Int_t* bins = new Int_t[fDim];
      Double_t *xmin = new Double_t[fDim];
      Double_t *xmax = new Double_t[fDim];
      for (Int_t d = 0; d < fDim; ++d) {
         bins[d] = fBins;
         xmin[d] = -1.;
         xmax[d] =  1.;
      }
      fSparse = new THnSparseF("hs", "hs", fDim, bins, xmin, xmax);
   }
}

Double_t TTimeHists::Check(EHist hist)
{
   // Check bin content of all bins
   Double_t check = 0.;
   Int_t* x = new Int_t[fDim];
   memset(x, 0, sizeof(Int_t) * fDim);

   if (hist == kHist) {
      Long_t idx = 0;
      Long_t size = 1;
      for (Int_t d = 0; d < fDim; ++d)
         size *= (fBins + 2);
      while (x[0] <= fBins + 1) {
         Double_t v = -1.;
         if (fDim < 4) {
            Long_t histidx = x[0];
            if (fDim == 2) histidx = fHist->GetBin(x[0], x[1]);
            else if (fDim == 3) histidx = fHist->GetBin(x[0], x[1], x[2]);
            v = fHist->GetBinContent(histidx);
         }
         else v = fHn->GetBinContent(x);
         Double_t checkx = 0.;
         if (v)
            for (Int_t d = 0; d < fDim; ++d)
               checkx += x[d];
         check += checkx * v;

         if (fgDebug > 2 || (fgDebug > 1 && v)) {
            printf("%s%d", fDim < 4 ? "hist" : "arr", fDim);
            for (Int_t d = 0; d < fDim; ++d)
               printf("[%d]", x[d]);
            printf(" = %g\n", v);
         }

         ++x[fDim - 1];
         // Adjust the bin idx
         // no wrapping for dim 0 - it's what we break on!
         for (Int_t d = fDim - 1; d > 0; --d) {
            if (x[d] > fBins + 1) {
               x[d] = 0;
               ++x[d - 1];
            }
         }
         ++idx;
      } // while next bin
   } else {
      for (Long64_t i = 0; i < fSparse->GetNbins(); ++i) {
         Double_t v = fSparse->GetBinContent(i, x);
         Double_t checkx = 0.;
         for (Int_t d = 0; d < fDim; ++d)
            checkx += x[d];
         check += checkx * v;

         if (fgDebug > 1) {
            printf("sparse%d", fDim);
            for (Int_t d = 0; d < fDim; ++d)
               printf("[%d]", x[d]);
            printf(" = %g\n", v);
         }
      }
   }
   check /= fNum;
   if (fgDebug > 0)
      printf("check %s%d = %g\n", hist == kHist ? (fDim < 4 ? "hist" : "arr") : "sparse", fDim, check);
   return check;
}


void sparsehist() {
#ifdef __CINT__
   printf("Please run this script in compiled mode by running \".x sparsehist.C+\"\n");
   return;
#endif

   TH2F* htime[TTimeHists::kNumHist][TTimeHists::kNumTime];
   for (int h = 0; h < TTimeHists::kNumHist; ++h)
      for (int t = 0; t < TTimeHists::kNumTime; ++t) {
         TString name("htime_");
         if (h == 0) name += "arr";
         else name += "sp";
         if (t == 0) name += "_r";

         TString title;
         title.Form("Throughput (fill,get) %s (%s, 1M entries/sec);dim;bins;1M entries/sec", h == 0 ? "TH1/2/3/nF" : "THnSparseF", t == 0 ? "real" : "CPU");
         htime[h][t] = new TH2F(name, title, 6, 0.5, 6.5, 10, 5, 105);
      }

   TH2F* hsparse_mem = new TH2F("hsparse_mem", "Fractional memory usage;dim;bins;fraction of memory used", 6, 0.5, 6.5, 10, 5, 105);
   TH2F* hsparse_bins = new TH2F("hsparse_bins", "Fractional number of used bins;dim;bins;fraction of filled bins", 6, 0.5, 6.5, 10, 5, 105);

   // TTimeHists::SetDebug(2);
   Double_t max = -1.;
   for (Int_t dim = 1; dim < 7; ++dim) {
      printf("Processing dimension %d", dim);
      for (Int_t bins = 10; bins <= 100; bins += 10) {
         TTimeHists timer(dim, bins, /*num*/ 1000);
         timer.Run();
         for (int h = 0; h < TTimeHists::kNumHist; ++h)
            for (int t = 0; t < TTimeHists::kNumTime; ++t) {
               Double_t time = timer.GetTime((TTimeHists::EHist)h, (TTimeHists::ETime)t);
               if (time >= 0.)
                  htime[h][t]->Fill(dim, bins, time);
            }

         hsparse_mem->Fill(dim, bins, timer.GetSparse()->GetSparseFractionMem());
         hsparse_bins->Fill(dim, bins, timer.GetSparse()->GetSparseFractionBins());

         if (max < timer.GetTime(TTimeHists::kSparse, TTimeHists::kReal))
            max = timer.GetTime(TTimeHists::kSparse, TTimeHists::kReal);
         printf(".");
         fflush(stdout);
      }
      printf(" done\n");
   }

   Double_t markersize = 2.5;
   hsparse_mem->SetMarkerSize(markersize);
   hsparse_bins->SetMarkerSize(markersize);

   TH2F* htime_ratio[TTimeHists::kNumTime];
   for (int t = 0; t < TTimeHists::kNumTime; ++t) {
      const char* name = t ? "htime_ratio" : "htime_ratio_r";
      htime_ratio[t] = (TH2F*) htime[TTimeHists::kSparse][t]->Clone(name);
      TString title;
      title.Form("Relative speed improvement (%s, 1M entries/sec): sparse/hist;dim;bins;#Delta 1M entries/sec", t == 0 ? "real" : "CPU");
      htime_ratio[t]->SetTitle(title);
      htime_ratio[t]->Divide(htime[TTimeHists::kHist][t]);
      htime_ratio[t]->SetMinimum(0.1);
      htime_ratio[t]->SetMarkerSize(markersize);
   }

   TFile* f = new TFile("sparsehist.root","RECREATE");

   TCanvas* canv= new TCanvas("c","c");
   canv->Divide(3,3);

   gStyle->SetPalette(8,0);
   gStyle->SetPaintTextFormat(".2g");
   gStyle->SetOptStat(0);
   const char* opt = "TEXT COL";

   for (int t = 0; t < TTimeHists::kNumTime; ++t) {
      for (int h = 0; h < TTimeHists::kNumHist; ++h) {
         htime[h][t]->SetMaximum(max);
         htime[h][t]->SetMarkerSize(markersize);
         canv->cd(1 + h + 3 * t);
         htime[h][t]->Draw(opt);
         htime[h][t]->Write();
      }
      canv->cd(3 + t * 3);
      htime_ratio[t]->Draw(opt); gPad->SetLogz();
      htime_ratio[t]->Write();
   }

   canv->cd(7); hsparse_mem->Draw(opt);
   canv->cd(8); hsparse_bins->Draw(opt);
   hsparse_mem->Write();
   hsparse_bins->Write();

   canv->Write();

   delete f;
}
