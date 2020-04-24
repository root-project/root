#include "gtest/gtest.h"


#include "TKDE.h"
#include "TString.h"
#include "TFile.h"
#include "TRandom.h"
#include "TVirtualPad.h"
#include "TF1.h"
#include "TH1.h"

struct  TestKDE  {

   int n = 1000;
   bool binned = false;
   bool adaptive = false;
   bool makePlot = true;
   bool debug = false; 
   int nbins = 40;

   std::vector<double> xtest;
   int nTestPoints = 21;
   double values1[21];
   double values2[21];

   double pval;

   TKDE * Create() {
      std::vector<double> v; v.reserve(n);
      for(int i=0;i<n;++i) v.push_back( (i < 0.2*n) ? gRandom->Gaus(10,1) : gRandom->Gaus(10,4) );
      TString opt = "KernelType:Gaussian;Iteration:Fixed;Mirror:noMirror;Binning:Unbinned";
      TKDE *kde = new TKDE(v.size(), &v[0], 0., 20., opt, 1);
      if (adaptive) {
         kde->SetIteration(TKDE::kAdaptive);
      }
      if (binned) {
         kde->SetBinning(TKDE::kRelaxedBinning);
         kde->SetUseBinsNEvents(100);
         kde->SetNBins(20);
      }

      xtest.resize(20);
      TAxis a(nTestPoints,-0.5,20.5);
      for (size_t i = 0; i < xtest.size(); ++i) {
         xtest[i] = a.GetBinCenter(i+1);
         values1[i] = (*kde)(xtest[i] );
      }
      if (debug) { 
         std::cout << "Before writing,  kde values :  ";
         for (size_t i = 0; i < xtest.size(); ++i) {
            std::cout << values1[i];
            if (i <  xtest.size()-1) std::cout << " , ";
            else std::cout << std::endl;
         }
      }

      if (makePlot) {
         //auto c1 = new TCanvas();
         kde->Draw();
         kde->GetDrawnFunction()->SetLineColor(kBlue);
      }
      return kde;
   }

   void CompareWithHist(TString name = "tkde") {
      auto kde = Create();
      int nhistbins = n/20;
      auto h1 = new TH1D("h1","h1",nhistbins,0.,20.);
      for(int i=0;i<n;++i) h1->Fill( (i < 0.2*n) ? gRandom->Gaus(10,1) : gRandom->Gaus(10,4) );

      h1->Sumw2();
      h1->Scale(1./h1->Integral(), "width");
      
      auto h2 = kde->GetFunction(nhistbins)->GetHistogram();
      h2->Sumw2();
      // set correct bin error
      for (int i = 1; i <= h2->GetNbinsX()+1; ++i)
         h2->SetBinError(i, sqrt( h2->GetBinContent(i)/( n * h2->GetBinWidth(i) ) ) );

      if (makePlot) {
         //auto c1 = new TCanvas();
         kde->Draw();
         h1->SetLineColor(kBlue); 
         h1->Draw("SAME");
         TString fname = name + "_hist.pdf";
         if (gPad) gPad->SaveAs(fname); 
      }


      // make chi2 test
      double pvalChi  = h1->Chi2Test(h2,"WW P"); 
      double pvalKS =  h1->KolmogorovTest(h2);
      std::cout << "CompareWithHist:   Chi2 test = " << pvalChi << " KS test = " << pvalKS << std::endl;
      pval = std::min(pvalKS, pvalChi);

      delete kde;
      delete h1; 
   }

   static bool IsPValid(double pval) {
      return (pval > 0.01);
   }


   void Write(TString name = "tkde") {
      TFile *f = new TFile("test_tkde.root", "RECREATE");

      auto kde = Create();

      kde->Write(name);
      f->Close();
   }

   void Read(TString name = "tkde") {
      TFile *f = new TFile("test_tkde.root");
      auto kde = (TKDE*) f->Get(name); 
      //kde->Dump();
      for (size_t i = 0; i < xtest.size(); ++i) values2[i] = (*kde)(xtest[i] );

      if (debug) { 
         std::cout << "After reading,  kde values : ";
         for (size_t i = 0; i < xtest.size(); ++i) {
            std::cout << values2[i];
            if (i <  xtest.size()-1) std::cout << " , ";
            else std::cout << std::endl;
         }
      }

      if (makePlot) { 
         kde->Draw("SAME");
         TString gfname = name + ".pdf"; 
         gPad->SaveAs(gfname);
      }
      //f->Close();
      delete f;
   }
   
};

/// Hstogram comparison tests
/// In this test we compare the TKDE with an histogram
/// filled with th same type of data. A GoF test (chi2 and KS) is applied to
/// check for compatibility
TEST(TKDE, tkde_hist)
{   
   TestKDE t;
   t.CompareWithHist(); 
   EXPECT_PRED1(TestKDE::IsPValid, t.pval);
}
TEST(TKDE, tkde_hist_adaptive)
{   
   TestKDE t;
   t.adaptive = true; 
   t.CompareWithHist("tkde_adaptive"); 
   EXPECT_PRED1(TestKDE::IsPValid, t.pval);
}
TEST(TKDE, tkde_hist_binned)
{  
   TestKDE t;
   t.adaptive = true; 
   t.binned = true; 
   t.CompareWithHist("tkde_binned"); 
   EXPECT_PRED1(TestKDE::IsPValid, t.pval);
}

/// IO tests
/// In this test we compare the value before writing and after reading of the TKDE
TEST(TKDE, tkde_io)
{
   TestKDE t;
   t.Write();
   t.Read();

   // double delta = 1.E-15; 
   // EXPECT_NEAR(t.values1[0], t.values2[0], delta);

   
   for (size_t i = 0; i < t.xtest.size(); ++i) {
      EXPECT_DOUBLE_EQ(t.values1[i], t.values2[i]);
   }
}

TEST(TKDE, tkde_io_adaptive)
{
   TString name = "tkde_adaptive";
   TestKDE t;
   t.adaptive = true;
   t.Write(name);
   t.Read(name);
   double delta = 1.E-3;
   for (size_t i = 0; i < t.xtest.size(); ++i) {
      EXPECT_NEAR(t.values1[i], t.values2[i], delta);
      //EXPECT_DOUBLE_EQ(t.values1[i], t.values2[i]);
   }
}

TEST(TKDE, tkde_io_binned)
{
   TestKDE t;
   t.adaptive = true; 
   t.binned = true;
   TString name = "tkde_binned";
   t.Write(name);
   t.Read(name);
   double delta = 1.E-3;
   for (size_t i = 0; i < t.xtest.size(); ++i) {
      EXPECT_NEAR(t.values1[i], t.values2[i], delta);
      //EXPECT_DOUBLE_EQ(t.values1[i], t.values2[i]);
   }
}

