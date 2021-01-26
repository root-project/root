#include "gtest/gtest.h"


#include "TKDE.h"
#include "TString.h"
#include "TFile.h"
#include "TRandom3.h"
#include "TVirtualPad.h"
#include "TF1.h"
#include "TH1.h"

struct  TestKDE  {

   // using n < 10000 use a default nbins=100 for KDE
   int n = 10000;
   bool useSetters = false;
   bool binned = false;
   bool adaptive = false;
   bool makePlot = true;
   bool debug = false;
   int nbins = 100;
   int seed = 1111;

   std::vector<double> data;  // kde data

   std::vector<double> xtest;

   int nTestPoints = 21;
   double values1[21];
   double values2[21];

   double pval;
   double pval2;

   TKDE * Create() {
      TRandom3 r(seed);
      data.resize(n);
      for(int i=0;i<n;++i) data[i] = (i < 0.2*n) ? r.Gaus(10,1) : r.Gaus(10,4);
      const char *iterationType = (adaptive) ? "Adaptive" : "Fixed";
      const char *binningType = (binned) ? "ForcedBinning" : "Unbinned";
      TString opt;
      if (useSetters)
         opt = "KernelType:Gaussian;Iteration:Fixed;Mirror:noMirror;Binning:Unbinned";
      else
         opt = TString::Format("KernelType:Gaussian;Iteration:%s;Mirror:noMirror;Binning:%s",iterationType, binningType);

      assert((int)data.size() == n);
      TKDE *kde = new TKDE(n, data.data(), 0., 20., opt, 1);
      if (useSetters) {
         if (adaptive) {
            kde->SetIteration(TKDE::kAdaptive);
         }
         if (binned) {
            // use relaxed binned mode
            kde->SetBinning(TKDE::kRelaxedBinning);
            kde->SetUseBinsNEvents(100);
            kde->SetNBins(nbins);  // when nbins=100 (using setters or not should give same results)
         }
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
      int nhistbins = 100;
      auto h1 = new TH1D("h1", "h1", nhistbins, 0., 20.);
      for(int i=0;i<n;++i) h1->Fill( data[i] );

      h1->Sumw2();
      h1->Scale(1./h1->Integral(), "width");

      auto fkde = kde->GetFunction(nhistbins);

      if (makePlot) {
         //auto c1 = new TCanvas();
         kde->Draw();
         h1->SetLineColor(kBlue);
         h1->Draw("SAME");
         TString fname = name + "_hist.pdf";
         if (gPad) gPad->SaveAs(fname);
      }


      // make chi2 test
      //double pvalChi2  = h1->Chi2Test(h2,"WW P");
      // note Chisquare called SetParameters and delete histogram returned from the function
      double chi2 = h1->Chisquare(fkde);
      double pvalChi2 = TMath::Prob(chi2, nhistbins);

      auto h2 = fkde->GetHistogram();
      h2->Sumw2();
      // set correct bin error
      for (int i = 1; i <= h2->GetNbinsX() + 1; ++i)
         h2->SetBinError(i, sqrt(h2->GetBinContent(i) / (n * h2->GetBinWidth(i))));

      // compute KS test
      double pvalKS = h1->KolmogorovTest(h2);
      std::cout << "CompareWithHist:   Chi2 test = " << chi2 << "/" << nhistbins << " pvalue = " << pvalChi2 <<
                   "  KS test p value = " << pvalKS << std::endl;
      pval = std::min(pvalKS, pvalChi2);
      pval2 = std::max(pvalKS, pvalChi2);

      delete kde;
      delete h1;
   }

   static bool IsPValid(double pval) {
      return pval > 0.01;
   }
   static bool IsPValid2(double pval1, double pval2){
      return pval1 > 1.E-4 && pval2 > 0.05;
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
/// filled with the same type of data. A GoF test (chi2 and KS) is applied to
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
   // reduce n here since unbinned adaptive is slow
   t.n = 2000;
   t.adaptive = true;
   t.CompareWithHist("tkde_adaptive");
   EXPECT_PRED1(TestKDE::IsPValid, t.pval);
   // test also default constructors + usage of setters functions
   t.useSetters = true;
   t.CompareWithHist("tkde_adaptive_2");
   EXPECT_PRED1(TestKDE::IsPValid, t.pval);
}
TEST(TKDE, tkde_hist_binned)
{
   TestKDE t;
   t.adaptive = false;
   t.binned = true;
   t.nbins = 3000;
   // with binned not adaptive it is good for high statistics
   t.CompareWithHist("tkde_binned");
   EXPECT_PRED2(TestKDE::IsPValid2, t.pval, t.pval2);
   t.useSetters = true;
   t.CompareWithHist("tkde_binned_2");
   EXPECT_PRED2(TestKDE::IsPValid2, t.pval, t.pval2);
}
TEST(TKDE, tkde_hist_adaptive_binned)
{
   TestKDE t;
   t.nbins = t.n/10;
   //t.nbins = 3000;
   t.adaptive = true;
   t.binned = true;
   t.CompareWithHist("tkde_adaptive_binned");
   EXPECT_PRED1(TestKDE::IsPValid, t.pval);
   t.useSetters = true;
   t.CompareWithHist("tkde_adaptive_binned_2");
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
   t.n = 2000;
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
