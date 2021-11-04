// test TMultiGraph::GetHistogram in log scale

#include "gtest/gtest.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TH1.h"

TEST(TMultiGraph, GetHistogram)
{
   gROOT->SetBatch(1);
   auto c = new TCanvas();
   c->SetLogy();

   std::vector<double> x1;
   std::vector<double> sig1;
   std::vector<double> sig2;
   for (double E=1e-4;E<2e7;E*=1.1) {
      x1.push_back(E);
      sig1.push_back(10*pow(E,-0.1));
      sig2.push_back(15*pow(E,-0.15));
   }

   auto mg = new TMultiGraph();
   auto g1 = new TGraph(x1.size(), x1.data(), sig1.data()); mg->Add(g1);
   auto g2 = new TGraph(x1.size(), x1.data(), sig2.data()); mg->Add(g2);
   auto h  = mg->GetHistogram();

   double delta = 1.E-3;
   EXPECT_NEAR(h->GetMinimum(), 0.65234, delta);
   EXPECT_NEAR(h->GetMaximum(), 79.9602, delta);
}