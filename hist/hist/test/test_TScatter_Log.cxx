// Unit tests for TScatter and TScatter2D log scale rendering (Issue #23833)

#include "gtest/gtest.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TScatter.h"
#include "TScatter2D.h"

TEST(TScatterLog, NonPositivePointsInLogScale)
{
   gROOT->SetBatch(true);
   auto c = new TCanvas("c_scatter_log", "c_scatter_log", 800, 600);
   c->SetLogy();

   // 5 points where index 2 has non-positive y = 0
   const int n = 5;
   double x[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
   double y[n] = {10.0, 20.0, 0.0, 40.0, 50.0};
   double col[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
   double size[n] = {2.0, 2.0, 2.0, 2.0, 2.0};

   auto scatter = new TScatter(n, x, y, col, size);
   scatter->Draw("A");
   c->Update();

   EXPECT_NE(scatter->GetHistogram(), nullptr);
   delete c;
}

TEST(TScatter2DLog, NonPositivePointsInLogScale)
{
   gROOT->SetBatch(true);
   auto c = new TCanvas("c_scatter2d_log", "c_scatter2d_log", 800, 600);
   c->SetLogz();

   // 5 points where index 2 has non-positive z = 0
   const int n = 5;
   double x[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
   double y[n] = {10.0, 20.0, 30.0, 40.0, 50.0};
   double z[n] = {1.0, 2.0, 0.0, 4.0, 5.0};
   double col[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
   double size[n] = {2.0, 2.0, 2.0, 2.0, 2.0};

   auto scatter2d = new TScatter2D(n, x, y, z, col, size);
   scatter2d->Draw();
   c->Update();

   EXPECT_NE(scatter2d->GetHistogram(), nullptr);
   delete c;
}
