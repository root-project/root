#include "gtest/gtest.h"

#include "THn.h"
#include "THnSparse.h"
#include "TH1.h"
#include "TH2.h"

// Filling THn
TEST(THn, Fill) {
   Int_t bins[2] = {2, 3};
   Double_t xmin[2] = {0., -3.};
   Double_t xmax[2] = {10., 3.};
   THnD hn("hn", "hn", 2, bins, xmin, xmax);

   {
      // Non-overflow
      Double_t x0[2]{4., -0.01};
      EXPECT_EQ(7, hn.GetBin(x0));
      EXPECT_DOUBLE_EQ(0.0, hn.GetBinContent(7));

      EXPECT_EQ(7, hn.Fill(x0, 0.42));
      EXPECT_DOUBLE_EQ(0.42, hn.GetBinContent(7));
   }

   {
      Double_t x0[2]{-0.49, -2.90};
      EXPECT_EQ(1, hn.GetBin(x0));
      EXPECT_DOUBLE_EQ(0., hn.GetBinContent(10));

      EXPECT_EQ(1, hn.Fill(x0, 0.17));
      EXPECT_EQ(1, hn.GetBin(x0));
      EXPECT_DOUBLE_EQ(0.17, hn.GetBinContent(1));
   }

}


TEST(THn, Projection) {
   Int_t bins[2] = {2, 3};
   Double_t xmin[2] = {0., -3.};
   Double_t xmax[2] = {10., 3.};
   THnD hn("hn", "hn", 2, bins, xmin, xmax);

   {
      // Non-overflow
      Double_t x0[2]{4., -0.01};
      hn.Fill(x0, 0.42);
   }

   {
      // Overflow x
      Double_t x0[2]{-0.49, -2.90};
      hn.Fill(x0, 0.17);
   }

   {
      TH1D* hProj = hn.Projection(0);
      EXPECT_EQ(2, hProj->GetNbinsX());
      EXPECT_DOUBLE_EQ(0.17, hProj->GetBinContent(0));
      EXPECT_DOUBLE_EQ(0.42, hProj->GetBinContent(1));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(2));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(3));
      delete hProj;
   }

   {
      TH1D* hProj = hn.Projection(1);
      EXPECT_EQ(3, hProj->GetNbinsX());
      EXPECT_DOUBLE_EQ(0.0, hProj->GetBinContent(0));
      EXPECT_DOUBLE_EQ(0.17, hProj->GetBinContent(1));
      EXPECT_DOUBLE_EQ(0.42, hProj->GetBinContent(2));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(3));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(4));
      delete hProj;
   }

   {
      TH2D* hProj = hn.Projection(0, 1);
      // Yes, x|y are intentionally inverted...
      EXPECT_EQ(3, hProj->GetNbinsX());
      EXPECT_EQ(2, hProj->GetNbinsY());
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(0));
      EXPECT_DOUBLE_EQ(.17, hProj->GetBinContent(1));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(2));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(3));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(4));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(5));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(6));
      EXPECT_DOUBLE_EQ(.42, hProj->GetBinContent(7));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(8));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(9));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(10));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(11));
      delete hProj;
   }

   {
      // Overflow x and y
      Double_t x0[2]{-0.49, 3.90};
      hn.Fill(x0, 43.);

      // With axis range, excluding above point; see ROOT-8457
      hn.GetAxis(1)->SetRange(1,2);
      TH1D* hProj = hn.Projection(1);
      EXPECT_EQ(2, hProj->GetNbinsX());
      EXPECT_DOUBLE_EQ(0.0, hProj->GetBinContent(0));
      EXPECT_DOUBLE_EQ(0.17, hProj->GetBinContent(1));
      EXPECT_DOUBLE_EQ(0.42, hProj->GetBinContent(2));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(3));
      EXPECT_DOUBLE_EQ(0., hProj->GetBinContent(4));
      delete hProj;
   }

}

TEST(THn, Integral)
{
   Int_t bins[2] = {2, 3};
   Double_t xmin[2] = {0., -3.};
   Double_t xmax[2] = {10., 3.};
   THnD hn("hn", "hn", 2, bins, xmin, xmax);
   Double_t x[2];
   x[0] = 4;
   x[1] = -0.1;
   hn.Fill(x);
   x[0] = 7;
   x[1] = 1.0;
   hn.Fill(x);
   x[0] = 1;
   x[1] = 2.6;
   hn.Fill(x);
   x[0] = 9;
   x[1] = -0.9;
   hn.Fill(x);
   EXPECT_DOUBLE_EQ(hn.Integral(false), 4);
}

TEST(THn, GetBinCenter)
{
   Int_t bins[2] = {2, 2};
   Double_t xmin[2] = {0., -3.};
   Double_t xmax[2] = {10., 3.};
   THnD hn("hn", "hn", 2, bins, xmin, xmax);
   auto centers = hn.GetBinCenter({1, 1});
   EXPECT_DOUBLE_EQ(centers.at(0), 2.5);
   EXPECT_DOUBLE_EQ(centers.at(1), -1.5);
}

TEST(THn, ErrorsOfProjection)
{
   const int bins[] = {10, 10, 10, 10};
   const double xmin[] = {0, 0, 0, 0};
   const double xmax[] = {10, 10, 10, 10};
   THnF thn("thn", "", 4, bins, xmin, xmax);
   thn.Sumw2();

   const double coordinates1[]{0.5, 0.5, 0.5, 0.5};

   for (int i = 0; i < 9; i++) {
      thn.Fill(coordinates1, 0.1);
      const double coordinates2[]{1.5, 1.5, 0.5 + i, 0.5 + i};
      thn.Fill(coordinates2, 2.);
   }

   const Int_t dimensions[] = {0, 1};
   // Despite the option "E", the errors are resetted to sqrt(N) instead of keeping the original ones
   std::unique_ptr<THnBase> proj{thn.ProjectionND(2, dimensions, "E")};

   const auto projectedBin = proj->GetBin(coordinates1);
   EXPECT_FLOAT_EQ(proj->GetBinContent(projectedBin), 0.9);
   EXPECT_FLOAT_EQ(proj->GetBinError(projectedBin), 0.3);

   const double coordinates2[]{1.5, 1.5};
   const auto projectedBin2 = proj->GetBin(coordinates2);
   EXPECT_FLOAT_EQ(proj->GetBinContent(projectedBin2), 18.);
   EXPECT_FLOAT_EQ(proj->GetBinError(projectedBin2), 6.);
}

// https://github.com/root-project/root/issues/19366
TEST(THn, CreateSparse)
{
   Int_t bins[1] = {5};
   Double_t xmin[1] = {0.};
   Double_t xmax[1] = {1.};
   THnD hn("hn", "hn", 1, bins, xmin, xmax);
   hn.Fill(0.5);
   auto hn_sparse = THnSparseD::CreateSparse("", "", &hn);
   EXPECT_EQ(hn_sparse->GetNbins(), 1);
   EXPECT_FLOAT_EQ(hn_sparse->GetSparseFractionBins(), 1. / 7); // 5 + under/overflows
}
