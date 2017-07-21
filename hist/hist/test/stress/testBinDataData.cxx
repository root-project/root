// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>
#include <cmath>

#include "TH2.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "Fit/SparseData.h"
#include "HFitInterface.h"

#include "Riostream.h"
#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestBinDataData1D)
{
   TRandom2 r(initialRandomSeed);
   TF1 func("GAUS", gaus1d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH1D h1("fbd1D", "h1-title", numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   unique_ptr<THnSparse> s1(THnSparse::CreateSparse("fbd1Ds", "THnSparse 1D - title", &h1));

   ROOT::Fit::BinData bdTH1;
   ROOT::Fit::FillData(bdTH1, &h1);

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spSparseTmp(dim, min, max);
   ROOT::Fit::FillData(spSparseTmp, s1.get(), 0);
   ROOT::Fit::BinData bdSparse;
   spSparseTmp.GetBinData(bdSparse);

   EXPECT_TRUE(bdTH1 == bdSparse);
   EXPECT_TRUE(bdSparse == bdTH1);
}

TEST(StressHistogram, TestBinDataData1DInt)
{
   TRandom2 r(initialRandomSeed);
   TF1 func("GAUS", gaus1d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH1D h1("fbdi1D", "h1-title", numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   unique_ptr<THnSparse> s1(THnSparse::CreateSparse("fbdi1Ds", "THnSparse 1D - title", &h1));

   ROOT::Fit::DataOptions opt;
   opt.fUseEmpty = true;
   opt.fIntegral = true;

   ROOT::Fit::BinData bdTH1(opt);
   ROOT::Fit::FillData(bdTH1, &h1);

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spSparseTmp(dim, min, max);
   ROOT::Fit::FillData(spSparseTmp, s1.get(), 0);
   ROOT::Fit::BinData bdSparse;
   spSparseTmp.GetBinDataIntegral(bdSparse);

   EXPECT_TRUE(bdTH1 == bdSparse);
   EXPECT_TRUE(bdSparse == bdTH1);
}

TEST(StressHistogram, TestBinDataData2D)
{
   TRandom2 r(initialRandomSeed);
   TF1 func("GAUS", gaus2d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH2D h2("fbd2D", "h2-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
   }

   unique_ptr<THnSparse> s2(THnSparse::CreateSparse("fbd2Ds", "THnSparse 2D - title", &h2));

   ROOT::Fit::BinData bdTH2;
   ROOT::Fit::FillData(bdTH2, &h2);

   unsigned int const dim = 2;
   double min[dim] = {minRange, minRange};
   double max[dim] = {maxRange, maxRange};
   ROOT::Fit::SparseData spSparseTmp(dim, min, max);
   ROOT::Fit::FillData(spSparseTmp, s2.get(), 0);
   ROOT::Fit::BinData bdSparse(spSparseTmp.NPoints(), spSparseTmp.NDim());
   spSparseTmp.GetBinData(bdSparse);

   EXPECT_TRUE(bdTH2 == bdSparse);
   EXPECT_TRUE(bdSparse == bdTH2);
}

TEST(StressHistogram, TestBinDataData2DInt)
{
   TRandom2 r(initialRandomSeed);
   TF1 func("GAUS", gaus2d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH2D h2("fbdi2D", "h2-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
   }

   unique_ptr<THnSparse> s2(THnSparse::CreateSparse("fbdi2Ds", "THnSparse 2D - title", &h2));

   ROOT::Fit::DataOptions opt;
   opt.fUseEmpty = true;
   opt.fIntegral = true;

   ROOT::Fit::BinData bdTH2(opt);
   ROOT::Fit::FillData(bdTH2, &h2);

   unsigned int const dim = 2;
   double min[dim] = {minRange, minRange};
   double max[dim] = {maxRange, maxRange};
   ROOT::Fit::SparseData spSparseTmp(dim, min, max);
   ROOT::Fit::FillData(spSparseTmp, s2.get(), 0);
   ROOT::Fit::BinData bdSparse(spSparseTmp.NPoints(), spSparseTmp.NDim());
   spSparseTmp.GetBinDataIntegral(bdSparse);

   EXPECT_TRUE(bdTH2 == bdSparse);
   EXPECT_TRUE(bdSparse == bdTH2);
}

TEST(StressHistogram, TestBinDataData3D)
{
   TRandom2 r(initialRandomSeed);
   TF1 func("GAUS", gaus3d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH3D h3("fbd3D", "h3-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange, numberOfBins,
           minRange, maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, z, 1.0);
   }

   unique_ptr<THnSparse> s3(THnSparse::CreateSparse("fbd3Ds", "THnSparse 3D - title", &h3));

   ROOT::Fit::BinData bdTH3;
   ROOT::Fit::FillData(bdTH3, &h3);

   unsigned int const dim = 3;
   double min[dim] = {minRange, minRange, minRange};
   double max[dim] = {maxRange, maxRange, maxRange};
   ROOT::Fit::SparseData spSparseTmp(dim, min, max);
   ROOT::Fit::FillData(spSparseTmp, s3.get(), 0);
   ROOT::Fit::BinData bdSparse(spSparseTmp.NPoints(), spSparseTmp.NDim());
   spSparseTmp.GetBinData(bdSparse);

   EXPECT_TRUE(bdTH3 == bdSparse);
   EXPECT_TRUE(bdSparse == bdTH3);
}

TEST(StressHistogram, TestBinDataData3DInt)
{
   TRandom2 r(initialRandomSeed);
   TF1 func("GAUS", gaus3d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH3D h3("fbdi3D", "h3-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange, numberOfBins,
           minRange, maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, z, 1.0);
   }

   unique_ptr<THnSparse> s3(THnSparse::CreateSparse("fbdi3Ds", "THnSparse 3D - title", &h3));

   ROOT::Fit::DataOptions opt;
   opt.fUseEmpty = true;
   opt.fIntegral = true;

   ROOT::Fit::BinData bdTH3(opt);
   ROOT::Fit::FillData(bdTH3, &h3);

   unsigned int const dim = 3;
   double min[dim] = {minRange, minRange, minRange};
   double max[dim] = {maxRange, maxRange, maxRange};
   ROOT::Fit::SparseData spSparseTmp(dim, min, max);
   ROOT::Fit::FillData(spSparseTmp, s3.get(), 0);
   ROOT::Fit::BinData bdSparse(spSparseTmp.NPoints(), spSparseTmp.NDim());
   spSparseTmp.GetBinDataIntegral(bdSparse);

   EXPECT_TRUE(bdTH3 == bdSparse);
   EXPECT_TRUE(bdSparse == bdTH3);
}
