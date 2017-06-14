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

TEST(StressHistorgram, TestSparseData1DFull)
{
   TF1 func("GAUS", gaus1d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH1D h1("fsdf1D", "h1-title", numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   unique_ptr<THnSparse> s1(THnSparse::CreateSparse("fsdf1Ds", "THnSparse 1D - title", &h1));

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spTH1(dim, min, max);
   ROOT::Fit::FillData(spTH1, &h1, 0);

   ROOT::Fit::SparseData spSparse(dim, min, max);
   ROOT::Fit::FillData(spSparse, s1.get(), 0);

   EXPECT_TRUE(spTH1 == spSparse);
   EXPECT_TRUE(spSparse == spTH1);
}

TEST(StressHistorgram, TestSparseData1DSparse)
{
   TF1 func("GAUS", gaus1d, minRange, maxRange, 3);
   func.SetParameters(0., 3., 200.);
   func.SetParLimits(1, 0, 5);

   TH1D h1("fsds1D", "h1-title", numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < numberOfBins; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   unique_ptr<THnSparse> s1(THnSparse::CreateSparse("fsds1Ds", "THnSparse 1D - title", &h1));

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spTH1(dim, min, max);
   ROOT::Fit::FillData(spTH1, &h1, 0);

   ROOT::Fit::SparseData spSparse(dim, min, max);
   ROOT::Fit::FillData(spSparse, s1.get(), 0);

   EXPECT_TRUE(spTH1 == spSparse);
   EXPECT_TRUE(spSparse == spTH1);
}

TEST(StressHistorgram, TestSparseData2DFull)
{
   TF2 func("GAUS2D", gaus2d, minRange, maxRange, 3);
   func.SetParameters(500., +.5, 1.5, -.5, 2.0);

   TH2D h2("fsdf2D", "h2-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
   }

   unique_ptr<THnSparse> s2(THnSparse::CreateSparse("fsdf2Ds", "THnSparse 2D - title", &h2));

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spTH2(dim, min, max);
   ROOT::Fit::FillData(spTH2, &h2, 0);

   ROOT::Fit::SparseData spSparse(dim, min, max);
   ROOT::Fit::FillData(spSparse, s2.get(), 0);

   EXPECT_TRUE(spTH2 == spSparse);
   EXPECT_TRUE(spSparse == spTH2);
}

TEST(StressHistorgram, TestSparseData2DSparse)
{
   TF2 func("GAUS2D", gaus2d, minRange, maxRange, 3);
   func.SetParameters(500., +.5, 1.5, -.5, 2.0);

   TH2D h2("fsds2D", "h2-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < numberOfBins * numberOfBins; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
   }

   unique_ptr<THnSparse> s2(THnSparse::CreateSparse("fsds2Ds", "THnSparse 2D - title", &h2));

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spTH2(dim, min, max);
   ROOT::Fit::FillData(spTH2, &h2, 0);

   ROOT::Fit::SparseData spSparse(dim, min, max);
   ROOT::Fit::FillData(spSparse, s2.get(), 0);

   EXPECT_TRUE(spTH2 == spSparse);
   EXPECT_TRUE(spSparse == spTH2);
}

TEST(StressHistorgram, TestSparseData3DFull)
{
   TF2 func("GAUS3D", gaus3d, minRange, maxRange, 3);
   func.SetParameters(500., +.5, 1.5, -.5, 2.0);

   TH3D h3("fsdf3D", "h3-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, z, 1.0);
   }

   unique_ptr<THnSparse> s3(THnSparse::CreateSparse("fsdf3Ds", "THnSparse 3D - title", &h3));

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spTH3(dim, min, max);
   ROOT::Fit::FillData(spTH3, &h3, 0);

   ROOT::Fit::SparseData spSparse(dim, min, max);
   ROOT::Fit::FillData(spSparse, s3.get(), 0);

   EXPECT_TRUE(spTH3 == spSparse);
   EXPECT_TRUE(spSparse == spTH3);
}

TEST(StressHistorgram, TestSparseData3DSparse)
{
   TF2 func("GAUS3D", gaus3d, minRange, maxRange, 3);
   func.SetParameters(500., +.5, 1.5, -.5, 2.0);

   TH3D h3("fsds3D", "h3-title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                       numberOfBins, minRange, maxRange);
   for (Int_t e = 0; e < numberOfBins * numberOfBins * numberOfBins; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, z, 1.0);
   }

   unique_ptr<THnSparse> s3(THnSparse::CreateSparse("fsds3Ds", "THnSparse 3D - title", &h3));

   unsigned int const dim = 1;
   double min[dim] = {minRange};
   double max[dim] = {maxRange};
   ROOT::Fit::SparseData spTH3(dim, min, max);
   ROOT::Fit::FillData(spTH3, &h3, 0);

   ROOT::Fit::SparseData spSparse(dim, min, max);
   ROOT::Fit::FillData(spSparse, s3.get(), 0);

   EXPECT_TRUE(spTH3 == spSparse);
   EXPECT_TRUE(spSparse == spTH3);
}
