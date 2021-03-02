/// Test the action helpers that fill roofit datasets from RDataFrame.
///
/// \date Mar 2021
/// \author Stephan Hageboeck (CERN)

#include <RooAbsDataHelper.h>

#include <TROOT.h>
#include <TRandom3.h>

#include "gtest/gtest.h"

#include <vector>

TEST(RooAbsDataHelper, MTConstruction)
{
#ifdef R__USE_IMT
  ROOT::EnableImplicitMT(4);
#endif

  const auto nSlots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
  std::vector<TRandom3> rngs(nSlots);
  for (unsigned int i=0; i < rngs.size(); ++i) {
    rngs[i].SetSeed(i+1);
  }

  // We create an RDataFrame with two columns filled with 2 million random numbers.
  constexpr std::size_t nEvent = 200000;
  ROOT::RDataFrame d(nEvent);
  auto dd = d.DefineSlot("x", [&rngs](unsigned int slot){ return rngs[slot].Uniform(-5.,  5.); })
             .DefineSlot("y", [&rngs](unsigned int slot){ return rngs[slot].Gaus(1., 3.); });
  auto meanX = dd.Mean("x");
  auto meanY = dd.Mean("y");


  // We create RooFit variables that will represent the dataset.
  RooRealVar x("x", "x", -5.,   5.);
  RooRealVar y("y", "y", -50., 50.);
  x.setBins(10);
  y.setBins(20);


  auto rooDataSet = dd.Book<double, double>(
      RooDataSetHelper("dataset", // Name
          "Title of dataset",     // Title
          RooArgSet(x, y)         // Variables in this dataset
          ),
      {"x", "y"}                  // Column names in RDataFrame.
  );

  RooDataHistHelper rdhMaker{"datahist",  // Name
    "Title of data hist",                 // Title
    RooArgSet(x, y)                       // Variables in this dataset
  };

  // Then, we move it into the RDataFrame action:
  auto rooDataHist = dd.Book<double, double>(std::move(rdhMaker), {"x", "y"});



  // Run it and inspect the results
  // -------------------------------
  EXPECT_NEAR(meanX.GetValue(), 0., 1.E-2); // Pretty bad accuracy, but might be the RNG or summation in RDF
  EXPECT_NEAR(meanY.GetValue(), 1., 1.E-2);

  constexpr double goodPrecision = 1.E-9;
  constexpr double middlePrecision = 3.E-3;
  constexpr double badPrecision = 2.E-2;
  ASSERT_EQ(rooDataSet->numEntries(), nEvent);
  EXPECT_NEAR(rooDataSet->sumEntries(), nEvent, nEvent * goodPrecision);
  EXPECT_NEAR(rooDataSet->mean(x),  meanX.GetValue(), goodPrecision);
  EXPECT_NEAR(rooDataSet->moment(x, 2.), 100./12., 100./12. * middlePrecision);
  EXPECT_NEAR(rooDataSet->mean(y),  meanY.GetValue(), goodPrecision);
  EXPECT_NEAR(rooDataSet->moment(y, 2.), 3.*3., 3.*3. * badPrecision);

  EXPECT_NEAR(rooDataHist->sumEntries(), nEvent, nEvent * goodPrecision);
  EXPECT_NEAR(rooDataHist->mean(x),  meanX.GetValue(), badPrecision);
  EXPECT_NEAR(rooDataHist->moment(x, 2.), 100./12., 100./12. * badPrecision);
  EXPECT_NEAR(rooDataHist->mean(y),  meanY.GetValue(), badPrecision);
  EXPECT_NEAR(std::sqrt(rooDataHist->moment(y, 2.)), 3., 0.4); // Sigma is hard to measure in a binned distribution
}

