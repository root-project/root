/// Test the action helpers that fill roofit datasets from RDataFrame.
///
/// \date Mar 2021
/// \author Stephan Hageboeck (CERN)

#include <RooAbsDataHelper.h>

#include <TROOT.h>
#include <Rtypes.h>

#include "gtest/gtest.h"

TEST(RooAbsDataHelper, MTConstruction)
{
#ifdef R__USE_IMT
  ROOT::EnableImplicitMT(4);
#endif

  // We create an RDataFrame with two columns filled with 2 million random numbers.
  constexpr std::size_t nEvent = 200000;
  ROOT::RDataFrame d(nEvent);
  auto dd = d.DefineSlot("x", [](unsigned int /*slot*/, ULong64_t entry){ return -5. + 10. * ((double)entry) / nEvent; }, {"rdfentry_"})
             .DefineSlot("y", [](unsigned int /*slot*/, ULong64_t entry){ return  0. +  2. * ((double)entry) / nEvent; }, {"rdfentry_"});
  auto meanX = dd.Mean("x");
  auto meanY = dd.Mean("y");

  constexpr double targetXMean  = 0.;
  constexpr double targetYMean  = 1.;
  constexpr double targetXVar = 100./12.;
  constexpr double targetYVar =   4./12.;

  // We create RooFit variables that will represent the dataset.
  RooRealVar x("x", "x", -5.,   5.);
  RooRealVar y("y", "y", -50., 50.);
  x.setBins(10);
  y.setBins(100);


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
  EXPECT_NEAR(meanX.GetValue(), targetXMean, 1.E-4);
  EXPECT_NEAR(meanY.GetValue(), targetYMean, 1.E-4);

  ASSERT_EQ(rooDataSet->numEntries(), nEvent);
  EXPECT_NEAR(rooDataSet->sumEntries(), nEvent, nEvent * 1.E-9);
  EXPECT_NEAR(rooDataSet->mean(x),  targetXMean, 1.E-4);
  EXPECT_NEAR(rooDataSet->moment(x, 2.), targetXVar, targetXVar * 1.E-4);
  EXPECT_NEAR(rooDataSet->mean(y),  targetYMean, 1.E-4);
  EXPECT_NEAR(rooDataSet->moment(y, 2.), targetYVar, targetYVar * 1.E-4);

  EXPECT_NEAR(rooDataHist->sumEntries(), nEvent, nEvent * 1.E-9);
  EXPECT_NEAR(rooDataHist->mean(x),  targetXMean, 1.E-4);
  EXPECT_NEAR(rooDataHist->moment(x, 2.), 8.25, 1.E-2); // Variance is affected in a binned distribution
  EXPECT_NEAR(rooDataHist->mean(y),  targetYMean, 1.E-4);
  EXPECT_NEAR(rooDataHist->moment(y, 2.), 0.25, 1.E-2); // Variance is affected in a binned distribution
}

