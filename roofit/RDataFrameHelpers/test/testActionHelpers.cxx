/// Test the action helpers that fill roofit datasets from RDataFrame.
///
/// \date Mar 2021
/// \author Stephan Hageboeck (CERN)

#include <RooAbsDataHelper.h>
#include <RooGlobalFunc.h>
#include <RooMsgService.h>

#include <TROOT.h>
#include <TRandom.h>
#include <TSystem.h>
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
  auto dd =
     d.DefineSlot("x", [=](unsigned int /*slot*/, ULong64_t entry) { return -5. + 10. * ((double)entry) / nEvent; },
                  {"rdfentry_"})
        .DefineSlot("y", [=](unsigned int /*slot*/, ULong64_t entry) { return 0. + 2. * ((double)entry) / nEvent; },
                    {"rdfentry_"});
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


/// This test verifies that out-of-range events are correctly skipped,
/// consistent with the construction of a RooDataSet from a TTree.
TEST(RooAbsDataHelper, SkipEventsOutOfRange) {

  RooMsgService::instance().getStream(0).removeTopic(RooFit::DataHandling);
  RooMsgService::instance().getStream(1).removeTopic(RooFit::DataHandling);

  std::size_t nEvents = 100;
  const char * filename = "testRooAbsDataHelper_SkipEventsOutOfRange_tree.root";
  const char * treename = "tree";

  {
    // Create the ROOT file with the dataset
    ROOT::RDataFrame rdf(nEvents);
    auto rdf_x = rdf.Define("x", [](){ return gRandom->Gaus(0.0, 1.0); });
    rdf_x.Snapshot(treename, filename);
    // We can't reuse the same RDataFrame now, because when we rerun the event
    // loop it would generate new random values. So this scope ends here and we
    // open a new RDF from the file later.
  }

  // Open dataset with RDataFrame and TTree
  std::unique_ptr<TFile> file{TFile::Open(filename, "READ")};
  auto tree = file->Get<TTree>(treename);
  ROOT::RDataFrame rdf(treename, filename);

  // Create RooFit variable
  RooRealVar x{"x", "x", 0.0, -2.0, 2.0};

  // Create a RooDataset from the TTree, and one from the RDataFrame
  RooDataSet dataSetTree{"dataSetTree", "dataSetTree", tree, x};
  auto dataSetRDF = rdf.Book<double>(RooDataSetHelper("dataSetRDF", "dataSetRDF", RooArgSet(x)), {"x"});

  // Check if in the creation of the datasets, the entries outside the
  // variable range were sucessfully discarded.
  double nPassing = *rdf.Filter("x >= -2 && x <= 2.0").Count();

  EXPECT_EQ(dataSetRDF->numEntries(), nPassing);
  EXPECT_EQ(dataSetTree.numEntries(), nPassing);

  file.reset(); // Close file
  gSystem->Unlink(filename); // delete temporary file

  RooMsgService::instance().getStream(0).addTopic(RooFit::DataHandling);
  RooMsgService::instance().getStream(1).addTopic(RooFit::DataHandling);
}
