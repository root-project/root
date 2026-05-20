/// Test the action helpers that fill roofit datasets from RDataFrame.
///
/// \date Mar 2021
/// \author Stephan Hageboeck (CERN)

#include <RooAbsDataHelper.h>
#include <RooGlobalFunc.h>
#include <RooMsgService.h>

#include <ROOT/RDataFrame.hxx>

#include <TROOT.h>
#include <TRandom.h>
#include <TSystem.h>
#include <Rtypes.h>

#include "gtest/gtest.h"

#include <TFile.h>
#include <TTree.h>

namespace {

constexpr std::size_t nEvent = 200000;

constexpr double targetXMean = 0.;
constexpr double targetYMean = 1.;
constexpr double targetXVar = 100. / 12.;
constexpr double targetYVar = 4. / 12.;

auto makeDataFrame()
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT(4);
#endif

   return ROOT::RDataFrame{nEvent}
      .DefineSlot("x", [=](unsigned int /*slot*/, ULong64_t entry) { return -5. + 10. * ((double)entry) / nEvent; },
                  {"rdfentry_"})
      .DefineSlot("y", [=](unsigned int /*slot*/, ULong64_t entry) { return 0. + 2. * ((double)entry) / nEvent; },
                  {"rdfentry_"})
      .DefineSlot("w", [=](unsigned int /*slot*/, ULong64_t /*entry*/) { return 0.5; }, {"rdfentry_"});
}

RooArgSet makeVariablesSet()
{
   // We create RooFit variables that will represent the dataset.
   auto x = std::make_unique<RooRealVar>("x", "x", -5., 5.);
   auto y = std::make_unique<RooRealVar>("y", "y", -50., 50.);
   x->setBins(10);
   y->setBins(100);

   RooArgSet out;
   out.addOwned(std::move(x));
   out.addOwned(std::move(y));
   return out;
}

} // namespace

TEST(RooAbsDataHelper, RooDataSet)
{
   auto dd = makeDataFrame();
   RooArgSet vars = makeVariablesSet();
   RooRealVar &x = static_cast<RooRealVar &>(vars["x"]);
   RooRealVar &y = static_cast<RooRealVar &>(vars["y"]);

   RooDataSetHelper rdsMaker{"dataset", "Title of dataset", vars};

   auto rooDataSet = dd.Book<double, double>(std::move(rdsMaker), {"x", "y"});

   double sumEntries = nEvent;

   ASSERT_EQ(rooDataSet->numEntries(), nEvent);
   EXPECT_NEAR(rooDataSet->sumEntries(), sumEntries, sumEntries * 1.E-9);
   EXPECT_NEAR(rooDataSet->mean(x), targetXMean, 1.E-4);
   EXPECT_NEAR(rooDataSet->moment(x, 2.), targetXVar, targetXVar * 1.E-4);
   EXPECT_NEAR(rooDataSet->mean(y), targetYMean, 1.E-4);
   EXPECT_NEAR(rooDataSet->moment(y, 2.), targetYVar, targetYVar * 1.E-4);
}

TEST(RooAbsDataHelper, RooDataSetWeighted)
{
   auto dd = makeDataFrame();
   RooArgSet vars = makeVariablesSet();
   RooRealVar &x = static_cast<RooRealVar &>(vars["x"]);
   RooRealVar &y = static_cast<RooRealVar &>(vars["y"]);

   RooDataSetHelper rdsMaker{"dataset", "Title of dataset", vars, RooFit::WeightVar("w")};

   auto rooDataSet = dd.Book<double, double, double>(std::move(rdsMaker), {"x", "y", "w"});

   auto sumEntriesResult = dd.Sum("w");

   double sumEntries = sumEntriesResult.GetValue();

   ASSERT_EQ(rooDataSet->numEntries(), nEvent);
   EXPECT_NEAR(rooDataSet->sumEntries(), sumEntries, sumEntries * 1.E-9);
   EXPECT_NEAR(rooDataSet->mean(x), targetXMean, 1.E-4);
   EXPECT_NEAR(rooDataSet->moment(x, 2.), targetXVar, targetXVar * 1.E-4);
   EXPECT_NEAR(rooDataSet->mean(y), targetYMean, 1.E-4);
   EXPECT_NEAR(rooDataSet->moment(y, 2.), targetYVar, targetYVar * 1.E-4);
}

TEST(RooAbsDataHelper, RooDataHist)
{
   auto dd = makeDataFrame();
   RooArgSet vars = makeVariablesSet();
   RooRealVar &x = static_cast<RooRealVar &>(vars["x"]);
   RooRealVar &y = static_cast<RooRealVar &>(vars["y"]);

   RooDataHistHelper rdhMaker{"datahist", "Title of data hist", vars};

   // Then, we move it into the RDataFrame action:
   auto rooDataHist = dd.Book<double, double>(std::move(rdhMaker), {"x", "y"});

   double sumEntries = nEvent;

   EXPECT_NEAR(rooDataHist->sumEntries(), sumEntries, sumEntries * 1.E-9);
   EXPECT_NEAR(rooDataHist->mean(x), targetXMean, 1.E-4);
   EXPECT_NEAR(rooDataHist->moment(x, 2.), 8.25, 1.E-2); // Variance is affected in a binned distribution
   EXPECT_NEAR(rooDataHist->mean(y), targetYMean, 1.E-4);
   EXPECT_NEAR(rooDataHist->moment(y, 2.), 0.25, 1.E-2); // Variance is affected in a binned distribution
}

TEST(RooAbsDataHelper, RooDataHistWeighted)
{
   auto dd = makeDataFrame();
   RooArgSet vars = makeVariablesSet();
   RooRealVar &x = static_cast<RooRealVar &>(vars["x"]);
   RooRealVar &y = static_cast<RooRealVar &>(vars["y"]);

   RooDataHistHelper rdhMaker{"datahist", "Title of data hist", vars};

   // Then, we move it into the RDataFrame action:
   auto rooDataHist = dd.Book<double, double, double>(std::move(rdhMaker), {"x", "y", "w"});

   auto sumEntriesResult = dd.Sum("w");

   double sumEntries = sumEntriesResult.GetValue();

   EXPECT_NEAR(rooDataHist->sumEntries(), sumEntries, sumEntries * 1.E-9);
   EXPECT_NEAR(rooDataHist->mean(x), targetXMean, 1.E-4);
   EXPECT_NEAR(rooDataHist->moment(x, 2.), 8.25, 1.E-2); // Variance is affected in a binned distribution
   EXPECT_NEAR(rooDataHist->mean(y), targetYMean, 1.E-4);
   EXPECT_NEAR(rooDataHist->moment(y, 2.), 0.25, 1.E-2); // Variance is affected in a binned distribution
}

/// This test verifies that out-of-range events are correctly skipped,
/// consistent with the construction of a RooDataSet from a TTree.
TEST(RooAbsDataHelper, SkipEventsOutOfRange)
{

   RooMsgService::instance().getStream(0).removeTopic(RooFit::DataHandling);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::DataHandling);

   std::size_t nEvents = 100;
   const char *filename = "testRooAbsDataHelper_SkipEventsOutOfRange_tree.root";
   const char *treename = "tree";

   {
      // Create the ROOT file with the dataset
      ROOT::RDataFrame rdf(nEvents);
      auto rdf_x = rdf.Define("x", []() { return gRandom->Gaus(0.0, 1.0); });
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
   RooDataSet dataSetTree{"dataSetTree", "dataSetTree", x, RooFit::Import(*tree)};
   auto dataSetRDF = rdf.Book<double>(RooDataSetHelper("dataSetRDF", "dataSetRDF", RooArgSet(x)), {"x"});

   // Check if in the creation of the datasets, the entries outside the
   // variable range were successfully discarded.
   double nPassing = *rdf.Filter("x >= -2 && x <= 2.0").Count();

   EXPECT_EQ(dataSetRDF->numEntries(), nPassing);
   EXPECT_EQ(dataSetTree.numEntries(), nPassing);

   file.reset();              // Close file
   gSystem->Unlink(filename); // delete temporary file

   RooMsgService::instance().getStream(0).addTopic(RooFit::DataHandling);
   RooMsgService::instance().getStream(1).addTopic(RooFit::DataHandling);
}

TEST(RooAbsDataHelper, ColumnsOfOtherTypes)
{

   RooRealVar x("x", "x", 0);
   RooRealVar y("y", "y", 0);
   RooRealVar z("z", "z", 0);

   RooDataSetHelper dsHelper{"dataset", "Title of dataset", RooArgSet{x, y, z}};

   constexpr auto nEvents{10};
   ROOT::RDataFrame df{nEvents};
   auto dfWithCols =
      df.Define("x", []() { return std::int32_t{1}; }).Define("y", []() { return std::uint32_t{2}; }).Define("z", []() {
         return 3.f;
      });

   // The RooAbsDataHelper should be able to work with column types that can be cast to double
   auto rooDataSet = dfWithCols.Book<std::int32_t, std::uint32_t, float>(std::move(dsHelper), {"x", "y", "z"});

   // Simple checks, just to check that the dataset was filled.
   ASSERT_EQ(rooDataSet->numEntries(), nEvents);
   ASSERT_NEAR(rooDataSet->mean(x), 1, 1E-9);
   ASSERT_NEAR(rooDataSet->mean(y), 2, 1E-9);
   ASSERT_NEAR(rooDataSet->mean(z), 3, 1E-9);
}
