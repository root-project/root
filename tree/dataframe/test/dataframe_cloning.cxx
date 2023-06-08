#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx> // VariationsFor
#include <ROOT/RResultPtr.hxx> // CloneResultAndAction
#include <ROOT/RSnapshotOptions.hxx>
#include <ROOT/RDF/RDatasetSpec.hxx>
#include <ROOT/RDF/RInterface.hxx> // ChangeEmptyEntryRange, ChangeSpec
#include <ROOT/RDF/RResultMap.hxx> // CloneResultAndAction
#include <TSystem.h>               // AccessPathName

#include <gtest/gtest.h>

using ROOT::Internal::RDF::ChangeEmptyEntryRange;
using ROOT::Internal::RDF::ChangeSpec;
using ROOT::Internal::RDF::CloneResultAndAction;

template <typename T>
void EXPECT_VEC_EQ(const std::vector<T> &v1, const std::vector<T> &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST(RDataFrameCloning, Count)
{
   ROOT::RDataFrame df{100};

   auto count = df.Count();
   auto clone = CloneResultAndAction(count);

   EXPECT_EQ(*count, 100);
   EXPECT_EQ(*clone, 100);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Graph)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col2 = col1.Define("y", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto graph = col2.Graph<double, double>("x", "y");
   auto clone = CloneResultAndAction(graph);

   EXPECT_EQ((*clone).GetN(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ((*graph).GetN(), 100);
   EXPECT_DOUBLE_EQ((*graph).GetMean(), 49.5);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Mean)
{
   ROOT::RDataFrame df{10};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto mean = col1.Mean<double>("x");
   auto clone = CloneResultAndAction(mean);

   EXPECT_DOUBLE_EQ(*mean, 4.5);
   EXPECT_DOUBLE_EQ(*clone, 4.5);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Histo1DNoModel)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto histo = col1.Histo1D<double>("x");
   auto clone = CloneResultAndAction(histo);

   EXPECT_EQ((*histo).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*histo).GetMean(), 49.5);
   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Histo1DModel)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto histo = col1.Histo1D<double>({"name", "title", 100, 0, 100}, "x");
   auto clone = CloneResultAndAction(histo);

   EXPECT_EQ((*histo).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*histo).GetMean(), 49.5);
   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Histo1DModelTriggerBeforeCloning)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto histo = col1.Histo1D<double>({"name", "title", 100, 0, 100}, "x");

   EXPECT_EQ((*histo).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*histo).GetMean(), 49.5);

   auto clone = CloneResultAndAction(histo);

   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ(df.GetNRuns(), 2);
}

TEST(RDataFrameCloning, Histo2D)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col2 = col1.Define("y", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto histo = col2.Histo2D<double, double>({"name", "title", 100, 0, 100, 100, 0, 100}, "x", "y");
   auto clone = CloneResultAndAction(histo);

   EXPECT_EQ((*histo).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*histo).GetMean(), 49.5);
   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, HistoND)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x0", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col2 = col1.Define("x1", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col3 = col2.Define("x2", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col4 = col3.Define("x3", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   int nbins[4] = {10, 10, 10, 10};
   double xmin[4] = {0., 0., 0., 0.};
   double xmax[4] = {100., 100., 100., 100.};
   auto histo =
      col4.HistoND<double, double, double, double>({"name", "title", 4, nbins, xmin, xmax}, {"x0", "x1", "x2", "x3"});
   auto clone = CloneResultAndAction(histo);

   EXPECT_EQ((*histo).GetEntries(), 100);
   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Profile1D)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col2 = col1.Define("y", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto prof = col2.Profile1D<double, double>({"name", "title", 100, 0, 100, 0, 100}, "x", "y");
   auto clone = CloneResultAndAction(prof);

   EXPECT_EQ((*prof).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*prof).GetMean(), 49.5);
   EXPECT_EQ((*prof).GetMaximum(), 99);
   EXPECT_EQ((*prof).GetMinimum(), 0);
   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ((*clone).GetMaximum(), 99);
   EXPECT_EQ((*clone).GetMinimum(), 0);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Profile2D)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col2 = col1.Define("y", [](ULong64_t e) { return double(e); }, {"rdfentry_"});
   auto col3 = col2.Define("z", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto prof = col3.Profile2D<double, double, double>({"name", "title", 100, 0, 100, 100, 0, 100}, "x", "y", "z");
   auto clone = CloneResultAndAction(prof);

   EXPECT_EQ((*prof).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*prof).GetMean(), 49.5);
   EXPECT_EQ((*prof).GetMaximum(), 99);
   EXPECT_EQ((*prof).GetMinimum(), 0);
   EXPECT_EQ((*clone).GetEntries(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ((*clone).GetMaximum(), 99);
   EXPECT_EQ((*clone).GetMinimum(), 0);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Sum)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto sum = col1.Sum<double>("x");
   auto clone = CloneResultAndAction(sum);

   EXPECT_DOUBLE_EQ(*sum, 4950.);
   EXPECT_DOUBLE_EQ(*clone, 4950.);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Min)
{
   ROOT::RDataFrame df{100};
   auto filter1 = df.Filter([](ULong64_t e) { return (e < 50); }, {"rdfentry_"});

   auto min = filter1.Min<ULong64_t>("rdfentry_");
   auto clone = CloneResultAndAction(min);

   EXPECT_EQ(*min, 0ull);
   EXPECT_EQ(*clone, 0ull);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Max)
{
   ROOT::RDataFrame df{100};
   auto filter1 = df.Filter([](ULong64_t e) { return (e < 50); }, {"rdfentry_"});

   auto max = filter1.Max<ULong64_t>("rdfentry_");
   auto clone = CloneResultAndAction(max);

   EXPECT_EQ(*max, 49ull);
   EXPECT_EQ(*clone, 49ull);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Snapshot)
{
   ROOT::RDF::RSnapshotOptions opts;
   opts.fLazy = true;

   auto treeName{"events"};
   auto firstFile{"test_rdataframe_cloneactions_snapshot_1.root"};
   auto secondFile{"test_rdataframe_cloneactions_snapshot_2.root"};

   ROOT::RDataFrame df{10};
   auto snap = df.Define("x", [] { return 10; }).Snapshot<int>(treeName, firstFile, {"x"}, opts);

   auto clone = CloneResultAndAction(snap, secondFile);

   // This will trigger execution of both Snapshot nodes
   auto clonedDf = *clone;

   // Check the Snapshot clone is usable
   auto clonedSnapCount = clonedDf.Count();
   EXPECT_EQ(*clonedSnapCount, 10);

   // Check both files were produced
   EXPECT_FALSE(gSystem->AccessPathName(firstFile));
   EXPECT_FALSE(gSystem->AccessPathName(secondFile));
   EXPECT_EQ(df.GetNRuns(), 1);

   gSystem->Unlink(firstFile);
   gSystem->Unlink(secondFile);
}

TEST(RDataFrameCloning, StdDev)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto dev = col1.StdDev<double>("x");
   auto clone = CloneResultAndAction(dev);

   const double truestddev = 29.011491975882016; // True std dev computed separately
   EXPECT_DOUBLE_EQ(*dev, truestddev);
   EXPECT_DOUBLE_EQ(*clone, truestddev);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, Stats)
{
   ROOT::RDataFrame df{100};
   auto col1 = df.Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"});

   auto stats = col1.Stats<double>("x");
   auto clone = CloneResultAndAction(stats);

   EXPECT_EQ((*stats).GetN(), 100);
   EXPECT_DOUBLE_EQ((*stats).GetMean(), 49.5);
   EXPECT_EQ((*stats).GetMin(), 0);
   EXPECT_EQ((*stats).GetMax(), 99);
   EXPECT_EQ((*clone).GetN(), 100);
   EXPECT_DOUBLE_EQ((*clone).GetMean(), 49.5);
   EXPECT_EQ((*clone).GetMin(), 0);
   EXPECT_EQ((*clone).GetMax(), 99);
   EXPECT_EQ(df.GetNRuns(), 1);
}

TEST(RDataFrameCloning, VariedHisto)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary(
                 "x",
                 []() {
                    return ROOT::RVecI{-1, 2};
                 },
                 {}, 2)
               .Histo1D<int>("x");
   auto histos = ROOT::RDF::Experimental::VariationsFor(h);
   auto clones = CloneResultAndAction(histos);

   const auto &originalKeys = histos.GetKeys();
   const auto &clonedKeys = clones.GetKeys();

   EXPECT_VEC_EQ(originalKeys, clonedKeys);

   for (const auto &variation : originalKeys) {
      const auto &histo = histos[variation];
      const auto &clone = clones[variation];
      EXPECT_EQ(histo.GetMean(), clone.GetMean());
      EXPECT_EQ(histo.GetEntries(), clone.GetEntries());
   }
}

TEST(RDataFrameCloning, ChangeEmptyEntryRange)
{
   ROOT::RDataFrame df{10};
   auto col1 = df.Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"});

   ChangeEmptyEntryRange(df, {0, 4});
   auto take1 = col1.Take<ULong64_t>("x");
   EXPECT_VEC_EQ(*take1, {0, 1, 2, 3});

   ChangeEmptyEntryRange(df, {4, 8});
   auto take2 = CloneResultAndAction(take1);
   EXPECT_VEC_EQ(*take2, {4, 5, 6, 7});

   ChangeEmptyEntryRange(df, {8, 10});
   auto take3 = CloneResultAndAction(take2);
   EXPECT_VEC_EQ(*take3, {8, 9});

   EXPECT_EQ(df.GetNRuns(), 3);
}

TEST(RDataFrameCloning, ChangeSpec)
{
   std::string treeName{"events"};
   std::vector<std::string> fileNames{"dataframe_cloning_changespec_0.root", "dataframe_cloning_changespec_1.root",
                                      "dataframe_cloning_changespec_2.root"};
   // The dataset will have a total of 90 entries. We partition it in 6 different global ranges.
   // Schema: one range per complete file, one range with a portion of a single file,
   // two ranges that span more than one file.
   std::vector<std::pair<Long64_t, Long64_t>> globalRanges{{0, 30}, {30, 60}, {60, 90}, {0, 20}, {20, 50}, {50, 90}};
   {
      ROOT::RDF::RSnapshotOptions opts;
      opts.fAutoFlush = 10;

      auto df = ROOT::RDataFrame(90).Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"});
      for (unsigned i = 0; i < 3; i++) {
         // This makes sure that each output file has a different range of values for column x,
         // according to the current global entry of the dataset.
         ChangeEmptyEntryRange(df, std::pair<Long64_t, Long64_t>(globalRanges[i]));
         df.Snapshot<ULong64_t>(treeName, fileNames[i], {"x"}, opts);
      }
   }
   std::vector<ROOT::RDF::Experimental::RDatasetSpec> specs;
   specs.reserve(6);
   for (unsigned i = 0; i < 6; i++) {
      ROOT::RDF::Experimental::RDatasetSpec spec;
      // Every spec represents a different portion of the global dataset
      spec.AddSample({"", treeName, fileNames});
      spec.WithGlobalRange({globalRanges[i].first, globalRanges[i].second});
      specs.push_back(spec);
   }

   // We want to check that we are indeed changing to a different
   // range of entries of the global dataset with every call to
   // ChangeSpec. Do this by `Take`ing the values of column x for
   // every partition and checking that they correspond to the values
   // in the ranges defined by globalRanges.
   std::vector<std::vector<ULong64_t>> expectedOutputs;
   expectedOutputs.reserve(6);
   for (unsigned i = 0; i < 6; i++) {
      const auto &currentRange = globalRanges[i];
      auto nValues{currentRange.second - currentRange.first};
      std::vector<ULong64_t> takeValues(nValues);
      std::iota(takeValues.begin(), takeValues.end(), currentRange.first);
      expectedOutputs.push_back(takeValues);
   }

   // Launch first execution with dataset spec
   ROOT::RDataFrame df{specs[0]};
   auto take = df.Take<ULong64_t>("x");
   EXPECT_VEC_EQ(*take, expectedOutputs[0]);

   // Other executions modify the internal spec
   for (unsigned i = 1; i < 6; i++) {
      ChangeSpec(df, std::move(specs[i]));
      auto clone = CloneResultAndAction(take);
      EXPECT_VEC_EQ(*clone, expectedOutputs[i]);
   }

   for (const auto &name : fileNames) {
      gSystem->Unlink(name.c_str());
   }
}
