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
#include <ROOT/RDF/RInterface.hxx> // ChangeEmptyEntryRange, ChangeBeginAndEndEntries, ChangeSpec
#include <ROOT/RDF/RResultMap.hxx> // CloneResultAndAction
#include <RtypesCore.h>            // ULong64_t
#include <TSystem.h>               // AccessPathName

#include <gtest/gtest.h>

#include <TFile.h>

using ROOT::Internal::RDF::ChangeBeginAndEndEntries;
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

struct InputFilesRAII {
   std::vector<std::string> fFileNames;

   InputFilesRAII(const std::string &treeName, const std::vector<std::string> &fileNames, ULong64_t entriesPerFile = 10,
                  const std::vector<ULong64_t> &beginEntryPerFile = {})
      : fFileNames(fileNames)
   {
      auto realBeginEntries = beginEntryPerFile.empty() ? std::vector<ULong64_t>(fileNames.size()) : beginEntryPerFile;

      for (std::size_t i = 0; i < fileNames.size(); i++) {
         TFile f(fFileNames[i].c_str(), "recreate");
         TTree t(treeName.c_str(), treeName.c_str());
         // Always create a new TTree cluster every 10 entries
         t.SetAutoFlush(10);
         ULong64_t x;
         t.Branch("x", &x);
         for (ULong64_t j = realBeginEntries[i]; j < (realBeginEntries[i] + entriesPerFile); j++) {
            x = j;
            t.Fill();
         }
         t.Write();
      }
   }

   ~InputFilesRAII()
   {
      for (const auto &fileName : fFileNames)
         gSystem->Unlink(fileName.c_str());
   }
};

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
   auto snap = df.Define("x", [] { return 10; }).Snapshot(treeName, firstFile, {"x"}, opts);

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

TEST(RDataFrameCloning, ChangeBeginAndEndEntries)
{
   auto treeName{"events"};
   auto firstFile{"test_rdataframe_cloneactions_ChangeEmptyRange.root"};
   {
      ROOT::RDataFrame df{10};
      df.Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"}).Snapshot(treeName, firstFile);
   }

   ROOT::RDataFrame df{treeName, firstFile};

   ChangeBeginAndEndEntries(df, 0, 3);
   auto take1 = df.Take<ULong64_t>("x");
   EXPECT_VEC_EQ(*take1, {0, 1, 2});

   ChangeBeginAndEndEntries(df, 3, 7);
   auto take2 = CloneResultAndAction(take1);
   EXPECT_VEC_EQ(*take2, {3, 4, 5, 6});

   ChangeBeginAndEndEntries(df, 7, 10);
   auto take3 = CloneResultAndAction(take2);
   EXPECT_VEC_EQ(*take3, {7, 8, 9});

   EXPECT_EQ(df.GetNRuns(), 3);

   gSystem->Unlink(firstFile);
}

TEST(RDataFrameCloning, ChangeSpec)
{
   std::string treeName{"events"};
   std::size_t nFiles{3};
   // Each file has 30 entries, starting from a different value
   ULong64_t entriesPerFile{30};
   std::vector<ULong64_t> beginEntryPerFile{0, 30, 60};
   std::vector<std::string> fileNames(nFiles);
   std::string prefix{"dataframe_cloning_changespec_"};
   std::generate(fileNames.begin(), fileNames.end(), [n = 0, &prefix]() mutable {
      auto name = prefix + std::to_string(n) + ".root";
      n++;
      return name;
   });
   InputFilesRAII files{treeName, fileNames, entriesPerFile, beginEntryPerFile};

   // The dataset will have a total of 90 entries. We partition it in 6 different global ranges.
   // Schema: one range per complete file, one range with a portion of a single file,
   // two ranges that span more than one file.
   std::vector<std::pair<Long64_t, Long64_t>> globalRanges{{0, 30}, {30, 60}, {60, 90}, {0, 20}, {20, 50}, {50, 90}};

   std::vector<ROOT::RDF::Experimental::RDatasetSpec> specs;
   specs.reserve(globalRanges.size());
   for (unsigned i = 0; i < globalRanges.size(); i++) {
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
   expectedOutputs.reserve(globalRanges.size());
   for (unsigned i = 0; i < globalRanges.size(); i++) {
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
   for (unsigned i = 1; i < globalRanges.size(); i++) {
      ChangeSpec(df, std::move(specs[i]));
      auto clone = CloneResultAndAction(take);
      EXPECT_VEC_EQ(*clone, expectedOutputs[i]);
   }
}

ROOT::RDF::Experimental::RResultMap<TH1D> dataframe_cloning_vary_with_filters_analysis(ROOT::RDF::RNode df)
{
   auto df1 = df.Define("jet_pt", []() { return ROOT::RVecF{30, 30, 30, 10, 10, 10, 10}; });

   auto df2 = df1.Vary("jet_pt",
                       [](ULong64_t) {
                          return ROOT::RVec<ROOT::RVecF>{{31, 31, 31, 31, 11, 11, 11}, {29, 29, 29, 29, 9, 9, 9}};
                       },
                       {"rdfentry_"}, {"pt_scale_up", "pt_res_up"});

   auto df3 =
      df2.Define("jet_pt_mask", [](const ROOT::RVecF &jet_pt) { return jet_pt > 25; }, {"jet_pt"})
         .Filter([](const ROOT::RVecI &jet_pt_mask) { return ROOT::VecOps::Sum(jet_pt_mask) >= 4; }, {"jet_pt_mask"});

   auto df4 = df3.Vary(
      "weights",
      [](const ROOT::RVecF &jet_pt, const ROOT::RVecI &jet_pt_mask) {
         // The next line triggered an error due to a previous faulty implementation
         // that was not connecting the cloned varied action to the correct upstream
         // varied filters. In particular, this lambda should be run after the `Filter`
         // above, which checks that the number of jets that passes the `jet_pt_mask`
         // is at least 4.
         auto test = ROOT::VecOps::Take(jet_pt[jet_pt_mask], 4);
         return ROOT::RVecD{-1, 1};
      },
      {"jet_pt", "jet_pt_mask"}, 2);

   auto h = df4.Histo1D<ROOT::RVecF, double>({"NAME", "TITLE", 10, -10, 10}, "jet_pt", "weights");

   auto vars = ROOT::RDF::Experimental::VariationsFor(h);

   return vars;
}

/*
 * This test reproduces a particular issue happening in analysis logic
 * including both systematic variations and filters, with the order
 *
 * 1. Some call to Vary
 * 2. Filter depending on the first Vary
 * 3. Another Vary that depends on the Filter
 *
 * The logic that reproduces the problem is shown in the above function
 * `dataframe_cloning_vary_with_filters_analysis`. In particular, the
 * lambda used in the Vary call (3) calls ROOT::VecOps::Take(vec, 4),
 * thus only works if `vec` has at least size `4`.
 *
 * In a previous implementation of cloning `RVariedAction`, the cloned action
 * was connected to the wrong upstream filter (in fact, it was using a varied
 * filter from the original action as if it was the nominal filter). This would
 * then pass to the lambda used in the downstream Vary call (3) a vector
 * `jet_pt` from which only 3 entries pass `jet_pt_mask`, eventually producing
 * an error when reaching the `Take(jet_pt[jet_pt_mask], 4)` call.
 */
TEST(RDataFrameCloning, VaryWithFilters)
{
   ROOT::RDataFrame d{1};
   auto df = d.Define("weights", []() { return 0.; });

   // Create the computation graph and trigger it.
   auto vars = dataframe_cloning_vary_with_filters_analysis(df);
   ROOT::Internal::RDF::TriggerRun(df);

   // Clone the RResultMap and re-run the computation graph
   auto vars_clone = CloneResultAndAction(vars);
   ROOT::Internal::RDF::TriggerRun(df);

   // Make sure that the cloned varied results are the same
   // as the original ones.
   const auto &originalKeys = vars.GetKeys();
   const auto &clonedKeys = vars_clone.GetKeys();

   EXPECT_VEC_EQ(originalKeys, clonedKeys);
   for (const auto &variation : originalKeys) {
      const auto &histo = vars[variation];
      const auto &clone = vars_clone[variation];
      EXPECT_DOUBLE_EQ(histo.GetMean(), clone.GetMean());
      EXPECT_EQ(histo.GetEntries(), clone.GetEntries());
   }
}

TEST(RDataFrameCloning, DefinePerSample)
{
   std::string treeName{"events"};
   std::size_t nFiles{3};
   std::vector<std::string> fileNames(nFiles);
   std::string prefix{"dataframe_cloning_definepersample_"};
   std::generate(fileNames.begin(), fileNames.end(), [n = 0, &prefix]() mutable {
      auto name = prefix + std::to_string(n) + ".root";
      n++;
      return name;
   });
   InputFilesRAII files{treeName, fileNames};
   std::vector<double> weights(nFiles);
   std::iota(weights.begin(), weights.end(), 1.);

   // The first specification takes the first two files and reads them both from beginning to end.
   // The second specification takes the second and third file, but only reads the third one.
   // This simulates two tasks that might be created when logically splitting the input dataset.
   std::vector<std::pair<Long64_t, Long64_t>> globalRanges{{0, 20}, {10, 20}};
   std::vector<std::vector<std::string>> taskFileNames{
      {"dataframe_cloning_definepersample_0.root", "dataframe_cloning_definepersample_1.root"},
      {"dataframe_cloning_definepersample_1.root", "dataframe_cloning_definepersample_2.root"}};
   std::vector<ROOT::RDF::Experimental::RDatasetSpec> specs;
   specs.reserve(2);
   for (unsigned i = 0; i < 2; i++) {
      ROOT::RDF::Experimental::RDatasetSpec spec;
      spec.AddSample({"", treeName, taskFileNames[i]});
      spec.WithGlobalRange({globalRanges[i].first, globalRanges[i].second});
      specs.push_back(spec);
   }

   // Launch first execution with dataset spec
   ROOT::RDataFrame df{specs[0]};
   auto dfWithCols =
      df.DefinePerSample("sample_weight",
                         [&weights, &fileNames](unsigned int, const ROOT::RDF::RSampleInfo &id) {
                            if (id.Contains(fileNames[0])) {
                               return weights[0];
                            } else if (id.Contains(fileNames[1])) {
                               return weights[1];
                            } else if (id.Contains(fileNames[2])) {
                               return weights[2];
                            } else {
                               return -999.;
                            }
                         })
         .DefinePerSample("sample_name", [](unsigned int, const ROOT::RDF::RSampleInfo &id) { return id.AsString(); });

   // One filter per each different combination of weight and sample name
   // Counting the entries passing each filter should return exactly the entries
   // of the corresponding file, i.e. 10.
   auto c0 = dfWithCols
                .Filter(
                   [&weights, &fileNames, &treeName](double weight, const std::string &name) {
                      return weight == weights[0] && name == (fileNames[0] + "/" + treeName);
                   },
                   {"sample_weight", "sample_name"})
                .Count();
   auto c1 = dfWithCols
                .Filter(
                   [&weights, &fileNames, &treeName](double weight, const std::string &name) {
                      return weight == weights[1] && name == (fileNames[1] + "/" + treeName);
                   },
                   {"sample_weight", "sample_name"})
                .Count();
   auto c2 = dfWithCols
                .Filter(
                   [&weights, &fileNames, &treeName](double weight, const std::string &name) {
                      return weight == weights[2] && name == (fileNames[2] + "/" + treeName);
                   },
                   {"sample_weight", "sample_name"})
                .Count();

   std::vector<ULong64_t> expectedFirstTask{10, 10, 0};
   EXPECT_EQ(*c0, expectedFirstTask[0]);
   EXPECT_EQ(*c1, expectedFirstTask[1]);
   EXPECT_EQ(*c2, expectedFirstTask[2]);

   // Assign the other specification and clone actions
   ChangeSpec(df, std::move(specs[1]));
   auto c3 = CloneResultAndAction(c0);
   auto c4 = CloneResultAndAction(c1);
   auto c5 = CloneResultAndAction(c2);
   std::vector<ULong64_t> expectedSecondTask{0, 0, 10};
   EXPECT_EQ(*c3, expectedSecondTask[0]);
   EXPECT_EQ(*c4, expectedSecondTask[1]);
   EXPECT_EQ(*c5, expectedSecondTask[2]);
}
