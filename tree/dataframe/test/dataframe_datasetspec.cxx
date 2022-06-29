#include <gtest/gtest.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/TestSupport.hxx>
#include <TSystem.h>

#include <thread> // std::thread::hardware_concurrency

using namespace ROOT;
using namespace ROOT::RDF::Experimental;

using namespace std::literals; // remove ambiguity of using std::vector<std::string>-s and std::string-s

void EXPECT_VEC_EQ(const std::vector<ULong64_t> &vec1, const std::vector<ULong64_t> &vec2)
{
   ASSERT_EQ(vec1.size(), vec2.size());
   for (auto i = 0u; i < vec1.size(); ++i)
      EXPECT_EQ(vec1[i], vec2[i]);
}

// The helper class is also responsible for the creation and the deletion of the root files
// Reasons: each tests needs (almost) all files; the Snapshot action is not available in MT
class RDatasetSpecTest : public ::testing::TestWithParam<bool> {
protected:
   RDatasetSpecTest() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      auto dfWriter0 = RDataFrame(5)
                          .Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"})
                          .Define("y", [](ULong64_t e) { return e + 10; }, {"rdfentry_"})
                          .Define("z", [](ULong64_t e) { return e + 100; }, {"rdfentry_"});
      dfWriter0.Snapshot<ULong64_t>("tree", "file0.root", {"x"});
      dfWriter0.Range(3).Snapshot<ULong64_t>("tree", "file1.root", {"y"});
      dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree", "file2.root", {"z"});
      dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree", "file3.root", {"z"});
      dfWriter0.Range(4, 5).Snapshot<ULong64_t>("subTree", "file4.root", {"z"});
      dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree1", "file5.root", {"z"});
      dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree2", "file6.root", {"z"});
      dfWriter0.Snapshot<ULong64_t>("anotherTree", "file7.root", {"z"});

      auto dfWriter1 = RDataFrame(10)
                          .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                          .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
      dfWriter1.Range(5).Snapshot<double>("subTree0", "file8.root", {"x"});
      dfWriter1.Range(5, 10).Snapshot<double>("subTree1", "file9.root", {"x"});
      dfWriter1.Range(5).Snapshot<double>("subTree2", "file10.root", {"w"});
      dfWriter1.Range(5, 10).Snapshot<double>("subTree3", "file11.root", {"w"});

      if (GetParam())
         ROOT::EnableImplicitMT(NSLOTS);
   }
   ~RDatasetSpecTest()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
      gSystem->Exec("rm file*.root");
   }
   const unsigned int NSLOTS;
};

// ensure that the chains are created as expected, no ranges, neither friends are passed
TEST_P(RDatasetSpecTest, SimpleChainsCreation)
{
   // first constructor: 1 tree, 1 file; both passed as string directly
   // advanced note: internally a TChain is created named the same as the tree provided
   const auto dfRDSc1 = *(RDataFrame(RDatasetSpec("tree", "file0.root")).Take<ULong64_t>("x"));
   EXPECT_VEC_EQ(dfRDSc1, {0u, 1u, 2u, 3u, 4u});

   // second constructor: 1 tree, many files; files passed as a vector; testing with 1 file
   // advanced note: internally a TChain is created named the same as the tree provided
   const auto dfRDSc21 = *(RDataFrame(RDatasetSpec("tree", {"file0.root"s})).Take<ULong64_t>("x"));
   EXPECT_VEC_EQ(dfRDSc21, {0u, 1u, 2u, 3u, 4u});

   // second constructor: here with multiple files
   auto dfRDSc2N =
      *(RDataFrame(RDatasetSpec("subTree", {"file2.root"s, "file3.root"s, "file4.root"s})).Take<ULong64_t>("z"));
   std::sort(dfRDSc2N.begin(), dfRDSc2N.end()); // the order is not necessary preserved by Take in MT
   EXPECT_VEC_EQ(dfRDSc2N, {100u, 101u, 102u, 103u, 104u});

   // third constructor: many trees, many files; trees and files passed in a vector of pairs; here 1 file
   // advanced note: when using this constructor, the internal TChain will always have no name
   const auto dfRDSc31 = *(RDataFrame(RDatasetSpec({{"tree"s}, {"file0.root"s}})).Take<ULong64_t>("x"));
   EXPECT_VEC_EQ(dfRDSc31, {0u, 1u, 2u, 3u, 4u});

   // third constructor: many trees and files
   auto dfRDSc3N =
      *(RDataFrame(
           RDatasetSpec({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}}))
           .Take<ULong64_t>("z"));
   std::sort(dfRDSc3N.begin(), dfRDSc3N.end());
   EXPECT_VEC_EQ(dfRDSc3N, {100u, 101u, 102u, 103u, 104u});
}

// ensure that ranges are properly handled, even if the ranges are invalid
// let: start = desired start, end = desired end, last = actually last entry
// possible cases: default: 0 = start < last < end = max (already implicitly tested above)
// 0. start < end <= last
// 1. similar to above but test the case the start is after the first tree)
// *  In MT runs, there is the additional optimization: once the desired end is reached,
//    stop processing further trees, i.e. range asked is [1, 3] and the df has 2 trees
//    of 5 entries each -> the second tree is not open.
// 2. 0 = start < last < end < max
// 3. 0 < start < last < end < max
// 4. start = end <= last -> enter the RLoopManager and do no work there (no sanity checks)
// 5. start = end > last -> enter the RLoopManager and do no work there (no sanity checks)
// 6. last = start < end -> error after getting the number of entries
// 7. last + 1 = start < end -> error after getting the number of entries
// 8. last < start < end -> error after getting the number of entries
// start > end -> error in the spec directly
// test all range cases for the 3 constructors
TEST_P(RDatasetSpecTest, Ranges)
{
   std::vector<std::pair<Long64_t, Long64_t>> ranges = {{1, 4}, {2, 4},   {0, 100}, {1, 100}, {2, 2},
                                                        {7, 7}, {5, 100}, {6, 100}, {42, 100}};
   std::vector<std::vector<ULong64_t>> expectedRess = {{101, 102, 103},      {102, 103}, {100, 101, 102, 103, 104},
                                                       {101, 102, 103, 104}, {},         {}};
   std::vector<std::vector<RDatasetSpec>> specs;
   // all entries across a column of specs should be the same, where each column tests all constructors
   for (const auto &r : ranges) {
      specs.push_back({});
      specs.back().push_back(RDatasetSpec("anotherTree", "file7.root", {r.first, r.second}));
      specs.back().push_back(
         RDatasetSpec("subTree", {"file2.root"s, "file3.root"s, "file4.root"s}, {r.first, r.second}));
      specs.back().push_back(
         RDatasetSpec({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}},
                      {r.first, r.second}));
   }

   // the first 6 ranges are valid entry ranges
   for (auto i = 0u; i < ranges.size(); ++i) {
      for (const auto &s : specs[i]) {
         auto takeRes = RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("z"); // lazy action
         if (i < 6u) {
            auto res = *(takeRes);
            std::sort(res.begin(), res.end());
            // std::cout << (res.empty() ? std::string(" ") : std::to_string(res[0])) << std::endl;
            EXPECT_VEC_EQ(res, expectedRess[i]);
         } else {
            EXPECT_THROW(
               try { *(takeRes); } catch (const std::logic_error &err) {
                  const auto msg = std::string("A range of entries was passed in the creation of the RDataFrame, "
                                               "but the starting entry (") +
                                   ranges[i].first +
                                   ") is larger "
                                   "than the total number of entries (5) in the dataset.";
                  EXPECT_EQ(std::string(err.what()), msg);
                  throw;
               },
               std::logic_error);
         }
      }
   }

   EXPECT_THROW(
      try {
         RDatasetSpec("anotherTree", "file7.root", {100, 7});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(
            std::string(err.what()),
            "The starting entry cannot be larger than the ending entry in the creation of a dataset specification.");
         throw;
      },
      std::logic_error);

   EXPECT_THROW(
      try {
         RDatasetSpec("subTree", {"file2.root"s, "file3.root"s, "file4.root"s}, {100, 7});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(
            std::string(err.what()),
            "The starting entry cannot be larger than the ending entry in the creation of a dataset specification.");
         throw;
      },
      std::logic_error);

   EXPECT_THROW(
      try {
         RDatasetSpec({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}},
                      {100, 7});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(
            std::string(err.what()),
            "The starting entry cannot be larger than the ending entry in the creation of a dataset specification.");
         throw;
      },
      std::logic_error);
}

// reuse the possible ranges from above
TEST_P(RDatasetSpecTest, Friends)
{
   std::vector<std::pair<Long64_t, Long64_t>> ranges = {
      {0, std::numeric_limits<Long64_t>::max()}, {1, 4}, {2, 4}, {0, 100}, {1, 100}, {2, 2}, {7, 7}};
   std::vector<std::vector<ULong64_t>> expectedRess = {
      {100, 101, 102, 103, 104}, {101, 102, 103}, {102, 103}, {100, 101, 102, 103, 104}, {101, 102, 103, 104}, {}, {}};
   std::vector<std::vector<RDatasetSpec>> specs;
   // all entries across a column of specs should be the same, where each column tests all constructors
   for (const auto &r : ranges) {
      specs.push_back({});
      specs.back().push_back(RDatasetSpec("anotherTree", "file7.root", {r.first, r.second}));
      specs.back().push_back(
         RDatasetSpec("subTree", {"file2.root"s, "file3.root"s, "file4.root"s}, {r.first, r.second}));
      specs.back().push_back(
         RDatasetSpec({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}},
                      {r.first, r.second}));

      // add all possible friends to every specification
      // friends cannot take range but can be shorter/longer than the main chain
      for (auto &sp : specs.back()) {
         sp.AddFriend("anotherTree", "file7.root", "friendTree");
         sp.AddFriend("tree", "file1.root", "friendShortTree");
         sp.AddFriend("subTree", {"file2.root"s, "file3.root"s, "file4.root"s}, "friendChain1");
         sp.AddFriend("subTree", {"file3.root"s, "file4.root"s}, "friendShortChain1");
         sp.AddFriend({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}},
                      "friendChainN");
         sp.AddFriend({{"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}}, "friendShortChainN");
      }
   }

   // ensure that the friends did not "corrupt" the main chain
   for (auto i = 0u; i < ranges.size(); ++i) {
      for (const auto &s : specs[i]) {
         auto res = *(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("z"));
         std::sort(res.begin(), res.end());
         EXPECT_VEC_EQ(res, expectedRess[i]);
      }
   }

   // TODO: error out on shorter friends, as currently silent garbage is produced
   // this garbage is of no interest to be tested
   for (auto i = 0u; i < ranges.size(); ++i) {
      for (const auto &s : specs[i]) {
         // nested structure so that the event loop is not called multiple times with different actions
         std::vector<ROOT::RDF::RResultPtr<std::vector<ULong64_t>>> friends;
         friends.push_back(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("friendTree.z"));
         // friends.push_back(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("friendShortTree.y"));
         friends.push_back(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("friendChain1.z"));
         // friends.push_back(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("friendShortChain1.z"));
         friends.push_back(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("friendChainN.z"));
         // friends.push_back(RDataFrame(RDatasetSpec(s)).Take<ULong64_t>("friendShortChainN.z"));

         for (auto &fr : friends) {
            std::sort((*fr).begin(), (*fr).end());
            EXPECT_VEC_EQ(*fr, expectedRess[i]);
         }
      }
   }
}

TEST_P(RDatasetSpecTest, Histo1D)
{
   RDatasetSpec spec({{"subTree0"s, "file8.root"s}, {"subTree1"s, "file9.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file10.root"s}, {"subTree3"s, "file11.root"s}});  // horizontal concatenation
   ROOT::RDataFrame d(spec);

   auto h1 = d.Histo1D(::TH1D("h1", "h1", 10, 0, 10), "x");
   auto h2 = d.Histo1D({"h2", "h2", 10, 0, 10}, "x");
   auto h1w = d.Histo1D(::TH1D("h0w", "h0w", 10, 0, 10), "x", "w");
   auto h2w = d.Histo1D({"h2w", "h2w", 10, 0, 10}, "x", "w");
   std::vector<double> edgesd{1, 2, 3, 4, 5, 6, 10};
   auto h1edgesd = d.Histo1D(::TH1D("h1edgesd", "h1edgesd", (int)edgesd.size() - 1, edgesd.data()), "x");
   auto h2edgesd = d.Histo1D({"h2edgesd", "h2edgesd", (int)edgesd.size() - 1, edgesd.data()}, "x");

   ROOT::RDF::TH1DModel m0("m0", "m0", 10, 0, 10);
   ROOT::RDF::TH1DModel m1(::TH1D("m1", "m1", 10, 0, 10));

   auto hm0 = d.Histo1D(m0, "x");
   auto hm1 = d.Histo1D(m1, "x");
   auto hm0w = d.Histo1D(m0, "x", "w");
   auto hm1w = d.Histo1D(m1, "x", "w");

   std::vector<double> ref({0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.});

   auto CheckBins = [](const TAxis *axis, const std::vector<double> &v) {
      auto nBins = axis->GetNbins();
      auto nBinsp1 = nBins + 1;
      auto nBinsm1 = nBins - 1;
      EXPECT_EQ(nBinsp1, (int)v.size());
      for (auto i : ROOT::TSeqI(1, nBinsp1)) {
         EXPECT_DOUBLE_EQ(axis->GetBinLowEdge(i), (double)v[i - 1]);
      }
      EXPECT_DOUBLE_EQ(axis->GetBinUpEdge(nBinsm1), (double)v[nBinsm1]);
   };

   CheckBins(h1->GetXaxis(), ref);
   CheckBins(h2->GetXaxis(), ref);
   CheckBins(hm0->GetXaxis(), ref);
   CheckBins(hm1->GetXaxis(), ref);
   CheckBins(h1edgesd->GetXaxis(), edgesd);
   CheckBins(h2edgesd->GetXaxis(), edgesd);
}

TEST_P(RDatasetSpecTest, FilterDependingOnVariation)
{
   RDatasetSpec spec({{"subTree0"s, "file8.root"s}, {"subTree1"s, "file9.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file10.root"s}, {"subTree3"s, "file11.root"s}});  // horizontal concatenation
   auto df = ROOT::RDataFrame(spec);

   auto sum = df.Vary(
                   "x",
                   [](double x) {
                      return ROOT::RVecD{x - 1., x + 2.};
                   },
                   {"x"}, 2)
                 .Filter([](double x) { return x > 5.; }, {"x"})
                 .Sum<double>("x");
   auto fr_sum = df.Vary(
                      "w",
                      [](double w) {
                         return ROOT::RVecD{w - 1., w + 2.};
                      },
                      {"w"}, 2)
                    .Filter([](double w) { return w > 5.; }, {"w"})
                    .Sum<double>("w");
   EXPECT_EQ(*sum, 30);
   EXPECT_EQ(*fr_sum, 40);

   auto sums = ROOT::RDF::Experimental::VariationsFor(sum);
   auto fr_sums = ROOT::RDF::Experimental::VariationsFor(fr_sum);

   EXPECT_EQ(sums["nominal"], 30);
   EXPECT_EQ(sums["x:0"], 21);
   EXPECT_EQ(sums["x:1"], 51);
   EXPECT_EQ(fr_sums["nominal"], 40);
   EXPECT_EQ(fr_sums["w:0"], 30);
   EXPECT_EQ(fr_sums["w:1"], 63);
}

TEST_P(RDatasetSpecTest, SaveGraph)
{
   RDatasetSpec spec({{"subTree0"s, "file8.root"s}, {"subTree1"s, "file9.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file10.root"s}, {"subTree3"s, "file11.root"s}});  // horizontal concatenation
   auto df = ROOT::RDataFrame(spec);
   auto res0 = df.Sum<double>("x");
   auto res1 = df.Sum<double>("w");

   std::string graph = ROOT::RDF::SaveGraph(df);
   // as expected the chain has no name
   static const std::string expectedGraph(
      "digraph {\n"
      "\t1 [label=<Sum>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t0 [label=<>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
      "\t2 [label=<Sum>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t0 -> 1;\n"
      "\t0 -> 2;\n"
      "}");
   EXPECT_EQ(expectedGraph, graph);
}

TEST(RDatasetSpecTest, Describe)
{
   auto dfWriter1 = RDataFrame(10)
                       .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                       .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
   dfWriter1.Range(5).Snapshot<double>("subTree0", "file8.root", {"x"});
   dfWriter1.Range(5, 10).Snapshot<double>("subTree1", "file9.root", {"x"});
   dfWriter1.Range(5).Snapshot<double>("subTree2", "file10.root", {"w"});
   dfWriter1.Range(5, 10).Snapshot<double>("subTree3", "file11.root", {"w"});

   RDatasetSpec spec({{"subTree0"s, "file8.root"s}, {"subTree1"s, "file9.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file10.root"s}, {"subTree3"s, "file11.root"s}});  // horizontal concatenation
   auto df = ROOT::RDataFrame(spec);
   auto res0 = df.Sum<double>("x");
   auto res1 = df.Sum<double>("w");

   static const std::string expectedDescribe("Dataframe from TChain  in files\n"
                                             "  file8.root\n"
                                             "  file9.root\n"
                                             "with friend\n"
                                             "  \n"
                                             "    subTree2 file10.root\n"
                                             "    subTree3 file11.root\n"
                                             "\n"
                                             "Property                Value\n"
                                             "--------                -----\n"
                                             "Columns in total            2\n"
                                             "Columns from defines        0\n"
                                             "Event loops run             0\n"
                                             "Processing slots            1\n"
                                             "\n"
                                             "Column  Type            Origin\n"
                                             "------  ----            ------\n"
                                             "w       Double_t        Dataset\n"
                                             "x       Double_t        Dataset");
   EXPECT_EQ(expectedDescribe, df.Describe().AsString());
   gSystem->Exec("rm file*.root");
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDatasetSpecTest, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDatasetSpecTest, ::testing::Values(true));
#endif
