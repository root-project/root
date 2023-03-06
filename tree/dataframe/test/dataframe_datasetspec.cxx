#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/TestSupport.hxx>
#include <ROOT/RDF/RSample.hxx>
#include <ROOT/RDF/RDatasetSpec.hxx>
#include <ROOT/RDF/RMetaData.hxx>
#include <nlohmann/json.hpp>
#include <TSystem.h>

#include <thread> // std::thread::hardware_concurrency

using namespace ROOT;
using namespace ROOT::RDF::Experimental;

using namespace std::literals; // remove ambiguity of using std::vector<std::string>-s and std::string-s

namespace {
void EXPECT_VEC_SEQ_EQ(const std::vector<ULong64_t> &vec, const ROOT::TSeq<ULong64_t> &seq)
{
   ASSERT_EQ(vec.size(), seq.size());
   for (auto i = 0u; i < vec.size(); ++i)
      EXPECT_EQ(vec[i], seq[i]);
}

struct RTestSample {
   std::string name;
   ULong64_t sampleStart;
   ULong64_t sampleEnd;
   std::vector<std::string> fileGlobs;

   struct RTestTree {
      std::string tree;
      std::string file;
      ULong64_t start;
      ULong64_t end;
      RTestTree(const std::string &t, const std::string &f, ULong64_t s, ULong64_t e)
         : tree(t), file(f), start(s), end(e)
      {
      }
   };
   std::vector<std::vector<RTestTree>> trees; // single tree/files with their ranges per glob
   RMetaData meta;

   RTestSample(const std::string &n, ULong64_t s, ULong64_t e, const std::vector<std::string> &f,
               const std::vector<std::vector<RTestTree>> &t)
      : name(n), sampleStart(s), sampleEnd(e), fileGlobs(f), trees(t)
   {
      meta.Add("sample_name", name);
      meta.Add("sample_start", static_cast<int>(s));
      meta.Add("sample_end", static_cast<double>(e));
   }
};

// the first sample has a single file glob
// the second sample has two file globs, but always the same tree name
// the last sample has two file globs, corresponding to different tree names
std::vector<RTestSample> data{
   {"simulated",
    0u,
    15u,
    {"specTestFile0*.root"},
    {{{"tree", "specTestFile0.root", 0u, 4u},
      {"tree", "specTestFile00.root", 4u, 9u},
      {"tree", "specTestFile000.root", 9u, 15u}}}},
   {"real",
    15u,
    39u,
    {"specTestFile1.root", "specTestFile11*.root"},
    {{{"subTree", "specTestFile1.root", 15u, 22u}},
     {{"subTree", "specTestFile11.root", 22u, 30u}, {"subTree", "specTestFile111.root", 30u, 39u}}}},
   {"raw",
    39u,
    85u,
    {"specTestFile2*.root", "specTestFile3*.root"},
    {{{"subTreeA", "specTestFile2.root", 39u, 49u}, {"subTreeA", "specTestFile22.root", 49u, 60u}},
     {{"subTreeB", "specTestFile3.root", 60u, 72u}, {"subTreeB", "specTestFile33.root", 72u, 85u}}}}};
} // namespace

// The helper class is also responsible for the creation and the deletion of the root specTestFiles
class RDatasetSpecTest : public ::testing::TestWithParam<bool> {
protected:
   RDatasetSpecTest() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(NSLOTS);
   }

   ~RDatasetSpecTest()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }

   const unsigned int NSLOTS;

public:
   static void SetUpTestCase()
   {
      auto dfWriter = ROOT::RDataFrame(85).Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"});
      for (const auto &d : data)
         for (const auto &trees : d.trees)
            for (const auto &t : trees)
               dfWriter.Range(t.start, t.end).Snapshot<ULong64_t>(t.tree, t.file, {"x"});
   }

   static void TearDownTestCase()
   {
      for (const auto &d : data)
         for (const auto &trees : d.trees)
            for (const auto &t : trees)
               gSystem->Unlink((t.file).c_str());
   }
};

// ensure that the chains are created as expected: no metadata, no ranges, neither friends are passed
TEST_P(RDatasetSpecTest, SimpleChainsCreation)
{
   // testing each single tree with hard-coded file names
   for (const auto &d : data) {
      for (const auto &trees : d.trees) {
         for (const auto &t : trees) {
            // first AddSample overload: 1 tree, 1 file; both passed as string directly
            const auto dfC1 = *(RDataFrame(RDatasetSpec().AddSample({"", t.tree, t.file})).Take<ULong64_t>("x"));
            EXPECT_VEC_SEQ_EQ(dfC1, ROOT::TSeq<ULong64_t>(t.start, t.end));

            // second AddSample overload: 1 tree, many files; files passed as a vector; testing with 1 specTestFile
            const auto dfC2 = *(RDataFrame(RDatasetSpec().AddSample({"", t.tree, {t.file}})).Take<ULong64_t>("x"));
            EXPECT_VEC_SEQ_EQ(dfC2, ROOT::TSeq<ULong64_t>(t.start, t.end));

            // third AddSample overload: many trees, many files; trees and files passed in a vector of pairs
            const auto dfC3 = *(RDataFrame(RDatasetSpec().AddSample({"", {{t.tree, t.file}}})).Take<ULong64_t>("x"));
            EXPECT_VEC_SEQ_EQ(dfC3, ROOT::TSeq<ULong64_t>(t.start, t.end));

            // fourth AddSample overload: many trees, many files; trees and files passed in separate vectors
            const auto dfC4 = *(RDataFrame(RDatasetSpec().AddSample({"", {t.tree}, {t.file}})).Take<ULong64_t>("x"));
            EXPECT_VEC_SEQ_EQ(dfC4, ROOT::TSeq<ULong64_t>(t.start, t.end));
         }
      }
   }

   // samples with hard-coded file names
   std::vector<RDatasetSpec> specs(4);
   for (const auto &d : data) {
      for (const auto &trees : d.trees) {
         for (const auto &t : trees) {
            specs[0].AddSample({"", t.tree, t.file});
            specs[1].AddSample({"", t.tree, {t.file}});
            specs[2].AddSample({"", {{t.tree, t.file}}});
            specs[3].AddSample({"", {t.tree}, {t.file}});
         }
      }
   }

   for (const auto &spec : specs) {
      auto df = *(RDataFrame(spec).Take<ULong64_t>("x"));
      std::sort(df.begin(), df.end());
      EXPECT_VEC_SEQ_EQ(df, ROOT::TSeq<ULong64_t>(data[0].sampleStart, data[data.size() - 1].sampleEnd));
   }

   // the first builder takes each tree/file as a separate sample
   // the second builder takes tree/file glob as separate sample
   // the third build takes tree/expanded glob (similar to second)
   // the fourth builder takes a vector of trees/vector of file globs as a separate sample
   std::vector<RDatasetSpec> specsGranularity(4);
   for (const auto &d : data) {
      std::vector<std::string> treeNames{};
      std::vector<std::string> fileGlobs{};
      for (auto i = 0u; i < d.fileGlobs.size(); ++i) {
         std::vector<std::string> treeNamesExpanded{};
         std::vector<std::string> fileGlobsExpanded{};
         for (const auto &t : d.trees[i]) {
            specsGranularity[0].AddSample({"", t.tree, t.file});
            treeNamesExpanded.emplace_back(t.tree);
            fileGlobsExpanded.emplace_back(t.file);
         }
         specsGranularity[1].AddSample({"", d.trees[i][0].tree, d.fileGlobs[i]});
         specsGranularity[2].AddSample({"", treeNamesExpanded, fileGlobsExpanded});
         treeNames.emplace_back(d.trees[i][0].tree);
         fileGlobs.emplace_back(d.fileGlobs[i]);
      }
      specsGranularity[3].AddSample({"", treeNames, fileGlobs});
   }
   for (const auto &spec : specsGranularity) {
      auto df = *(RDataFrame(spec).Take<ULong64_t>("x"));
      std::sort(df.begin(), df.end());
      EXPECT_VEC_SEQ_EQ(df, ROOT::TSeq<ULong64_t>(data[0].sampleStart, data[data.size() - 1].sampleEnd));
   }
}

TEST_P(RDatasetSpecTest, SimpleMetaDataHandling)
{
   std::vector<RDatasetSpec> specs(2);
   for (const auto &d : data) {
      std::vector<std::string> treeNames{};
      std::vector<std::string> fileGlobs{};
      std::vector<std::string> treeNamesExpanded{};
      std::vector<std::string> fileGlobsExpanded{};
      for (auto i = 0u; i < d.fileGlobs.size(); ++i) {
         for (const auto &t : d.trees[i]) {
            treeNamesExpanded.emplace_back(t.tree);
            fileGlobsExpanded.emplace_back(t.file);
         }
         treeNames.emplace_back(d.trees[i][0].tree);
         fileGlobs.emplace_back(d.fileGlobs[i]);
      }
      specs[0].AddSample({d.name, treeNames, fileGlobs, d.meta});
      specs[1].AddSample({d.name, treeNamesExpanded, fileGlobsExpanded, d.meta});
   }
   for (const auto &spec : specs) {
      auto df =
         RDataFrame(spec)
            .DefinePerSample("sample_name_col",
                             [](unsigned int, const ROOT::RDF::RSampleInfo &id) { return id.GetS("sample_name"); })
            .DefinePerSample("sample_start_col",
                             [](unsigned int, const ROOT::RDF::RSampleInfo &id) { return id.GetI("sample_start"); })
            .DefinePerSample("sample_end_col",
                             [](unsigned int, const ROOT::RDF::RSampleInfo &id) { return id.GetD("sample_end"); });
      auto g1S = (df.Filter([](std::string m) { return m == "simulated"; }, {"sample_name_col"}).Count());
      auto g2S = (df.Filter([](std::string m) { return m == "real"; }, {"sample_name_col"}).Count());
      auto g3S = (df.Filter([](std::string m) { return m == "raw"; }, {"sample_name_col"}).Count());
      auto g1I = (df.Filter([](int m) { return m == 0; }, {"sample_start_col"}).Count());
      auto g2I = (df.Filter([](int m) { return m == 15; }, {"sample_start_col"}).Count());
      auto g3I = (df.Filter([](int m) { return m == 39; }, {"sample_start_col"}).Count());
      auto g1D = (df.Filter([](double m) { return m == 15.; }, {"sample_end_col"}).Count());
      auto g2D = (df.Filter([](double m) { return m == 39.; }, {"sample_end_col"}).Count());
      auto g3D = (df.Filter([](double m) { return m == 85.; }, {"sample_end_col"}).Count());
      EXPECT_EQ(*g1S, 15u); // number of entries in the first sample = 4 + 5 + 6 = 15
      EXPECT_EQ(*g2S, 24u); // num. entries = 7 + 8 + 9 = 24
      EXPECT_EQ(*g3S, 46u); // num. entries = 10 + 11 + 12 + 13 = 46
      EXPECT_EQ(*g1I, 15u);
      EXPECT_EQ(*g2I, 24u);
      EXPECT_EQ(*g3I, 46u);
      EXPECT_EQ(*g1D, 15u);
      EXPECT_EQ(*g2D, 24u);
      EXPECT_EQ(*g3D, 46u);
   }
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
TEST_P(RDatasetSpecTest, Ranges)
{
   std::vector<RDatasetSpec::REntryRange> ranges = {{1, 4},     {2, 4},    {100},     {1, 100}, {2, 2},
                                                    {100, 100}, {85, 100}, {86, 100}, {92, 100}};

   std::vector<RDatasetSpec> specs(2);
   for (const auto &d : data) {
      std::vector<std::string> treeNames{};
      std::vector<std::string> fileGlobs{};
      std::vector<std::string> treeNamesExpanded{};
      std::vector<std::string> fileGlobsExpanded{};
      for (auto i = 0u; i < d.fileGlobs.size(); ++i) {
         for (const auto &t : d.trees[i]) {
            treeNamesExpanded.emplace_back(t.tree);
            fileGlobsExpanded.emplace_back(t.file);
         }
         treeNames.emplace_back(d.trees[i][0].tree);
         fileGlobs.emplace_back(d.fileGlobs[i]);
      }
      specs[0].AddSample({d.name, treeNames, fileGlobs, d.meta});
      specs[1].AddSample({d.name, treeNamesExpanded, fileGlobsExpanded, d.meta});
   }
   for (auto &spec : specs) {
      for (auto i = 0u; i < ranges.size(); ++i) {
         auto df = RDataFrame(spec.WithGlobalRange(ranges[i]));
         auto takeRes = df.Take<ULong64_t>("x"); // lazy action
         if (i < 6u) {                           // the first 6 ranges, are logically correct
            auto &res = *takeRes;
            std::sort(res.begin(), res.end());
            if (ranges[i].fBegin != ranges[i].fEnd)
               EXPECT_VEC_SEQ_EQ(res, ROOT::TSeq<ULong64_t>(ranges[i].fBegin, std::min(85ll, ranges[i].fEnd)));
            else
               EXPECT_EQ((*takeRes).size(), 0u);
         } else {
            if (GetParam()) { // MT case, error coming from the TTreeProcessorMT
               EXPECT_THROW(
                  try { *takeRes; } catch (const std::logic_error &err) {
                     const auto msg =
                        std::string("A range of entries was passed in the creation of the TTreeProcessorMT, "
                                    "but the starting entry (") +
                        ranges[i].fBegin +
                        ") is larger "
                        "than the total number of entries (85) in the dataset.";
                     EXPECT_EQ(std::string(err.what()), msg);
                     throw;
                  },
                  std::logic_error);
            } else {
               if (i == 6u) // Single-threaded case undesired behaviour, see
                            // https://github.com/root-project/root/issues/10774
                  EXPECT_EQ((*takeRes).size(), 0u);
               else {
                  EXPECT_THROW(
                     try {
                        ROOT_EXPECT_ERROR(*takeRes, "TTreeReader::SetEntriesRange()",
                                          (std::string("Error setting first entry ") + ranges[i].fBegin +
                                           ": one of the readers was not successfully initialized")
                                             .Data());
                     } catch (const std::logic_error &err) { // error coming from the RLoopManager
                        EXPECT_STREQ(err.what(), "Something went wrong in initializing the TTreeReader.");
                        throw;
                     },
                     std::logic_error);
               }
            }
         }
      }
   }

   EXPECT_THROW(
      try { RDatasetSpec::REntryRange(100, 7); } catch (const std::logic_error &err) {
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

   RDatasetSpec spec;
   // pick the second sample as the main chain, so that can test shorter, equal-sized, longer friends
   spec.AddSample({data[1].name, data[1].trees[0][0].tree, data[1].fileGlobs});
   for (const auto &d : data) {
      std::vector<std::string> treeNames{};
      std::vector<std::string> fileGlobs{};
      std::vector<std::string> treeNamesExpanded{};
      std::vector<std::string> fileGlobsExpanded{};
      for (auto i = 0u; i < d.fileGlobs.size(); ++i) {
         for (const auto &t : d.trees[i]) {
            treeNamesExpanded.emplace_back(t.tree);
            fileGlobsExpanded.emplace_back(t.file);
         }
         treeNames.emplace_back(d.trees[i][0].tree);
         fileGlobs.emplace_back(d.fileGlobs[i]);
      }
      spec.WithGlobalFriends(treeNames, fileGlobs, "friend_glob_" + d.name);
      spec.WithGlobalFriends(treeNamesExpanded, fileGlobsExpanded, "friend_expanded_" + d.name);
   }
   auto df = RDataFrame(spec);
   std::unordered_map<std::string, ROOT::RDF::RResultPtr<std::vector<ULong64_t>>> res;
   for (const auto &d : data) {
      res["friend_glob_" + d.name + "x"] = df.Take<ULong64_t>("friend_glob_" + d.name + ".x");
      res["friend_expanded_" + d.name + "x"] = df.Take<ULong64_t>("friend_expanded_" + d.name + ".x");
   }

   for (auto i = 0u; i < data.size(); ++i) {
      std::sort(res["friend_glob_" + data[i].name + "x"]->begin(), res["friend_glob_" + data[i].name + "x"]->end());
      std::sort(res["friend_expanded_" + data[i].name + "x"]->begin(),
                res["friend_expanded_" + data[i].name + "x"]->end());
      // case i = 0 is shorter friend, which currently leads to undesired behaviour
      // see: https://github.com/root-project/root/issues/9137
      if (i > 0) {
         EXPECT_VEC_SEQ_EQ(*res["friend_glob_" + data[i].name + "x"],
                           ROOT::TSeq<ULong64_t>(data[i].sampleStart, data[i].sampleStart + 24));
      }
   }
}

TEST_P(RDatasetSpecTest, Histo1D)
{
   RDatasetSpec spec;
   spec.AddSample({"real0", "tree"s, {"specTestFile0.root"s}});
   spec.AddSample({"real1", {{"tree"s, "specTestFile00*.root"s}}});
   // 1 friend with entries from 15 up to 39 -> shortened to have the size of the main chain
   spec.WithGlobalFriends({{"subTree"s, "specTestFile1*.root"s}}, "friend"s);
   ROOT::RDataFrame d(spec);

   auto h1 = d.Histo1D(::TH1D("h1", "h1", 10, 0, 10), "x");
   auto h2 = d.Histo1D({"h2", "h2", 10, 0, 10}, "x");
   // use the friend column as weight
   auto h1w = d.Histo1D(::TH1D("h0w", "h0w", 10, 0, 10), "x", "friend.x");
   auto h2w = d.Histo1D({"h2w", "h2w", 10, 0, 10}, "x", "friend.x");
   std::vector<double> edgesd{1, 2, 3, 4, 5, 6, 10};
   auto h1edgesd = d.Histo1D(::TH1D("h1edgesd", "h1edgesd", (int)edgesd.size() - 1, edgesd.data()), "x");
   auto h2edgesd = d.Histo1D({"h2edgesd", "h2edgesd", (int)edgesd.size() - 1, edgesd.data()}, "x");

   ROOT::RDF::TH1DModel m0("m0", "m0", 10, 0, 10);
   ROOT::RDF::TH1DModel m1(::TH1D("m1", "m1", 10, 0, 10));

   auto hm0 = d.Histo1D(m0, "x");
   auto hm1 = d.Histo1D(m1, "x");
   auto hm0w = d.Histo1D(m0, "x", "friend.x");
   auto hm1w = d.Histo1D(m1, "x", "friend.x");

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
   RDatasetSpec spec;
   spec.AddSample({"real0", "tree"s, {"specTestFile0.root"s}});
   spec.AddSample({"real1", {{"tree"s, "specTestFile00*.root"s}}});
   // 1 friend with entries from 15 up to 39 -> shortened to have the size of the main chain
   spec.WithGlobalFriends({{"subTree"s, "specTestFile1*.root"s}}, "friend"s);
   ROOT::RDataFrame df(spec);

   auto sum = df.Vary(
                   "x",
                   [](ULong64_t x) {
                      return ROOT::RVec<ULong64_t>{x + 5, x + 6};
                   },
                   {"x"}, 2)
                 .Filter([](ULong64_t x) { return x > 5u; }, {"x"})
                 .Sum<ULong64_t>("x");
   auto fr_sum = df.Vary(
                      "friend.x",
                      [](ULong64_t w) {
                         return ROOT::RVec<ULong64_t>{w + 1, w + 2};
                      },
                      {"friend.x"}, 2, "w")
                    .Filter([](ULong64_t w) { return w > 5u; }, {"friend.x"})
                    .Sum<ULong64_t>("friend.x");
   EXPECT_EQ(*sum, 90);
   EXPECT_EQ(*fr_sum, 330);

   auto sums = ROOT::RDF::Experimental::VariationsFor(sum);
   auto fr_sums = ROOT::RDF::Experimental::VariationsFor(fr_sum);

   EXPECT_EQ(sums["nominal"], 90);
   EXPECT_EQ(sums["x:0"], 175);
   EXPECT_EQ(sums["x:1"], 195);
   EXPECT_EQ(fr_sums["nominal"], 330);
   EXPECT_EQ(fr_sums["w:0"], 345);
   EXPECT_EQ(fr_sums["w:1"], 360);
}

// TODO: Enhance how TChains with no names are showed by SaveGraph (and Describe)
// see: https://github.com/root-project/root/issues/10928
TEST_P(RDatasetSpecTest, SaveGraph)
{
   RDatasetSpec spec;
   spec.AddSample({"real0", "tree"s, {"specTestFile0.root"s}});
   spec.AddSample({"real1", {{"tree"s, "specTestFile00*.root"s}}});
   // 1 friend with entries from 15 up to 39 -> shortened to have the size of the main chain
   spec.WithGlobalFriends({{"subTree"s, "specTestFile1*.root"s}}, "friend"s);
   ROOT::RDataFrame df(spec);
   auto res0 = df.Sum<double>("x");
   auto res1 = df.Sum<double>("friend.x");

   std::string graph = ROOT::RDF::SaveGraph(df);
   // as expected the chain has no name
   static const std::string expectedGraph(
      "digraph {\n"
      "\t1 [label=\"Sum\", style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t0 [label=\"\", style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
      "\t2 [label=\"Sum\", style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
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
   dfWriter1.Range(5).Snapshot<double>("subTree0", "specTestDescribe1.root", {"x"});
   dfWriter1.Range(5, 10).Snapshot<double>("subTree1", "specTestDescribe2.root", {"x"});
   dfWriter1.Range(5).Snapshot<double>("subTree2", "specTestDescribe3.root", {"w"});
   dfWriter1.Range(5, 10).Snapshot<double>("subTree3", "specTestDescribe4.root", {"w"});

   RDatasetSpec spec;
   spec.AddSample({"sampleA", "subTree0"s, "specTestDescribe1.root"s});
   spec.AddSample({"sampleB", "subTree1"s, "specTestDescribe2.root"s});
   spec.WithGlobalFriends({{"subTree2"s, "specTestDescribe3.root"s}, {"subTree3"s, "specTestDescribe4.root"s}});
   auto df = ROOT::RDataFrame(spec);
   auto res0 = df.Sum<double>("x");
   auto res1 = df.Sum<double>("w");

   static const std::string expectedDescribe("Dataframe from TChain  in files\n"
                                             "  specTestDescribe1.root\n"
                                             "  specTestDescribe2.root\n"
                                             "with friend\n"
                                             "  \n"
                                             "    subTree2 specTestDescribe3.root\n"
                                             "    subTree3 specTestDescribe4.root\n"
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
   for (auto i = 12u; i < 16; ++i)
      gSystem->Unlink(("specTestFile" + std::to_string(i) + ".root").c_str());
}

TEST(RDatasetSpecTest, FromSpec)
{
   auto dfWriter0 = ROOT::RDataFrame(5).Define("z", [](ULong64_t e) { return e + 100; }, {"rdfentry_"});
   dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree", "PYspecTestFile2.root", {"z"});
   dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree", "PYspecTestFile3.root", {"z"});
   dfWriter0.Range(4, 5).Snapshot<ULong64_t>("subTree", "PYspecTestFile4.root", {"z"});
   dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree1", "PYspecTestFile5.root", {"z"});
   dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree2", "PYspecTestFile6.root", {"z"});
   dfWriter0.Snapshot<ULong64_t>("anotherTree", "PYspecTestFile7.root", {"z"});

   auto rdf =
      FromSpec("spec.json")
         .DefinePerSample("name", [](unsigned int, const ROOT::RDF::RSampleInfo &id) { return id.GetSampleName(); })
         .DefinePerSample("lumi", "rdfsampleinfo_.GetD(\"lumi\")");
   auto resP = rdf.Take<ULong64_t>("z");
   auto namP = rdf.Take<std::string>("name");
   auto lumP = rdf.Take<double>("lumi");
   auto fr1P = rdf.Take<ULong64_t>("friendTree.z");
   auto fr2P = rdf.Take<ULong64_t>("friendChain1.z");
   auto res = resP.GetValue();
   auto nam = namP.GetValue();
   auto lum = lumP.GetValue();
   auto fr1 = fr1P.GetValue();
   auto fr2 = fr2P.GetValue();

   std::vector<ULong64_t> expectedRes{101, 102, 103, 104};
   std::vector<std::string> names{"sampleA", "sampleA", "sampleA", "sampleB"};
   std::vector<double> lumis{1.0, 1.0, 1.0, 0.5};

   ASSERT_EQ(res.size(), expectedRes.size());
   for (auto i = 0u; i < expectedRes.size(); ++i) {
      EXPECT_EQ(nam[i], names[i]);
      EXPECT_DOUBLE_EQ(lum[i], lumis[i]);
      EXPECT_EQ(res[i], expectedRes[i]);
      EXPECT_EQ(fr1[i], expectedRes[i]);
      EXPECT_EQ(fr2[i], expectedRes[i]);
   }

   for (auto i = 2u; i < 8u; ++i)
      gSystem->Unlink(("PYspecTestFile" + std::to_string(i) + ".root").c_str());
}

TEST(RMetaData, SimpleOperations)
{
   ROOT::RDF::Experimental::RMetaData m;
   m.Add("year", 2022);
   m.Add("energy", 13.6);
   m.Add("framework", "ROOT6");
   EXPECT_EQ(m.GetI("year"), 2022);
   EXPECT_DOUBLE_EQ(m.GetD("energy"), 13.6);
   EXPECT_EQ(m.GetS("framework"), "ROOT6");
   EXPECT_EQ(m.Dump("year"), "2022");
   EXPECT_EQ(m.Dump("energy"), "13.6");
   EXPECT_EQ(m.Dump("framework"), "\"ROOT6\"");

   // adding to the same key currently overwrites
   // overwriting with the same type is okay
   m.Add("year", 2023);
   m.Add("energy", 13.9);
   m.Add("framework", "ROOT7");
   EXPECT_EQ(m.GetI("year"), 2023);
   EXPECT_DOUBLE_EQ(m.GetD("energy"), 13.9);
   EXPECT_EQ(m.GetS("framework"), "ROOT7");
   // overwriting with different types
   m.Add("year", 20.23);
   m.Add("energy", "14");
   m.Add("framework", 7);
   EXPECT_DOUBLE_EQ(m.GetD("year"), 20.23);
   EXPECT_EQ(m.GetS("energy"), "14");
   EXPECT_EQ(m.GetI("framework"), 7);
}

TEST(RMetaData, InvalidQueries)
{
   ROOT::RDF::Experimental::RMetaData m;
   m.Add("year", 2022);
   m.Add("energy", 13.6);
   m.Add("framework", "ROOT6");

   // asking for the wrong type, but valid column
   EXPECT_THROW(m.GetD("year"), std::logic_error);
   EXPECT_THROW(m.GetS("energy"), std::logic_error);
   EXPECT_THROW(m.GetI("framework"), std::logic_error);

   // asking for non-existent columns
   EXPECT_THROW(m.GetD("alpha"), std::logic_error);
   EXPECT_THROW(m.GetS("beta"), std::logic_error);
   EXPECT_THROW(m.GetI("gamma"), std::logic_error);
}

#ifdef R__USE_IMT
TEST(RDatasetSpecTest, Clusters)
{
   // schema: main chain: x {0-39, 39-119, 120-199}, friend chain: friend.x {0-49, 50-150, 150-200}
   auto dfWriter2 = RDataFrame(200).Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"});
   std::vector<ROOT::RDF::RSnapshotOptions> opts(4);
   opts[0].fAutoFlush = 10; // split where each tree is evenly split
   opts[1].fAutoFlush = 20; // split where only the main trees are evenly split
   opts[2].fAutoFlush = 7;  // last split is always shorter
   opts[3].fAutoFlush = 1;  // each entry is a split
   for (auto &opt : opts) {
      dfWriter2.Range(40).Snapshot<ULong64_t>("mainA", "CspecTestFile0.root", {"x"}, opt);
      dfWriter2.Range(40, 120).Snapshot<ULong64_t>("mainB", "CspecTestFile1.root", {"x"}, opt);
      dfWriter2.Range(120, 200).Snapshot<ULong64_t>("mainC", "CspecTestFile2.root", {"x"}, opt);
      dfWriter2.Range(50).Snapshot<ULong64_t>("friendA", "CspecTestFile3.root", {"x"}, opt);
      dfWriter2.Range(50, 150).Snapshot<ULong64_t>("friendB", "CspecTestFile4.root", {"x"}, opt);
      dfWriter2.Range(150, 200).Snapshot<ULong64_t>("friendC", "CspecTestFile5.root", {"x"}, opt);

      ROOT::EnableImplicitMT(std::min(4u, std::thread::hardware_concurrency()));

      std::vector<ROOT::RDF::Experimental::RDatasetSpec::REntryRange> ranges = {{}, {20}, {100, 110}, {130, 200}};

      for (const auto &range : ranges) {
         RDatasetSpec spec;
         spec.AddSample({"",
                         {{"mainA"s, "CspecTestFile0.root"s},
                          {"mainB"s, "CspecTestFile1.root"s},
                          {"mainC"s, "CspecTestFile2.root"s}}});
         spec.WithGlobalRange(range);
         spec.WithGlobalFriends({{"friendA"s, "CspecTestFile3.root"s},
                                 {"friendB"s, "CspecTestFile4.root"s},
                                 {"friendC"s, "CspecTestFile5.root"s}},
                                "friend");
         auto takeRes = RDataFrame(spec).Take<ULong64_t>("x");              // lazy action
         auto takeFriendRes = RDataFrame(spec).Take<ULong64_t>("friend.x"); // lazy action
         auto &res = *takeRes;
         auto &friendRes = *takeFriendRes;
         std::sort(res.begin(), res.end());
         std::sort(friendRes.begin(), friendRes.end());

         EXPECT_VEC_SEQ_EQ(res, ROOT::TSeq<ULong64_t>(range.fBegin, std::min(200ll, range.fEnd)));
         EXPECT_VEC_SEQ_EQ(friendRes, ROOT::TSeq<ULong64_t>(range.fBegin, std::min(200ll, range.fEnd)));
      }

      ROOT::DisableImplicitMT();

      for (auto i = 0u; i < 6; ++i)
         gSystem->Unlink(("CspecTestFile" + std::to_string(i) + ".root").c_str());
   }
}
#endif

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDatasetSpecTest, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDatasetSpecTest, ::testing::Values(true));
#endif
