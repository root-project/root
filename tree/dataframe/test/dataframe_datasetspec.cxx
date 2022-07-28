#include <gtest/gtest.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TSystem.h>

using namespace ROOT;
using namespace ROOT::Internal::RDF;

using namespace std::literals; // remove ambiguity of using std::vector<std::string>-s and std::string-s

void EXPECT_VEC_EQ(const std::vector<ULong64_t> &vec1, const std::vector<ULong64_t> &vec2)
{
   ASSERT_EQ(vec1.size(), vec2.size());
   for (auto i = 0u; i < vec1.size(); ++i)
      EXPECT_EQ(vec1[i], vec2[i]);
}

// The helper class is also responsible for the creation and the deletion of the root specTestFiles
class RDatasetSpecTest : public ::testing::TestWithParam<bool> {
protected:
   RDatasetSpecTest() {}

   ~RDatasetSpecTest() {}

   static void SetUpTestCase()
   {
      auto dfWriter0 = RDataFrame(5).Define("z", [](ULong64_t e) { return e + 100; }, {"rdfentry_"});
      dfWriter0.Snapshot<ULong64_t>("tree", "specTestFile0.root", {"z"});
      dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree", "specTestFile1.root", {"z"});
      dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree", "specTestFile2.root", {"z"});
      dfWriter0.Range(4, 5).Snapshot<ULong64_t>("subTree", "specTestFile3.root", {"z"});
      dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTreeA", "specTestFile4.root", {"z"});
      dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTreeB", "specTestFile5.root", {"z"});

      auto dfWriter1 = RDataFrame(10)
                          .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                          .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
      dfWriter1.Range(5).Snapshot<double>("subTree0", "specTestFile6.root", {"x"});
      dfWriter1.Range(5, 10).Snapshot<double>("subTree1", "specTestFile7.root", {"x"});
      dfWriter1.Range(5).Snapshot<double>("subTree2", "specTestFile8.root", {"w"});
      dfWriter1.Range(5, 10).Snapshot<double>("subTree3", "specTestFile9.root", {"w"});
   }

   static void TearDownTestCase()
   {
      for (auto i = 0u; i < 10; ++i)
         gSystem->Unlink(("specTestFile" + std::to_string(i) + ".root").c_str());
   }
};

// ensure that the chains are created as expected, no ranges, neither friends are passed
TEST_P(RDatasetSpecTest, SimpleChainsCreation)
{
   // first constructor: 1 tree, 1 file; both passed as string directly
   // advanced note: internally a TChain is created named the same as the tree provided
   const auto dfRDSc1 = *(MakeDataFrameFromSpec(RDatasetSpec("tree", "specTestFile0.root")).Take<ULong64_t>("z"));
   EXPECT_VEC_EQ(dfRDSc1, {100u, 101u, 102u, 103u, 104u});

   // second constructor: 1 tree, many files; files passed as a vector; testing with 1 specTestFile
   // advanced note: internally a TChain is created named the same as the tree provided
   const auto dfRDSc21 = *(MakeDataFrameFromSpec(RDatasetSpec("tree", {"specTestFile0.root"s})).Take<ULong64_t>("z"));
   EXPECT_VEC_EQ(dfRDSc21, {100u, 101u, 102u, 103u, 104u});

   // second constructor: here with multiple files
   auto dfRDSc2N = *(MakeDataFrameFromSpec(
                        RDatasetSpec("subTree", {"specTestFile1.root"s, "specTestFile2.root"s, "specTestFile3.root"s}))
                        .Take<ULong64_t>("z"));
   std::sort(dfRDSc2N.begin(), dfRDSc2N.end()); // the order is not necessary preserved by Take in MT
   EXPECT_VEC_EQ(dfRDSc2N, {100u, 101u, 102u, 103u, 104u});

   // third constructor: many trees, many files; trees and files passed in a vector of pairs; here 1 specTestFile
   // advanced note: when using this constructor, the internal TChain will always have no name
   const auto dfRDSc31 =
      *(MakeDataFrameFromSpec(RDatasetSpec({{"tree"s, "specTestFile0.root"s}})).Take<ULong64_t>("z"));
   EXPECT_VEC_EQ(dfRDSc31, {100u, 101u, 102u, 103u, 104u});

   // third constructor: many trees and files
   auto dfRDSc3N = *(MakeDataFrameFromSpec(RDatasetSpec({{"subTreeA"s, "specTestFile4.root"s},
                                                         {"subTreeB"s, "specTestFile5.root"s},
                                                         {"subTree"s, "specTestFile3.root"s}}))
                        .Take<ULong64_t>("z"));
   std::sort(dfRDSc3N.begin(), dfRDSc3N.end());
   EXPECT_VEC_EQ(dfRDSc3N, {100u, 101u, 102u, 103u, 104u});
}

// reuse the possible ranges from above
TEST_P(RDatasetSpecTest, Friends)
{
   std::vector<RDatasetSpec::REntryRange> ranges = {{}, {1, 4}, {2, 4}, {100}, {1, 100}, {2, 2}, {7, 7}};
   std::vector<std::vector<ULong64_t>> expectedRess = {
      {100, 101, 102, 103, 104}, {101, 102, 103}, {102, 103}, {100, 101, 102, 103, 104}, {101, 102, 103, 104}, {}, {}};
   std::vector<std::vector<RDatasetSpec>> specs;
   // all entries across a column of specs should be the same, where each column tests all constructors
   for (const auto &r : ranges) {
      specs.push_back({});
      specs.back().push_back(RDatasetSpec("tree", "specTestFile0.root", r));
      specs.back().push_back(
         RDatasetSpec("subTree", {"specTestFile1.root"s, "specTestFile2.root"s, "specTestFile3.root"s}, r));
      specs.back().push_back(RDatasetSpec({{"subTreeA"s, "specTestFile4.root"s},
                                           {"subTreeB"s, "specTestFile5.root"s},
                                           {"subTree"s, "specTestFile3.root"s}},
                                          r));

      // add all possible friends to every specification
      // friends cannot take range but can be shorter/longer than the main chain
      for (auto &sp : specs.back()) {
         sp.AddFriend("tree", "specTestFile0.root", "friendTree");
         sp.AddFriend("subTree", "specTestFile1.root", "friendShortTree");
         sp.AddFriend("subTree", {"specTestFile1.root"s, "specTestFile2.root"s, "specTestFile3.root"s}, "friendChain1");
         sp.AddFriend("subTree", {"specTestFile1.root"s, "specTestFile2.root"s}, "friendShortChain1");
         sp.AddFriend({{"subTreeA"s, "specTestFile4.root"s},
                       {"subTreeB"s, "specTestFile5.root"s},
                       {"subTree"s, "specTestFile3.root"s}},
                      "friendChainN");
         sp.AddFriend({{"subTreeA"s, "specTestFile4.root"s}, {"subTreeB"s, "specTestFile5.root"s}},
                      "friendShortChainN");
      }
   }

   // ensure that the friends did not "corrupt" the main chain
   for (auto i = 0u; i < ranges.size(); ++i) {
      for (const auto &s : specs[i]) {
         auto res = *(MakeDataFrameFromSpec(s).Take<ULong64_t>("z"));
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
         friends.push_back(MakeDataFrameFromSpec(s).Take<ULong64_t>("friendTree.z"));
         friends.push_back(MakeDataFrameFromSpec(s).Take<ULong64_t>("friendChain1.z"));
         friends.push_back(MakeDataFrameFromSpec(s).Take<ULong64_t>("friendChainN.z"));

         for (auto &fr : friends) {
            std::sort(fr->begin(), fr->end());
            EXPECT_VEC_EQ(*fr, expectedRess[i]);
         }
      }
   }
}

TEST_P(RDatasetSpecTest, Histo1D)
{
   RDatasetSpec spec(
      {{"subTree0"s, "specTestFile6.root"s}, {"subTree1"s, "specTestFile7.root"s}}); // vertical concatenation
   spec.AddFriend(
      {{"subTree2"s, "specTestFile8.root"s}, {"subTree3"s, "specTestFile9.root"s}}); // horizontal concatenation
   auto d = MakeDataFrameFromSpec(spec);

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
   RDatasetSpec spec(
      {{"subTree0"s, "specTestFile6.root"s}, {"subTree1"s, "specTestFile7.root"s}}); // vertical concatenation
   spec.AddFriend(
      {{"subTree2"s, "specTestFile8.root"s}, {"subTree3"s, "specTestFile9.root"s}}); // horizontal concatenation
   auto df = MakeDataFrameFromSpec(spec);

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

TEST(RDatasetSpecTest, Describe)
{
   auto dfWriter1 = RDataFrame(10)
                       .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                       .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
   dfWriter1.Range(5).Snapshot<double>("subTree0", "specTestFile12.root", {"x"});
   dfWriter1.Range(5, 10).Snapshot<double>("subTree1", "specTestFile13.root", {"x"});
   dfWriter1.Range(5).Snapshot<double>("subTree2", "specTestFile14.root", {"w"});
   dfWriter1.Range(5, 10).Snapshot<double>("subTree3", "specTestFile15.root", {"w"});

   RDatasetSpec spec(
      {{"subTree0"s, "specTestFile12.root"s}, {"subTree1"s, "specTestFile13.root"s}}); // vertical concatenation
   spec.AddFriend(
      {{"subTree2"s, "specTestFile14.root"s}, {"subTree3"s, "specTestFile15.root"s}}); // horizontal concatenation
   auto df = MakeDataFrameFromSpec(spec);
   auto res0 = df.Sum<double>("x");
   auto res1 = df.Sum<double>("w");

   static const std::string expectedDescribe("Dataframe from TChain  in files\n"
                                             "  specTestFile12.root\n"
                                             "  specTestFile13.root\n"
                                             "with friend\n"
                                             "  \n"
                                             "    subTree2 specTestFile14.root\n"
                                             "    subTree3 specTestFile15.root\n"
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

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDatasetSpecTest, ::testing::Values(false));
