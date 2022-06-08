#include <gtest/gtest.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/TestSupport.hxx>
#include <TSystem.h>

#include <thread> // std::thread::hardware_concurrency

using namespace ROOT;
using namespace ROOT::RDF;

using namespace std::literals; // remove ambiguity of using std::vector<std::string>-s and std::string-s

TEST(RDFDatasetSpec, SingleFileSingleColConstructor)
{
   auto dfWriter = RDataFrame(5).Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"});
   dfWriter.Snapshot<int>("tree", "file.root", {"x"});

   // "reference" values
   const auto dfSimple = RDataFrame("tree", "file.root").Display<int>({"x"})->AsString();
   const auto dfRange0 = RDataFrame("tree", "file.root").Range(2, 4).Display<int>({"x"})->AsString();
   const auto dfRange1 = RDataFrame("tree", "file.root").Range(2, 5).Display<int>({"x"})->AsString();
   const auto dfRange2 = RDataFrame("tree", "file.root").Range(2).Display<int>({"x"})->AsString();
   const auto dfEmpty = "+-----+---+\n| Row | x | \n|     |   | \n+-----+---+\n";

   // specify only tree and file names
   const auto dfRDS0 = RDataFrame(RDatasetSpec("tree", "file.root")).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS0, dfSimple);

   // specify end, which is < numEntries
   const auto dfRDSX = RDataFrame(RDatasetSpec("tree", "file.root", {2})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDSX, dfRange2);

   // specify end, which is == numEntries
   auto dfRDSY = RDataFrame(RDatasetSpec("tree", "file.root", {5})).Display<int>({"x"})->AsString();
   // std::string dfRDSYAsString;
   // ROOT_EXPECT_ERROR(dfRDSYAsString = dfRDSY->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..5");
   EXPECT_EQ(dfRDSY, dfSimple);

   // specify end, which is > numEntries
   auto dfRDSZ = RDataFrame(RDatasetSpec("tree", "file.root", {7})).Display<int>({"x"})->AsString();
   // std::string dfRDSZAsString;
   // ROOT_EXPECT_ERROR(dfRDSYAsString = dfRDSZ->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..5");
   EXPECT_EQ(dfRDSZ, dfSimple);

   // specify tree name, file name, starting index, ending index, where both indices are valid => [2, 4)
   const auto dfRDS1 = RDataFrame(RDatasetSpec("tree", "file.root", {2, 4})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS1, dfRange0);

   // specify range [2, 2) (3 is a valid index) => range is disregarded
   const auto dfRDS7 = RDataFrame(RDatasetSpec("tree", "file.root", {2, 2})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS7, dfEmpty);

   // specify range [7, 7) (7 is not a valid index) => range is disregarded
   const auto dfRDS8 = RDataFrame(RDatasetSpec("tree", "file.root", {7, 7})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS8, dfEmpty);

   // specify range [2, 6) (6 is not a valid index) => range becomes [2, 5)
   const auto dfRDS9 = RDataFrame(RDatasetSpec("tree", "file.root", {2, 6})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS9, dfRange1);

   // specify range [9, 7) (neither is a valid index) => logic error
   EXPECT_THROW(
      try {
         RDatasetSpec("tree", "file.root", {9, 7});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(std::string(err.what()), "RDatasetSpec: fStartEntry cannot be larger than fEndEntry.");
         throw;
      },
      std::logic_error);

   // specify range [5, 6) (neither is a valid index, but 5 < 6) => error out
   auto h = RDataFrame(RDatasetSpec("tree", "file.root", {5, 6})).Display<int>({"x"});
   EXPECT_THROW(
      try {
         ROOT_EXPECT_ERROR(h->AsString(), "TTreeReader::SetEntriesRange()", "first entry out of range 0..5");
      } catch (const std::runtime_error &err) {
         EXPECT_EQ(std::string(err.what()), "RLoopManager: fStartEntry cannot be larger than the number of entries.");
         throw;
      },
      std::runtime_error);

   // test the second constructor, second argument is now a vector
   const auto dfRDS13 = RDataFrame(RDatasetSpec("tree", {"file.root"s})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS13, dfSimple);

   // test the second constructor, second argument is now a vector, specify range as well
   const auto dfRDS14 = RDataFrame(RDatasetSpec("tree", {"file.root"s}, {2, 4})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS14, dfRange0);

   gSystem->Exec("rm file.root");
}

TEST(RDFDatasetSpec, SingleFileMultiColsConstructor)
{
   auto dfWriter = RDataFrame(5)
                      .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
                      .Define("y", [](ULong64_t e) { return int(e) + 1; }, {"rdfentry_"});
   dfWriter.Snapshot<int, int>("tree", "file.root", {"x", "y"});

   // "reference" values
   const auto dfSimple = RDataFrame("tree", "file.root").Display<int, int>({"x", "y"})->AsString();
   const auto dfRange0 = RDataFrame("tree", "file.root").Range(2, 4).Display<int, int>({"x", "y"})->AsString();
   const auto dfRange1 = RDataFrame("tree", "file.root").Range(2, 5).Display<int, int>({"x", "y"})->AsString();
   const auto dfRange2 = RDataFrame("tree", "file.root").Range(2).Display<int, int>({"x", "y"})->AsString();
   const auto dfEmpty = "+-----+---+---+\n| Row | x | y | \n|     |   |   | \n+-----+---+---+\n";

   // specify only tree and file names, do not pass arguments to Display to show all available columns
   const auto dfRDS0 = RDataFrame(RDatasetSpec("tree", "file.root")).Display()->AsString();
   EXPECT_EQ(dfRDS0, dfSimple);

   // specify meaningful ranges
   const auto dfRDS1 = RDataFrame(RDatasetSpec("tree", "file.root", {0, 5})).Display()->AsString();
   EXPECT_EQ(dfRDS1, dfSimple);

   const auto dfRDS4 = RDataFrame(RDatasetSpec("tree", "file.root", {2, 4})).Display()->AsString();
   EXPECT_EQ(dfRDS4, dfRange0);

   // specify range [5, 6) (neither is a valid index, but 5 < 6) => error out
   auto h = RDataFrame(RDatasetSpec("tree", "file.root", {5, 6})).Display<int>({"x"});
   EXPECT_THROW(
      try {
         ROOT_EXPECT_ERROR(h->AsString(), "TTreeReader::SetEntriesRange()", "first entry out of range 0..5");
      } catch (const std::runtime_error &err) {
         EXPECT_EQ(std::string(err.what()), "RLoopManager: fStartEntry cannot be larger than the number of entries.");
         throw;
      },
      std::runtime_error);

   // specify irrelgular ranges (similar to above): [2, 2), [7, 7), [2, 6), [2, 0), [9, 7), [9, 2)
   const auto dfRDS9 = RDataFrame(RDatasetSpec("tree", "file.root", {2, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS9, dfEmpty);

   const auto dfRDS10 = RDataFrame(RDatasetSpec("tree", "file.root", {7, 7})).Display()->AsString();
   EXPECT_EQ(dfRDS10, dfEmpty);

   const auto dfRDS11 = RDataFrame(RDatasetSpec("tree", "file.root", {2, 6})).Display()->AsString();
   EXPECT_EQ(dfRDS11, dfRange1);

   const auto dfRDSX = RDataFrame(RDatasetSpec("tree", "file.root", {2})).Display()->AsString();
   EXPECT_EQ(dfRDSX, dfRange2);

   auto dfRDSY = RDataFrame(RDatasetSpec("tree", "file.root", {5})).Display()->AsString();
   // std::string dfRDSYAsString;
   // ROOT_EXPECT_ERROR(dfRDSYAsString = dfRDSY->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..5");
   EXPECT_EQ(dfRDSY, dfSimple);

   auto dfRDSZ = RDataFrame(RDatasetSpec("tree", "file.root", {7})).Display()->AsString();
   // std::string dfRDSZAsString;
   // ROOT_EXPECT_ERROR(dfRDSZAsString = dfRDSZ->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..5");
   EXPECT_EQ(dfRDSZ, dfSimple);

   EXPECT_THROW(
      try {
         RDatasetSpec("tree", "file.root", {9, 7});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(std::string(err.what()), "RDatasetSpec: fStartEntry cannot be larger than fEndEntry.");
         throw;
      },
      std::logic_error);
   EXPECT_THROW(
      try {
         RDatasetSpec("tree", "file.root", {9, 2});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(std::string(err.what()), "RDatasetSpec: fStartEntry cannot be larger than fEndEntry.");
         throw;
      },
      std::logic_error);

   // test the second constructor, second argument is now a vector
   const auto dfRDS12 = RDataFrame(RDatasetSpec("tree", {"file.root"s})).Display()->AsString();
   EXPECT_EQ(dfRDS12, dfSimple);

   gSystem->Exec("rm file.root");
}

TEST(RDFDatasetSpec, MultipleFiles)
{
   auto dfWriter0 = RDataFrame(3)
                       .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
                       .Define("y", [](ULong64_t e) { return int(e) + 1; }, {"rdfentry_"});
   dfWriter0.Snapshot<int, int>("treeA", "file0.root", {"x", "y"});
   dfWriter0.Snapshot<int, int>("treeA", "file1.root", {"y", "x"}); // columns in different order
   dfWriter0.Snapshot<int, int>("treeB", "file2.root", {"x", "y"}); // different tree's name

   // "reference" values
   TChain ch0("treeA");
   ch0.Add("file0.root");
   ch0.Add("file1.root");
   const auto dfSimple = RDataFrame(ch0).Display<int, int>({"x", "y"})->AsString();
   const auto dfRange = RDataFrame(ch0).Range(1, 2).Display<int, int>({"x", "y"})->AsString();
   const auto dfRange1 = RDataFrame(ch0).Range(2, 6).Display<int, int>({"x", "y"})->AsString();
   const auto dfRange2 = RDataFrame(ch0).Range(2).Display<int, int>({"x", "y"})->AsString();
   const auto dfEmpty = "+-----+---+---+\n| Row | x | y | \n|     |   |   | \n+-----+---+---+\n";

   const auto dfRDS0 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s})).Display()->AsString();
   EXPECT_EQ(dfRDS0, dfSimple);

   const auto dfRDS1 =
      RDataFrame(RDatasetSpec({{"treeA"s, "file0.root"s}, {"treeA"s, "file1.root"s}}, {0, 5})).Display()->AsString();
   EXPECT_EQ(dfRDS1, dfSimple);

   // files have different chain name => need a chain
   const auto dfRDS2 =
      RDataFrame(RDatasetSpec({{"treeA"s, "file1.root"s}, {"treeB"s, "file2.root"s}}, {0, 5})).Display()->AsString();
   EXPECT_EQ(dfRDS2, dfSimple);

   // cases similar to above, but now range is applied, note that the range is global (i.e. not per tree, but per chain)
   const auto dfRDS3 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {1, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS3, dfRange);

   const auto dfRDS4 =
      RDataFrame(RDatasetSpec({{"treeA"s, "file0.root"s}, {"treeA"s, "file1.root"s}}, {1, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS4, dfRange);

   const auto dfRDS5 =
      RDataFrame(RDatasetSpec({{"treeA"s, "file1.root"s}, {"treeB"s, "file2.root"s}}, {1, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS5, dfRange);

   // specify irregular range [6, 7) (similar to above)
   auto dfRDS8 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {6, 7})).Display()->AsString();
   // std::string dfRDS8AsString;
   // ROOT_EXPECT_ERROR(dfRDS8AsString = dfRDS8->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..6");
   EXPECT_EQ(dfRDS8, dfEmpty); // ?

   // specify irrelgular ranges (similar to above): [2, 2), [7, 7), [2, 6), [2, 0), [9, 7), [9, 2)
   const auto dfRDS9 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {2, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS9, dfEmpty);

   const auto dfRDS10 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {7, 7})).Display()->AsString();
   EXPECT_EQ(dfRDS10, dfEmpty);

   const auto dfRDS11 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {2, 7})).Display()->AsString();
   EXPECT_EQ(dfRDS11, dfRange1);

   const auto dfRDSX = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {2})).Display()->AsString();
   EXPECT_EQ(dfRDSX, dfRange2);

   auto dfRDSY = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {6})).Display()->AsString();
   // std::string dfRDSYAsString;
   // ROOT_EXPECT_ERROR(dfRDSYAsString = dfRDSY->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..5");
   EXPECT_EQ(dfRDSY, dfSimple);

   auto dfRDSZ = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {7})).Display()->AsString();
   // std::string dfRDSZAsString;
   // ROOT_EXPECT_ERROR(dfRDSZAsString = dfRDSZ->AsString(), "TTreeReader::SetEntriesRange()",
   //                   "first entry out of range 0..5");
   EXPECT_EQ(dfRDSZ, dfSimple);

   EXPECT_THROW(
      try {
         RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {9, 7});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(std::string(err.what()), "RDatasetSpec: fStartEntry cannot be larger than fEndEntry.");
         throw;
      },
      std::logic_error);
   EXPECT_THROW(
      try {
         RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {9, 2});
      } catch (const std::logic_error &err) {
         EXPECT_EQ(std::string(err.what()), "RDatasetSpec: fStartEntry cannot be larger than fEndEntry.");
         throw;
      },
      std::logic_error);

   gSystem->Exec("rm file0.root file1.root file2.root");
}

TEST(RDatasetSpecTest, Friends)
{
   // write some columns to files
   auto dfWriter0 = RDataFrame(5)
                       .Define("x", [](ULong64_t e) { return e; }, {"rdfentry_"})
                       .Define("y", [](ULong64_t e) { return e + 10; }, {"rdfentry_"})
                       .Define("z", [](ULong64_t e) { return e + 100; }, {"rdfentry_"});
   dfWriter0.Snapshot<ULong64_t>("tree", "file0.root", {"x"});
   dfWriter0.Range(4).Snapshot<ULong64_t>("tree", "file1.root", {"y"});
   dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree", "file2.root", {"z"});
   dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree", "file3.root", {"z"});
   dfWriter0.Range(4, 5).Snapshot<ULong64_t>("subTree", "file4.root", {"z"});
   dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree1", "file5.root", {"z"});
   dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree2", "file6.root", {"z"});

   // test all possible cases for specs, that are:
   // a. tree; chain where all trees have the same names; chain with different tree names
   // b. shorter and longer columns (to test cases when the friend has different size than the main change)
   // c. with and without range specified
   // hence, there are 3*2*2=12 (3 from a, 2 from b, 2 from c) cases to test in total
   std::vector<RDatasetSpec> specs;
   specs.reserve(12);
   specs.emplace_back(RDatasetSpec{"tree", "file0.root"});
   specs.emplace_back(RDatasetSpec{"tree", "file0.root", {1, 3}});
   specs.emplace_back(RDatasetSpec{"tree", "file1.root"});
   specs.emplace_back(RDatasetSpec{"tree", "file1.root", {1, 3}});
   specs.emplace_back(RDatasetSpec{"subTree", {"file2.root"s, "file3.root"s, "file4.root"s}});
   specs.emplace_back(RDatasetSpec{"subTree", {"file2.root"s, "file3.root"s, "file4.root"s}, {1, 3}});
   specs.emplace_back(RDatasetSpec{"subTree", {"file2.root"s, "file3.root"s}});
   specs.emplace_back(RDatasetSpec{"subTree", {"file2.root"s, "file3.root"s}, {1, 3}});
   specs.emplace_back(
      RDatasetSpec{{{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}}});
   specs.emplace_back(RDatasetSpec{{{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}}});
   specs.emplace_back(RDatasetSpec{{{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}}});
   specs.emplace_back(RDatasetSpec{{{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}}, {1, 3}});

   // for each spec, add all possible types of friends
   // friends cannot take range, hence valid cases are a. and b., or 3*2=6 possible friends
   for (auto &sp : specs) {
      sp.AddFriend("tree", "file0.root", "friendTree");
      sp.AddFriend("tree", "file1.root", "friendShortTree");
      sp.AddFriend("subTree", {"file2.root"s, "file3.root"s, "file4.root"s}, "friendChain1");
      sp.AddFriend("subTree", {"file2.root"s, "file3.root"s}, "friendShortChain1");
      sp.AddFriend({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}, {"subTree"s, "file4.root"s}},
                   "friendChainN");
      sp.AddFriend({{"subTree1"s, "file5.root"s}, {"subTree2"s, "file6.root"s}}, "friendShortChainN");

      // important is to construct the dataframe after all friends are added
      auto df = RDataFrame(sp);

      // lazily ask to get each column that came from a friend
      auto friendTree = df.Take<ULong64_t>("friendTree.x");
      auto friendShortTree = df.Take<ULong64_t>("friendShortTree.y");
      auto friendChain1 = df.Take<ULong64_t>("friendChain1.z");
      auto friendShortChain1 = df.Take<ULong64_t>("friendShortChain1.z");
      auto friendChainN = df.Take<ULong64_t>("friendChainN.z");
      auto friendShortChainN = df.Take<ULong64_t>("friendShortChainN.z");

      // invoke the event loop; each friend column has the same number of entries
      auto nEntries = (*friendTree).size();

      // entries being 2 correspond to application of the {1, 3} range
      for (ULong64_t i = 0u; i < nEntries; i++)
         EXPECT_EQ((*friendTree)[i], nEntries == 2 ? i + 1 : i);

      for (ULong64_t i = 0u; i < nEntries; i++)
         EXPECT_EQ((*friendChain1)[i], nEntries == 2 ? i + 101 : i + 100);

      for (ULong64_t i = 0u; i < nEntries; i++)
         EXPECT_EQ((*friendChainN)[i], nEntries == 2 ? i + 101 : i + 100);

      // the short trees/chains are repeating their last element, hence the extra case handling
      for (ULong64_t i = 0u; i < nEntries; i++)
         EXPECT_EQ((*friendShortTree)[i], (nEntries == 2) ? (i + 11) : (i == 4 ? 13 : i + 10));

      for (ULong64_t i = 0u; i < nEntries; i++)
         EXPECT_EQ((*friendShortChain1)[i], (nEntries == 2) ? (i + 101) : (i == 4 ? 103 : i + 100));

      for (ULong64_t i = 0u; i < nEntries; i++)
         EXPECT_EQ((*friendShortChainN)[i], (nEntries == 2) ? (i + 101) : (i == 4 ? 103 : i + 100));
   }

   gSystem->Exec("rm file*.root");
}

template <typename COLL>
void CheckBins(const TAxis *axis, const COLL &v)
{
   auto nBins = axis->GetNbins();
   auto nBinsp1 = nBins + 1;
   auto nBinsm1 = nBins - 1;
   EXPECT_EQ(nBinsp1, (int)v.size());
   for (auto i : ROOT::TSeqI(1, nBinsp1)) {
      EXPECT_DOUBLE_EQ(axis->GetBinLowEdge(i), (double)v[i - 1]);
   }
   EXPECT_DOUBLE_EQ(axis->GetBinUpEdge(nBinsm1), (double)v[nBinsm1]);
}

TEST(RDatasetSpecTest, Histo1D)
{
   // want to construct a chain with different tree names with such a friend chain
   auto dfWriter0 = RDataFrame(10)
                       .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                       .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
   dfWriter0.Range(5).Snapshot<double>("subTree0", "file0.root", {"x"});
   dfWriter0.Range(5, 10).Snapshot<double>("subTree1", "file1.root", {"x"});
   dfWriter0.Range(5).Snapshot<double>("subTree2", "file2.root", {"w"});
   dfWriter0.Range(5, 10).Snapshot<double>("subTree3", "file3.root", {"w"});

   RDatasetSpec spec({{"subTree0"s, "file0.root"s}, {"subTree1"s, "file1.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file2.root"s}, {"subTree3"s, "file3.root"s}});    // horizontal concatenation

   ROOT::RDataFrame d(spec);
   auto h1 = d.Histo1D(::TH1D("h1", "h1", 10, 0, 10), "x");
   auto h2 = d.Histo1D({"h2", "h2", 10, 0, 10}, "x");
   auto h1w = d.Histo1D(::TH1D("h0w", "h0w", 10, 0, 10), "x", "w");
   auto h2w = d.Histo1D({"h2w", "h2w", 10, 0, 10}, "x", "w");
   std::vector<float> edgesf{1, 2, 3, 4, 5, 6, 10};
   auto h1edgesf = d.Histo1D(::TH1D("h1edgesf", "h1edgesf", (int)edgesf.size() - 1, edgesf.data()), "x");
   auto h2edgesf = d.Histo1D({"h2edgesf", "h2edgesf", (int)edgesf.size() - 1, edgesf.data()}, "x");
   std::vector<double> edgesd{1, 2, 3, 4, 5, 6, 10};
   auto h1edgesd = d.Histo1D(::TH1D("h1edgesd", "h1edgesd", (int)edgesd.size() - 1, edgesd.data()), "x");
   auto h2edgesd = d.Histo1D({"h2edgesd", "h2edgesd", (int)edgesd.size() - 1, edgesd.data()}, "x");

   TH1DModel m0("m0", "m0", 10, 0, 10);
   TH1DModel m1(::TH1D("m1", "m1", 10, 0, 10));

   auto hm0 = d.Histo1D(m0, "x");
   auto hm1 = d.Histo1D(m1, "x");
   auto hm0w = d.Histo1D(m0, "x", "w");
   auto hm1w = d.Histo1D(m1, "x", "w");

   std::vector<double> ref({0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.});

   CheckBins(h1->GetXaxis(), ref);
   CheckBins(h2->GetXaxis(), ref);
   CheckBins(hm0->GetXaxis(), ref);
   CheckBins(hm1->GetXaxis(), ref);

   CheckBins(h1edgesf->GetXaxis(), edgesf);
   CheckBins(h2edgesf->GetXaxis(), edgesf);
   CheckBins(h1edgesd->GetXaxis(), edgesd);
   CheckBins(h2edgesd->GetXaxis(), edgesd);
}

TEST(RDatasetSpecTest, FilterDependingOnVariation)
{
   // want to construct a chain with different tree names with such a friend chain
   auto dfWriter0 = RDataFrame(10)
                       .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                       .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
   dfWriter0.Range(5).Snapshot<double>("subTree0", "file0.root", {"x"});
   dfWriter0.Range(5, 10).Snapshot<double>("subTree1", "file1.root", {"x"});
   dfWriter0.Range(5).Snapshot<double>("subTree2", "file2.root", {"w"});
   dfWriter0.Range(5, 10).Snapshot<double>("subTree3", "file3.root", {"w"});

   RDatasetSpec spec({{"subTree0"s, "file0.root"s}, {"subTree1"s, "file1.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file2.root"s}, {"subTree3"s, "file3.root"s}});    // horizontal concatenation

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

TEST(RDatasetSpecTest, SaveGraph)
{
   // want to construct a chain with different tree names with such a friend chain
   auto dfWriter0 = RDataFrame(10)
                       .Define("x", [](ULong64_t e) { return double(e); }, {"rdfentry_"})
                       .Define("w", [](ULong64_t e) { return e + 1.; }, {"rdfentry_"});
   dfWriter0.Range(5).Snapshot<double>("subTree0", "file0.root", {"x"});
   dfWriter0.Range(5, 10).Snapshot<double>("subTree1", "file1.root", {"x"});
   dfWriter0.Range(5).Snapshot<double>("subTree2", "file2.root", {"w"});
   dfWriter0.Range(5, 10).Snapshot<double>("subTree3", "file3.root", {"w"});

   RDatasetSpec spec({{"subTree0"s, "file0.root"s}, {"subTree1"s, "file1.root"s}}); // vertical concatenation
   spec.AddFriend({{"subTree2"s, "file2.root"s}, {"subTree3"s, "file3.root"s}});    // horizontal concatenation

   // actions to be shown in SaveGraph and Describe
   auto df = ROOT::RDataFrame(spec);
   auto res0 = df.Sum<double>("x");
   auto res1 = df.Sum<double>("w");

   std::string graph = ROOT::RDF::SaveGraph(df);
   static const std::string expectedGraph(
      "digraph {\n"
      "\t1 [label=<Sum>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t0 [label=<>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
      "\t2 [label=<Sum>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t0 -> 1;\n"
      "\t0 -> 2;\n"
      "}");
   EXPECT_EQ(expectedGraph, graph);

   static const std::string expectedDescribe("Dataframe from TChain  in files\n"
                                             "  file0.root\n"
                                             "  file1.root\n"
                                             "with friend\n"
                                             "  \n"
                                             "    subTree2 file2.root\n"
                                             "    subTree3 file3.root\n"
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
}