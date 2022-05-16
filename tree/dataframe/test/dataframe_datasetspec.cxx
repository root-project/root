#include <gtest/gtest.h>
#include <ROOT/RDataFrame.hxx>
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

   // specify 2 trees, second tree is irrelevant, this is correct
   const auto dfRDS6 =
      RDataFrame(RDatasetSpec({"tree"s, "nottree"s}, {"file.root"s}, {2, 4})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS6, dfRange0);

   // specify 2 trees, first tree is irrelevant, this is wrong, emitting C++ error and ROOT error
   EXPECT_THROW(
      try {
         ROOT_EXPECT_ERROR(
            RDataFrame(RDatasetSpec({"nottree"s, "tree"s}, {"file.root"s}, {2, 4})).Display<int>({"x"})->AsString(),
            "TChain::LoadTree", "Cannot find tree with name nottree in file file.root");
      } catch (const std::runtime_error &err) {
         EXPECT_EQ(std::string(err.what()),
                   "Column \"x\" is not in a dataset and is not a custom column been defined.");
         throw;
      },
      std::runtime_error);

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
      RDataFrame(RDatasetSpec({"treeA"s, "treeA"s}, {"file0.root"s, "file1.root"s}, {0, 5})).Display()->AsString();
   EXPECT_EQ(dfRDS1, dfSimple);

   // files have different chain name => need a chain
   const auto dfRDS2 =
      RDataFrame(RDatasetSpec({"treeA"s, "treeB"s}, {"file1.root"s, "file2.root"s}, {0, 5})).Display()->AsString();
   EXPECT_EQ(dfRDS2, dfSimple);

   // cases similar to above, but now range is applied, note that the range is global (i.e. not per tree, but per chain)
   const auto dfRDS3 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, {1, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS3, dfRange);

   const auto dfRDS4 =
      RDataFrame(RDatasetSpec({"treeA"s, "treeA"s}, {"file0.root"s, "file1.root"s}, {1, 2})).Display()->AsString();
   EXPECT_EQ(dfRDS4, dfRange);

   const auto dfRDS5 =
      RDataFrame(RDatasetSpec({"treeA"s, "treeB"s}, {"file1.root"s, "file2.root"s}, {1, 2})).Display()->AsString();
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
