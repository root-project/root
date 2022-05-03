#include <gtest/gtest.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/TestSupport.hxx>
#include <TSystem.h>

#include <thread> // std::thread::hardware_concurrency

using namespace ROOT;
using namespace ROOT::RDF;

TEST(RDFDatasetSpec, SingleFileSingleColConstructor)
{
   auto dfWriter = RDataFrame(5).Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"});
   dfWriter.Snapshot<int>("tree", "file.root", {"x"});

   // "reference" values
   const auto dfSimple = RDataFrame("tree", "file.root").Display<int>({"x"})->AsString();
   const auto dfRange0 = RDataFrame("tree", "file.root").Range(2, 4).Display<int>({"x"})->AsString();

   // specify only tree and file names
   const auto dfRDS0 = RDataFrame(RDatasetSpec("tree", "file.root")).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS0, dfSimple);

   // specify tree name, file name, starting index => [2, 0) => range is disregarded (all entries are displayed)
   const auto dfRDS1 = RDataFrame(RDatasetSpec("tree", "file.root", 2)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS1, dfSimple);

   // specify tree name, file name, starting index, ending index, where both indices are valid => [2, 4)
   const auto dfRDS2 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS2, dfRange0);

   // specify explicitly the column
   const auto dfRDS3 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"x"})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS3, dfRange0);

   // specify explicitly the column twice (and this time not explicitly requesting 1 column from Display), this produces
   // 1 column
   const auto dfRDS4 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"x", "x"})).Display()->AsString();
   EXPECT_EQ(dfRDS4, dfRange0);

   // specify the treename as fifth argument => the first argument becomes the name of the chain of trees
   const auto dfRDS5 =
      RDataFrame(RDatasetSpec("chain", "file.root", 2, 4, {}, {"tree"})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS5, dfRange0);

   // specify the chain to have the same name as the tree
   const auto dfRDS6 =
      RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {}, {"tree"})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS6, dfRange0);

   // specify 2 trees, second tree is irrelevant, this is correct
   const auto dfRDS7 =
      RDataFrame(RDatasetSpec("chain", "file.root", 2, 4, {}, {"tree", "nottree"})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS7, dfRange0);

   // specify 2 trees, first tree is irrelevant, this is wrong, emitting C++ error and ROOT error
   EXPECT_THROW(
      try {
         ROOT_EXPECT_ERROR(RDataFrame(RDatasetSpec("chain", "file.root", 2, 4, {}, {"nottree", "tree"}))
                              .Display<int>({"x"})
                              ->AsString(),
                           "TChain::LoadTree", "Cannot find tree with name nottree in file file.root");
      } catch (const std::runtime_error &err) {
         EXPECT_EQ(std::string(err.what()),
                   "Column \"x\" is not in a dataset and is not a custom column been defined.");
         throw;
      },
      std::runtime_error);

   // specify range [3, 3) (3 is a valid index) => range is disregarded
   const auto dfRDS8 = RDataFrame(RDatasetSpec("tree", "file.root", 3, 3)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS8, dfSimple);

   // specify range [7, 7) (7 is not a valid index) => range is disregarded
   const auto dfRDS9 = RDataFrame(RDatasetSpec("tree", "file.root", 7, 7)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS9, dfSimple);

   // specify range [2, 6) (6 is not a valid index) => range becomes [2, 5)
   const auto dfRDS10 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 6)).Display<int>({"x"})->AsString();
   const auto dfRange1 = RDataFrame("tree", "file.root").Range(2, 5).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS10, dfRange1);

   // specify range [9, 7) (neither is a valid index) => range is disregarded
   const auto dfRDS11 = RDataFrame(RDatasetSpec("tree", "file.root", 9, 7)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS11, dfSimple);

   // specify range [9, 2) (9 is not a valid index) => range is disregarded
   const auto dfRDS12 = RDataFrame(RDatasetSpec("tree", "file.root", 9, 2)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS12, dfSimple);

   // specify range [5, 6) (neither is a valid index, but 5 < 6) => despite the ROOT error, assignment is made
   // disregarding the range
   auto dfRDS13 = RDataFrame(RDatasetSpec("tree", "file.root", 5, 6)).Display<int>({"x"});
   std::string dfRDS13AsString;
   ROOT_EXPECT_ERROR(dfRDS13AsString = dfRDS13->AsString(), "TTreeReader::SetEntriesRange()",
                     "first entry out of range 0..5");
   EXPECT_EQ(dfRDS13AsString, dfSimple);

   // specify 2 columns, second column is irrelevant, this is correct
   const auto dfRDS14 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"x", "y"})).Display()->AsString();
   EXPECT_EQ(dfRDS14, dfRange0);

   // specify 2 columns, first column is irrelevant, this is also correct
   const auto dfRDS15 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"y", "x"})).Display()->AsString();
   EXPECT_EQ(dfRDS15, dfRange0);

   using namespace std::literals; // remove ambiguity of using std::vector<std::string>-s and std::string-s

   // test the second constructor, second argument is now a vector
   const auto dfRDS16 = RDataFrame(RDatasetSpec("tree", {"file.root"s})).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS16, dfSimple);

   // test the second constructor, second argument is now a vector, specify range as well
   const auto dfRDS17 = RDataFrame(RDatasetSpec("tree", {"file.root"s}, 2, 4)).Display<int>({"x"})->AsString();
   EXPECT_EQ(dfRDS17, dfRange0);

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
   const auto dfRange = RDataFrame("tree", "file.root").Range(2, 4).Display<int, int>({"x", "y"})->AsString();
   const auto dfRangeN = RDataFrame("tree", "file.root").Range(2, 5).Display<int, int>({"x", "y"})->AsString();

   // specify only tree and file names, do not pass arguments to Display to show all available columns
   const auto dfRDS0 = RDataFrame(RDatasetSpec("tree", "file.root")).Display()->AsString();
   EXPECT_EQ(dfRDS0, dfSimple);

   // specify meaningful ranges and columns
   const auto dfRDS1 = RDataFrame(RDatasetSpec("tree", "file.root", 0, 5, {"x", "y"})).Display()->AsString();
   const auto dfRDS2 = RDataFrame(RDatasetSpec("tree", "file.root", 0, 5, {"x"})).Display()->AsString();
   const auto dfRDS3 = RDataFrame(RDatasetSpec("tree", "file.root", 0, 5, {"y"})).Display()->AsString();
   const auto dfRDS4 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4)).Display()->AsString();
   const auto dfRDS5 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"x", "y"})).Display()->AsString();
   const auto dfRDS6 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"x"})).Display()->AsString();
   const auto dfRDS7 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 4, {"y"})).Display()->AsString();
   EXPECT_EQ(dfRDS1, dfSimple);
   EXPECT_EQ(dfRDS2, dfSimple); // `Display` picks columns from `columnNameRegexp`, hence all
   EXPECT_EQ(dfRDS3, dfSimple);
   EXPECT_EQ(dfRDS4, dfRange);
   EXPECT_EQ(dfRDS5, dfRange);
   EXPECT_EQ(dfRDS6, dfRange);
   EXPECT_EQ(dfRDS7, dfRange);

   // specify irrelgular ranges (similar to above): [2, 0), [3, 3), [7, 7), [2, 6), [9, 7), [9, 2)
   const auto dfRDS8 = RDataFrame(RDatasetSpec("tree", "file.root", 2)).Display()->AsString();
   const auto dfRDS9 = RDataFrame(RDatasetSpec("tree", "file.root", 3, 3)).Display()->AsString();
   const auto dfRDS10 = RDataFrame(RDatasetSpec("tree", "file.root", 7, 7)).Display()->AsString();
   const auto dfRDS11 = RDataFrame(RDatasetSpec("tree", "file.root", 2, 6)).Display()->AsString();
   const auto dfRDS12 = RDataFrame(RDatasetSpec("tree", "file.root", 9, 7)).Display()->AsString();
   const auto dfRDS13 = RDataFrame(RDatasetSpec("tree", "file.root", 9, 2)).Display()->AsString();
   EXPECT_EQ(dfRDS8, dfSimple);
   EXPECT_EQ(dfRDS9, dfSimple);
   EXPECT_EQ(dfRDS10, dfSimple);
   EXPECT_EQ(dfRDS11, dfRangeN);
   EXPECT_EQ(dfRDS12, dfSimple);
   EXPECT_EQ(dfRDS13, dfSimple);

   // specify irregular range [5, 6) (similar to above)
   auto dfRDS14 = RDataFrame(RDatasetSpec("tree", "file.root", 5, 6)).Display();
   std::string dfRDS14AsString;
   ROOT_EXPECT_ERROR(dfRDS14AsString = dfRDS14->AsString(), "TTreeReader::SetEntriesRange()",
                     "first entry out of range 0..5");
   EXPECT_EQ(dfRDS14AsString, dfSimple);

   using namespace std::literals; // remove ambiguity of using std::vector<std::string>-s and std::string-s

   // test the second constructor, second argument is now a vector
   const auto dfRDS15 = RDataFrame(RDatasetSpec("tree", {"file.root"s})).Display()->AsString();
   EXPECT_EQ(dfRDS15, dfSimple);

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

   using namespace std::literals;

   // both files have the same tree, do not ask for chain
   const auto dfRDS0 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s})).Display()->AsString();
   EXPECT_EQ(dfRDS0, dfSimple);

   // both files have the same tree, but ask for chain
   const auto dfRDS1 = RDataFrame(RDatasetSpec("chain", {"file0.root"s, "file1.root"s}, 0, 0, {}, {"treeA", "treeA"}))
                          .Display()
                          ->AsString();
   EXPECT_EQ(dfRDS1, dfSimple);

   // files have different chain name => need a chain
   const auto dfRDS2 = RDataFrame(RDatasetSpec("chain", {"file1.root"s, "file2.root"s}, 0, 0, {}, {"treeA", "treeB"}))
                          .Display()
                          ->AsString();
   EXPECT_EQ(dfRDS2, dfSimple);

   // cases similar to above, but now range is applied, note that the range is global (i.e. not per tree, but per chain)
   const auto dfRDS3 = RDataFrame(RDatasetSpec("treeA", {"file0.root"s, "file1.root"s}, 1, 2)).Display()->AsString();
   const auto dfRDS4 = RDataFrame(RDatasetSpec("chain", {"file0.root"s, "file1.root"s}, 1, 2, {}, {"treeA", "treeA"}))
                          .Display()
                          ->AsString();
   const auto dfRDS5 = RDataFrame(RDatasetSpec("chain", {"file1.root"s, "file2.root"s}, 1, 2, {}, {"treeA", "treeB"}))
                          .Display()
                          ->AsString();
   EXPECT_EQ(dfRDS3, dfRange);
   EXPECT_EQ(dfRDS4, dfRange);
   EXPECT_EQ(dfRDS5, dfRange);

   gSystem->Exec("rm file0.root file1.root file2.root");
}
