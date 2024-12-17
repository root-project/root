#include <TChain.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/TestSupport.hxx>

#include <gtest/gtest.h>

#include <numeric> // std::numeric_limits
#include <random>
#include <string>
#include <thread>  // std::thread::hardware_concurrency
#include <vector>

template <typename T0, typename T1>
void expect_vec_eq(const T0 &v1, const T1 &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST(RDefaultValueFor, EmptyDataSourceAndNonExistingColumnName)
{
   // We disable the functionality in case of empty data source and no previously
   // defined column
   ROOT::RDataFrame df{1};
   EXPECT_THROW(
      try { df.DefaultValueFor("rndm", 1); } catch (const std::runtime_error &err) {
         const auto msg = "RDataFrame::DefaultValueFor: cannot provide default values for column \"rndm\". No column "
                          "with that name was found in the dataset. Use Define to create a new column.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

TEST(RDefaultValueFor, VaryDefaultValueFor)
{
   // Vary a column then try to provide a default value --> not legal since
   // the first Vary operation implies the column always has a value.
   ROOT::RDataFrame df{1};
   EXPECT_THROW(
      try {
         auto df1 = df.Define("x", [] { return 42; }).Vary("x", [] { return ROOT::RVecI{32, 52}; }, {}, {"up", "down"});
         df1.DefaultValueFor("x", 100);
      } catch (const std::runtime_error &err) {
         constexpr auto msg{
            "RDataFrame::DefaultValueFor: cannot provide a default value for column \"x\". The column depends on one "
            "or more systematic variations and it should not be possible to have missing values in varied columns."};
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

TEST(FilterWithMissingValues, EmptyDataSourceAndNonExistingColumnName)
{
   // We disable the functionality in case of empty data source and no previously
   // defined column
   ROOT::RDataFrame df{1};
   EXPECT_THROW(
      try { df.FilterAvailable("rndm"); } catch (const std::runtime_error &err) {
         const auto msg = "Unknown column: \"rndm\"";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);

   EXPECT_THROW(
      try { df.FilterMissing("rndm"); } catch (const std::runtime_error &err) {
         const auto msg = "Unknown column: \"rndm\"";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

struct RDFMissingBranches : public ::testing::TestWithParam<bool> {
   constexpr static std::array<const char *, 3> fFileNames{"dataframe_incomplete_entries_missing_branches_1.root",
                                                           "dataframe_incomplete_entries_missing_branches_2.root",
                                                           "dataframe_incomplete_entries_missing_branches_3.root"};
   constexpr static std::array<const char *, 3> fTreeNames{"tree_1", "tree_2", "tree_3"};
   constexpr static auto fTreeEntries{5};
   const unsigned int fNSlots;

   RDFMissingBranches() : fNSlots(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(fNSlots);
   }

   ~RDFMissingBranches() override
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }

   static void SetUpTestSuite()
   {
      {
         TFile f(fFileNames[0], "RECREATE");
         TTree t(fTreeNames[0], fTreeNames[0]);
         int x{};
         int y{};
         t.Branch("x", &x, "x/I");
         t.Branch("y", &y, "y/I");
         for (int i = 1; i <= fTreeEntries; i++) {
            x = i;
            y = 2 * i;
            t.Fill();
            t.FlushBaskets();
         }
         t.Write();
      }

      {
         TFile f(fFileNames[1], "RECREATE");
         TTree t(fTreeNames[1], fTreeNames[1]);
         int y{};
         t.Branch("y", &y, "y/I");
         for (int i = 1; i <= fTreeEntries; i++) {
            y = 3 * i;
            t.Fill();
            t.FlushBaskets();
         }
         t.Write();
      }

      {
         TFile f(fFileNames[2], "RECREATE");
         TTree t(fTreeNames[2], fTreeNames[2]);
         int x{};
         t.Branch("x", &x, "x/I");
         for (int i = 1; i <= fTreeEntries; i++) {
            x = 4 * i;
            t.Fill();
            t.FlushBaskets();
         }
         t.Write();
      }
   }

   static void TearDownTestSuite()
   {
      for (auto &&fileName : fFileNames)
         std::remove(fileName);
   }
};

TEST_P(RDFMissingBranches, TwoMissingBranches)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];
   std::string fullpath_3 = std::string(fFileNames[2]) + "?#" + fTreeNames[2];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());
   c.Add(fullpath_3.c_str());

   ROOT::RDataFrame df{c};

   std::vector<int> expected_withdefault_x{1, 2, 3, 4, 5, 42, 42, 42, 42, 42, 4, 8, 12, 16, 20};
   auto take_withdefault_x = df.DefaultValueFor("x", 42).Take<int>("x");

   std::vector<int> expected_skip_x{1, 2, 3, 4, 5, 4, 8, 12, 16, 20};
   auto take_skip_x = df.FilterAvailable("x").Take<int>("x");

   std::vector<int> expected_withdefault_y{2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 99, 99, 99, 99, 99};
   auto take_withdefault_y = df.DefaultValueFor("y", 99).Take<int>("y");

   std::vector<int> expected_skip_y{2, 4, 6, 8, 10, 3, 6, 9, 12, 15};
   auto take_skip_y = df.FilterAvailable("y").Take<int>("y");

   std::vector<int> expected_missing_y{4, 8, 12, 16, 20};
   auto take_keep_missing_y = df.FilterMissing("y").Take<int>("x");

   if (GetParam()) {
      // MT processing scrumbles result order
      auto take_withdefault_x_val = *take_withdefault_x;
      auto take_skip_x_val = *take_skip_x;
      auto take_withdefault_y_val = *take_withdefault_y;
      auto take_skip_y_val = *take_skip_y;
      auto take_keep_missing_y_val = *take_keep_missing_y;

      // Sort expected vectors
      std::sort(expected_withdefault_x.begin(), expected_withdefault_x.end());
      std::sort(expected_skip_x.begin(), expected_skip_x.end());
      std::sort(expected_withdefault_y.begin(), expected_withdefault_y.end());
      std::sort(expected_skip_y.begin(), expected_skip_y.end());
      std::sort(expected_missing_y.begin(), expected_missing_y.end());

      // Sort result vectors
      std::sort(take_withdefault_x_val.begin(), take_withdefault_x_val.end());
      std::sort(take_skip_x_val.begin(), take_skip_x_val.end());
      std::sort(take_withdefault_y_val.begin(), take_withdefault_y_val.end());
      std::sort(take_skip_y_val.begin(), take_skip_y_val.end());
      std::sort(take_keep_missing_y_val.begin(), take_keep_missing_y_val.end());

      expect_vec_eq(expected_withdefault_x, take_withdefault_x_val);
      expect_vec_eq(expected_skip_x, take_skip_x_val);
      expect_vec_eq(expected_withdefault_y, take_withdefault_y_val);
      expect_vec_eq(expected_skip_y, take_skip_y_val);
      expect_vec_eq(expected_missing_y, take_keep_missing_y_val);
   } else {
      expect_vec_eq(expected_withdefault_x, *take_withdefault_x);
      expect_vec_eq(expected_skip_x, *take_skip_x);
      expect_vec_eq(expected_withdefault_y, *take_withdefault_y);
      expect_vec_eq(expected_skip_y, *take_skip_y);
      expect_vec_eq(expected_missing_y, *take_keep_missing_y);
   }
}

TEST_P(RDFMissingBranches, MissingBranchFirstTree)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];
   std::string fullpath_3 = std::string(fFileNames[2]) + "?#" + fTreeNames[2];

   c.Add(fullpath_2.c_str());
   c.Add(fullpath_1.c_str());
   c.Add(fullpath_3.c_str());

   ROOT::RDataFrame df{c};

   std::vector<int> expected_withdefault{42, 42, 42, 42, 42, 1, 2, 3, 4, 5, 4, 8, 12, 16, 20};
   auto take_withdefault = df.DefaultValueFor("x", 42).Take<int>("x");

   std::vector<int> expected_skip{1, 2, 3, 4, 5, 4, 8, 12, 16, 20};
   auto take_skip = df.FilterAvailable("x").Take<int>("x");

   if (GetParam()) {
      auto take_withdefault_val = *take_withdefault;
      auto take_skip_val = *take_skip;
      // Sort expected vectors
      std::sort(expected_withdefault.begin(), expected_withdefault.end());
      std::sort(expected_skip.begin(), expected_skip.end());
      // Sort result vectors
      std::sort(take_withdefault_val.begin(), take_withdefault_val.end());
      std::sort(take_skip_val.begin(), take_skip_val.end());
      expect_vec_eq(expected_withdefault, take_withdefault_val);
      expect_vec_eq(expected_skip, take_skip_val);
   } else {
      expect_vec_eq(expected_withdefault, *take_withdefault);
      expect_vec_eq(expected_skip, *take_skip);
   }
}

TEST_P(RDFMissingBranches, MissingBranchAllTrees)
{
   TChain c{};
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];

   c.Add(fullpath_2.c_str());
   c.Add(fullpath_2.c_str());

   ROOT::RDataFrame df{c};

   std::vector<int> expected_withdefault{42, 42, 42, 42, 42, 42, 42, 42, 42, 42};
   auto take_withdefault = df.DefaultValueFor("x", 42).Take<int>("x");

   std::vector<int> expected_skip{};
   auto take_skip = df.FilterAvailable("x").Take<int>("x");

   if (GetParam()) {
      auto take_withdefault_val = *take_withdefault;
      auto take_skip_val = *take_skip;
      // Sort expected vectors
      std::sort(expected_withdefault.begin(), expected_withdefault.end());
      std::sort(expected_skip.begin(), expected_skip.end());
      // Sort result vectors
      std::sort(take_withdefault_val.begin(), take_withdefault_val.end());
      std::sort(take_skip_val.begin(), take_skip_val.end());
      expect_vec_eq(expected_withdefault, take_withdefault_val);
      expect_vec_eq(expected_skip, take_skip_val);
   } else {
      expect_vec_eq(expected_withdefault, *take_withdefault);
      expect_vec_eq(expected_skip, *take_skip);
   }
}

TEST_P(RDFMissingBranches, DefaultValueForAndFilterAvailable)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];
   std::string fullpath_3 = std::string(fFileNames[2]) + "?#" + fTreeNames[2];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());
   c.Add(fullpath_3.c_str());

   ROOT::RDataFrame df{c};

   constexpr static auto maxint = std::numeric_limits<int>::max();

   std::vector<int> expected_1{1, 2, 3, 4, 5, maxint, maxint, maxint, maxint, maxint, 4, 8, 12, 16, 20};
   auto take_1 = df.DefaultValueFor("x", maxint).FilterAvailable("x").Take<int>("x");

   std::vector<int> expected_2{1, 2, 3, 4, 5, 4, 8, 12, 16, 20};
   auto take_2 = df.FilterAvailable("x").DefaultValueFor("x", maxint).Take<int>("x");

   if (GetParam()) {
      auto take_1_val = *take_1;
      auto take_2_val = *take_2;
      // Sort expected vectors
      std::sort(expected_1.begin(), expected_1.end());
      std::sort(expected_2.begin(), expected_2.end());
      // Sort result vectors
      std::sort(take_1_val.begin(), take_1_val.end());
      std::sort(take_2_val.begin(), take_2_val.end());

      expect_vec_eq(expected_1, take_1_val);
      expect_vec_eq(expected_2, take_2_val);
   } else {
      expect_vec_eq(expected_1, *take_1);
      expect_vec_eq(expected_2, *take_2);
   }
}

TEST_P(RDFMissingBranches, DefaultValueForAndFilterMissing)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];
   std::string fullpath_3 = std::string(fFileNames[2]) + "?#" + fTreeNames[2];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());
   c.Add(fullpath_3.c_str());

   ROOT::RDataFrame df{c};

   constexpr static auto maxint = std::numeric_limits<int>::max();

   std::vector<int> expected_1{};
   auto take_1 = df.DefaultValueFor("x", maxint).FilterMissing("x").Take<int>("x");

   std::vector<int> expected_2{maxint, maxint, maxint, maxint, maxint};
   auto take_2 = df.FilterMissing("x").DefaultValueFor("x", maxint).Take<int>("x");

   if (GetParam()) {
      auto take_1_val = *take_1;
      auto take_2_val = *take_2;
      // Sort expected vectors
      std::sort(expected_1.begin(), expected_1.end());
      std::sort(expected_2.begin(), expected_2.end());
      // Sort result vectors
      std::sort(take_1_val.begin(), take_1_val.end());
      std::sort(take_2_val.begin(), take_2_val.end());

      expect_vec_eq(expected_1, take_1_val);
      expect_vec_eq(expected_2, take_2_val);
   } else {
      expect_vec_eq(expected_1, *take_1);
      expect_vec_eq(expected_2, *take_2);
   }
}

TEST_P(RDFMissingBranches, MissingValueExceptionAction)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());

   // Various error message that may be printed by TTreeReader depending on
   // whether we are running with IMT or not
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(
      kError, "TTreeReaderValueBase::CreateProxy()",
      "The tree does not have a branch called x. You could check with TTree::Print() for available branches.");
   diagRAII.optionalDiag(kError, "TBranchProxy::Setup()",
                         std::string("Branch 'x' is not available from tree '") + fTreeNames[1] + "' in file '" +
                            fFileNames[1] + "'.");
   diagRAII.optionalDiag(kError, "TTreeReader::SetEntryBase()", "There was an error while notifying the proxies.");

   // Exception from action
   EXPECT_THROW(
      try {
         ROOT::RDataFrame df{c};
         *df.Take<int>("x");
      } catch (const std::out_of_range &err) {
         const std::string errmsg{err.what()};
         EXPECT_TRUE(errmsg.find("Action (Take) could not retrieve value for column 'x' for entry") !=
                     std::string::npos)
            << "Actual error message: " << errmsg;
         throw;
      },
      std::out_of_range);
}

TEST_P(RDFMissingBranches, MissingValueExceptionDefine)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());

   // Various error message that may be printed by TTreeReader depending on
   // whether we are running with IMT or not
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(
      kError, "TTreeReaderValueBase::CreateProxy()",
      "The tree does not have a branch called x. You could check with TTree::Print() for available branches.");
   diagRAII.optionalDiag(kError, "TBranchProxy::Setup()",
                         std::string("Branch 'x' is not available from tree '") + fTreeNames[1] + "' in file '" +
                            fFileNames[1] + "'.");
   diagRAII.optionalDiag(kError, "TTreeReader::SetEntryBase()", "There was an error while notifying the proxies.");

   // Exception from Define
   EXPECT_THROW(
      try {
         ROOT::RDataFrame df{c};
         auto def = df.Define("newcol", [](int x) { return x * 2; }, {"x"});
         *def.Take<int>("newcol");
      } catch (const std::out_of_range &err) {
         const std::string errmsg{err.what()};
         EXPECT_TRUE(errmsg.find("Define could not retrieve value for column 'x' for entry") != std::string::npos)
            << "Actual error message: " << errmsg;
         throw;
      },
      std::out_of_range);
}

TEST_P(RDFMissingBranches, MissingValueExceptionFilter)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());

   // Various error message that may be printed by TTreeReader depending on
   // whether we are running with IMT or not
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(
      kError, "TTreeReaderValueBase::CreateProxy()",
      "The tree does not have a branch called x. You could check with TTree::Print() for available branches.");
   diagRAII.optionalDiag(kError, "TBranchProxy::Setup()",
                         std::string("Branch 'x' is not available from tree '") + fTreeNames[1] + "' in file '" +
                            fFileNames[1] + "'.");
   diagRAII.optionalDiag(kError, "TTreeReader::SetEntryBase()", "There was an error while notifying the proxies.");

   // Exception from filter
   EXPECT_THROW(
      try {
         ROOT::RDataFrame df{c};
         auto filt = df.Filter([](int x) { return x < 2; }, {"x"});
         *filt.Take<int>("x");
      } catch (const std::out_of_range &err) {
         const std::string errmsg{err.what()};
         EXPECT_TRUE(errmsg.find("Filter could not retrieve value for column 'x' for entry") != std::string::npos)
            << "Actual error message: " << errmsg;
         throw;
      },
      std::out_of_range);
}

TEST_P(RDFMissingBranches, MissingValueExceptionVariedAction)
{
   TChain c{};
   std::string fullpath_1 = std::string(fFileNames[0]) + "?#" + fTreeNames[0];
   std::string fullpath_2 = std::string(fFileNames[1]) + "?#" + fTreeNames[1];

   c.Add(fullpath_1.c_str());
   c.Add(fullpath_2.c_str());

   // Various error message that may be printed by TTreeReader depending on
   // whether we are running with IMT or not
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(
      kError, "TTreeReaderValueBase::CreateProxy()",
      "The tree does not have a branch called x. You could check with TTree::Print() for available branches.");
   diagRAII.optionalDiag(kError, "TBranchProxy::Setup()",
                         std::string("Branch 'x' is not available from tree '") + fTreeNames[1] + "' in file '" +
                            fFileNames[1] + "'.");
   diagRAII.optionalDiag(kError, "TTreeReader::SetEntryBase()", "There was an error while notifying the proxies.");

   // Exception with a varied action
   EXPECT_THROW(
      try {
         ROOT::RDataFrame df{c};
         auto vary = df.Vary("x",
                             [](int x) {
                                return ROOT::RVecI{-1 * x, 2 * x};
                             },
                             {"x"}, {"down", "up"})
                        .Take<int>("x");
         auto variations = ROOT::RDF::Experimental::VariationsFor(vary);
         variations["x:down"];
      } catch (const std::out_of_range &err) {
         const std::string errmsg{err.what()};
         // The nominal action is the first one encountering the missing value, so it will throw the exception first
         EXPECT_TRUE(errmsg.find("Action (Take) could not retrieve value for column 'x' for entry") !=
                     std::string::npos)
            << "Actual error message: " << errmsg;
         throw;
      },
      std::out_of_range);
}

struct RDFIndexNoMatchSimple : public ::testing::TestWithParam<std::pair<bool, bool>> {
   constexpr static auto fMainFile{"main.root"};
   constexpr static auto fAuxFile1{"aux_1.root"};
   constexpr static auto fAuxFile2{"aux_2.root"};
   constexpr static auto fMainTreeName{"events"};
   constexpr static auto fAuxTreeName1{"auxdata_1"};
   constexpr static auto fAuxTreeName2{"auxdata_2"};
   const unsigned int fNSlots;

   RDFIndexNoMatchSimple() : fNSlots(GetParam().first ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam().first) {
         ROOT::EnableImplicitMT(fNSlots);
      }
   }

   ~RDFIndexNoMatchSimple()
   {
      if (GetParam().first)
         ROOT::DisableImplicitMT();
   }

   static void SetUpTestSuite()
   {
      {
         // Main dataset:
         // idx: 1, 2, 3, 4, 5
         // x: 1, 2, 3, 4, 5
         // 1 entry per cluster to better exercise MT runs
         TFile f(fMainFile, "RECREATE");
         TTree mainTree(fMainTreeName, fMainTreeName);
         int idx;
         int x;
         mainTree.Branch("idx", &idx);
         mainTree.Branch("x", &x);

         std::vector<int> idxs{1, 2, 3, 4, 5};
         std::vector<int> xs = idxs;
         for (std::size_t i = 0; i < idxs.size(); i++) {
            idx = idxs[i];
            x = xs[i];
            mainTree.Fill();
            mainTree.FlushBaskets();
         }

         mainTree.Write();
      }
      {
         // First auxiliary dataset
         // idx: 1, 2
         // y: 11, 22
         // 1 entry per cluster to better exercise MT runs
         TFile f(fAuxFile1, "RECREATE");
         TTree auxTree(fAuxTreeName1, fAuxTreeName1);
         int y;
         int idx;
         auxTree.Branch("idx", &idx);
         auxTree.Branch("y", &y);

         std::vector<int> idxs{1, 2};
         std::vector<int> ys = {11, 22};
         for (std::size_t i = 0; i < idxs.size(); i++) {
            idx = idxs[i];
            y = ys[i];
            auxTree.Fill();
            auxTree.FlushBaskets();
         }

         auxTree.Write();
      }
      {
         // Second auxiliary dataset:
         // idx: 1, 3, 5
         // z: 111, 333, 555
         // 1 entry per cluster to better exercise MT runs
         TFile f(fAuxFile2, "RECREATE");
         TTree auxTree(fAuxTreeName2, fAuxTreeName2);
         int z;
         int idx;
         auxTree.Branch("idx", &idx);
         auxTree.Branch("z", &z);

         std::vector<int> idxs{1, 3, 5};
         std::vector<int> zs = {111, 333, 555};
         for (std::size_t i = 0; i < idxs.size(); i++) {
            idx = idxs[i];
            z = zs[i];
            auxTree.Fill();
            auxTree.FlushBaskets();
         }

         auxTree.Write();
      }
   }

   static void TearDownTestSuite()
   {
      std::remove(fMainFile);
      std::remove(fAuxFile1);
      std::remove(fAuxFile2);
   }
};

void testAllDefaults(ROOT::RDF::RNode df, bool imt)
{
   // Provide default values for all columns that might have missing values
   auto dfAllDefaults = df.DefaultValueFor("y", -2).DefaultValueFor("z", -4);

   // Take values of columns from all datasets
   std::vector<int> expected_x{1, 2, 3, 4, 5};
   std::vector<int> expected_y{11, 22, -2, -2, -2};
   std::vector<int> expected_z{111, -4, 333, -4, 555};

   auto take_x = dfAllDefaults.Take<int>("x");
   auto take_y = dfAllDefaults.Take<int>("y");
   auto take_z = dfAllDefaults.Take<int>("z");

   if (imt) {
      auto take_x_val = *take_x;
      auto take_y_val = *take_y;
      auto take_z_val = *take_z;
      std::sort(expected_x.begin(), expected_x.end());
      std::sort(expected_y.begin(), expected_y.end());
      std::sort(expected_z.begin(), expected_z.end());
      std::sort(take_x_val.begin(), take_x_val.end());
      std::sort(take_y_val.begin(), take_y_val.end());
      std::sort(take_z_val.begin(), take_z_val.end());

      expect_vec_eq(expected_x, take_x_val);
      expect_vec_eq(expected_y, take_y_val);
      expect_vec_eq(expected_z, take_z_val);
   } else {
      expect_vec_eq(expected_x, *take_x);
      expect_vec_eq(expected_y, *take_y);
      expect_vec_eq(expected_z, *take_z);
   }
}

TEST_P(RDFIndexNoMatchSimple, AllDefaults)
{
   const auto &[imt, ttree] = GetParam();

   if (ttree) {
      std::unique_ptr<TFile> mainfile{TFile::Open(fMainFile)};
      std::unique_ptr<TTree> maintree{mainfile->Get<TTree>(fMainTreeName)};

      std::unique_ptr<TFile> auxfile1{TFile::Open(fAuxFile1)};
      std::unique_ptr<TTree> auxtree1{auxfile1->Get<TTree>(fAuxTreeName1)};
      auxtree1->BuildIndex("idx");

      std::unique_ptr<TFile> auxfile2{TFile::Open(fAuxFile2)};
      std::unique_ptr<TTree> auxtree2{auxfile2->Get<TTree>(fAuxTreeName2)};
      auxtree2->BuildIndex("idx");

      maintree->AddFriend(auxtree1.get());
      maintree->AddFriend(auxtree2.get());
      testAllDefaults(ROOT::RDataFrame{*maintree}, imt);
   } else {
      TChain mainChain{fMainTreeName};
      mainChain.Add(fMainFile);

      TChain auxChain1(fAuxTreeName1);
      auxChain1.Add(fAuxFile1);
      auxChain1.BuildIndex("idx");

      TChain auxChain2(fAuxTreeName2);
      auxChain2.Add(fAuxFile2);
      auxChain2.BuildIndex("idx");

      mainChain.AddFriend(&auxChain1);
      mainChain.AddFriend(&auxChain2);

      testAllDefaults(ROOT::RDataFrame{mainChain}, imt);
   }
}

void testSkipAux1KeepAux2(ROOT::RDF::RNode df, bool imt)
{
   // Discard missing values from the first auxiliary tree but keep those from the second
   auto dfSkipAux1KeepAux2 = df.FilterAvailable("y").DefaultValueFor("z", -4);

   // Take values of columns from all datasets
   std::vector<int> expected_x{1, 2};
   std::vector<int> expected_y{11, 22};
   std::vector<int> expected_z{111, -4};

   auto take_x = dfSkipAux1KeepAux2.Take<int>("x");
   auto take_y = dfSkipAux1KeepAux2.Take<int>("y");
   auto take_z = dfSkipAux1KeepAux2.Take<int>("z");

   if (imt) {
      auto take_x_val = *take_x;
      auto take_y_val = *take_y;
      auto take_z_val = *take_z;
      std::sort(expected_x.begin(), expected_x.end());
      std::sort(expected_y.begin(), expected_y.end());
      std::sort(expected_z.begin(), expected_z.end());
      std::sort(take_x_val.begin(), take_x_val.end());
      std::sort(take_y_val.begin(), take_y_val.end());
      std::sort(take_z_val.begin(), take_z_val.end());

      expect_vec_eq(expected_x, take_x_val);
      expect_vec_eq(expected_y, take_y_val);
      expect_vec_eq(expected_z, take_z_val);
   } else {
      expect_vec_eq(expected_x, *take_x);
      expect_vec_eq(expected_y, *take_y);
      expect_vec_eq(expected_z, *take_z);
   }
}

TEST_P(RDFIndexNoMatchSimple, SkipAux1KeepAux2)
{
   const auto &[imt, ttree] = GetParam();

   if (ttree) {
      std::unique_ptr<TFile> mainfile{TFile::Open(fMainFile)};
      std::unique_ptr<TTree> maintree{mainfile->Get<TTree>(fMainTreeName)};

      std::unique_ptr<TFile> auxfile1{TFile::Open(fAuxFile1)};
      std::unique_ptr<TTree> auxtree1{auxfile1->Get<TTree>(fAuxTreeName1)};
      auxtree1->BuildIndex("idx");

      std::unique_ptr<TFile> auxfile2{TFile::Open(fAuxFile2)};
      std::unique_ptr<TTree> auxtree2{auxfile2->Get<TTree>(fAuxTreeName2)};
      auxtree2->BuildIndex("idx");

      maintree->AddFriend(auxtree1.get());
      maintree->AddFriend(auxtree2.get());
      testSkipAux1KeepAux2(ROOT::RDataFrame{*maintree}, imt);
   } else {
      TChain mainChain{fMainTreeName};
      mainChain.Add(fMainFile);

      TChain auxChain1(fAuxTreeName1);
      auxChain1.Add(fAuxFile1);
      auxChain1.BuildIndex("idx");

      TChain auxChain2(fAuxTreeName2);
      auxChain2.Add(fAuxFile2);
      auxChain2.BuildIndex("idx");

      mainChain.AddFriend(&auxChain1);
      mainChain.AddFriend(&auxChain2);

      testSkipAux1KeepAux2(ROOT::RDataFrame{mainChain}, imt);
   }
}

struct RDFMissingBranchesVec : public ::testing::TestWithParam<bool> {
   constexpr static std::array<const char *, 3> fFileNames{"dataframe_incomplete_entries_missing_branches_vec_1.root",
                                                           "dataframe_incomplete_entries_missing_branches_vec_2.root",
                                                           "dataframe_incomplete_entries_missing_branches_vec_3.root"};
   constexpr static auto fTreeName{"events"};
   constexpr static auto fTreeEntries{5};
   const unsigned int fNSlots;

   RDFMissingBranchesVec() : fNSlots(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(fNSlots);
   }

   ~RDFMissingBranchesVec() override
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }

   static void SetUpTestSuite()
   {
      {
         TFile f(fFileNames[0], "RECREATE");
         TTree t(fTreeName, fTreeName);
         std::vector<std::vector<int>> xs{{1}, {2, 3}, {4, 5, 6}, {7, 8, 9, 10}, {11, 12, 13, 14, 15}};
         std::vector<std::vector<int>> ys{{11, 12, 13, 14, 15}, {7, 8, 9, 10}, {4, 5, 6}, {2, 3}, {1}};
         std::vector<int> vecx;
         std::vector<int> vecy;

         t.Branch("vecx", &vecx);
         t.Branch("vecy", &vecy);

         for (int i = 0; i < fTreeEntries; i++) {
            vecx = xs[i];
            vecy = ys[i];
            t.Fill();
            t.FlushBaskets();
         }

         t.Write();
      }

      {
         TFile f(fFileNames[1], "RECREATE");
         TTree t(fTreeName, fTreeName);
         std::vector<std::vector<int>> xs{{10}, {20, 30}, {40, 50, 60}, {70, 80, 90, 100}, {110, 120, 130, 140, 150}};
         std::vector<int> vecx;

         t.Branch("vecx", &vecx);

         for (int i = 0; i < fTreeEntries; i++) {
            vecx = xs[i];
            t.Fill();
            t.FlushBaskets();
         }

         t.Write();
      }

      {
         TFile f(fFileNames[2], "RECREATE");
         TTree t(fTreeName, fTreeName);
         std::vector<std::vector<int>> ys{{110, 120, 130, 140, 150}, {70, 80, 90, 100}, {40, 50, 60}, {20, 30}, {10}};
         std::vector<int> vecy;

         t.Branch("vecy", &vecy);

         for (int i = 0; i < fTreeEntries; i++) {
            vecy = ys[i];
            t.Fill();
            t.FlushBaskets();
         }

         t.Write();
      }
   }

   static void TearDownTestSuite()
   {
      for (auto &&fileName : fFileNames)
         std::remove(fileName);
   }
};

TEST_P(RDFMissingBranchesVec, DefaultValueForAndRVecTake)
{
   ROOT::RDataFrame df{fTreeName, {fFileNames[0], fFileNames[1], fFileNames[2]}};
   auto df_withcols = df.DefaultValueFor("vecy", ROOT::RVecI{700, 800, 900})
                         .DefaultValueFor("vecx", ROOT::RVecI{400, 500, 600})
                         .Define("vecz",
                                 [](const ROOT::RVecI &vec1, const ROOT::RVecI &vec2) {
                                    return vec1 + ROOT::VecOps::Take(vec2, vec1.size(), 10);
                                 },
                                 {"vecx", "vecy"});

   std::vector<std::vector<int>> expected_xs{// first file
                                             {1},
                                             {2, 3},
                                             {4, 5, 6},
                                             {7, 8, 9, 10},
                                             {11, 12, 13, 14, 15},
                                             // second file
                                             {10},
                                             {20, 30},
                                             {40, 50, 60},
                                             {70, 80, 90, 100},
                                             {110, 120, 130, 140, 150},
                                             // third file
                                             {400, 500, 600},
                                             {400, 500, 600},
                                             {400, 500, 600},
                                             {400, 500, 600},
                                             {400, 500, 600}};

   std::vector<std::vector<int>> expected_ys{// first file
                                             {11, 12, 13, 14, 15},
                                             {7, 8, 9, 10},
                                             {4, 5, 6},
                                             {2, 3},
                                             {1},
                                             // second file
                                             {700, 800, 900},
                                             {700, 800, 900},
                                             {700, 800, 900},
                                             {700, 800, 900},
                                             {700, 800, 900},
                                             // third file
                                             {110, 120, 130, 140, 150},
                                             {70, 80, 90, 100},
                                             {40, 50, 60},
                                             {20, 30},
                                             {10}};

   std::vector<std::vector<int>> expected_zs{// first file
                                             {12},
                                             {9, 11},
                                             {8, 10, 12},
                                             {9, 11, 19, 20},
                                             {12, 22, 23, 24, 25},
                                             // second file
                                             {710},
                                             {720, 830},
                                             {740, 850, 960},
                                             {770, 880, 990, 110},
                                             {810, 920, 1030, 150, 160},
                                             // third file
                                             {510, 620, 730},
                                             {470, 580, 690},
                                             {440, 550, 660},
                                             {420, 530, 610},
                                             {410, 510, 610}};

   auto take_x = df_withcols.Take<ROOT::RVecI>("vecx");
   auto take_y = df_withcols.Take<ROOT::RVecI>("vecy");
   auto take_z = df_withcols.Take<ROOT::RVecI>("vecz");

   if (GetParam()) {
      // Easiest way to check for equality of outputs is to sort the expected
      // vectors and the output vectors from Take the same way
      std::vector<std::vector<int>> take_x_val;
      take_x_val.reserve(take_x->size());
      for (auto &&rvec : take_x) {
         std::vector<int> vec;
         vec.reserve(rvec.size());
         for (auto &&v : rvec)
            vec.push_back(v);
         take_x_val.push_back(vec);
      }
      std::vector<std::vector<int>> take_y_val;
      take_y_val.reserve(take_y->size());
      for (auto &&rvec : take_y) {
         std::vector<int> vec;
         vec.reserve(rvec.size());
         for (auto &&v : rvec)
            vec.push_back(v);
         take_y_val.push_back(vec);
      }
      std::vector<std::vector<int>> take_z_val;
      take_z_val.reserve(take_z->size());
      for (auto &&rvec : take_z) {
         std::vector<int> vec;
         vec.reserve(rvec.size());
         for (auto &&v : rvec)
            vec.push_back(v);
         take_z_val.push_back(vec);
      }
      std::sort(take_x_val.begin(), take_x_val.end());
      std::sort(take_y_val.begin(), take_y_val.end());
      std::sort(take_z_val.begin(), take_z_val.end());
      std::sort(expected_xs.begin(), expected_xs.end());
      std::sort(expected_ys.begin(), expected_ys.end());
      std::sort(expected_zs.begin(), expected_zs.end());

      for (std::size_t i = 0; i < expected_xs.size(); i++) {
         expect_vec_eq(expected_xs[i], take_x_val[i]);
      }
      for (std::size_t i = 0; i < expected_ys.size(); i++) {
         expect_vec_eq(expected_ys[i], take_y_val[i]);
      }
      for (std::size_t i = 0; i < expected_zs.size(); i++) {
         expect_vec_eq(expected_zs[i], take_z_val[i]);
      }

   } else {
      auto take_x_val = *take_x;
      auto take_y_val = *take_y;
      auto take_z_val = *take_z;

      for (std::size_t i = 0; i < expected_xs.size(); i++) {
         expect_vec_eq(expected_xs[i], take_x_val[i]);
      }
      for (std::size_t i = 0; i < expected_ys.size(); i++) {
         expect_vec_eq(expected_ys[i], take_y_val[i]);
      }
      for (std::size_t i = 0; i < expected_zs.size(); i++) {
         expect_vec_eq(expected_zs[i], take_z_val[i]);
      }
   }
}

TEST_P(RDFMissingBranchesVec, DefaultValueForSimpleVariation)
{
   ROOT::RDataFrame df{fTreeName, {fFileNames[0], fFileNames[1], fFileNames[2]}};

   auto varyfn = [](const ROOT::RVecI &invec) { return ROOT::RVec<ROOT::RVecI>{invec - 1, invec + 2}; };

   auto df_withcols =
      df.DefaultValueFor("vecx", ROOT::RVecI{400, 500, 600}).Vary("vecx", varyfn, {"vecx"}, {"down", "up"});

   std::vector<std::vector<int>> expected_xs_nominal{// first file
                                                     {1},
                                                     {2, 3},
                                                     {4, 5, 6},
                                                     {7, 8, 9, 10},
                                                     {11, 12, 13, 14, 15},
                                                     // second file
                                                     {10},
                                                     {20, 30},
                                                     {40, 50, 60},
                                                     {70, 80, 90, 100},
                                                     {110, 120, 130, 140, 150},
                                                     // third file
                                                     {400, 500, 600},
                                                     {400, 500, 600},
                                                     {400, 500, 600},
                                                     {400, 500, 600},
                                                     {400, 500, 600}};

   // Expected values for variation up
   std::vector<std::vector<int>> expected_xs_up(expected_xs_nominal.size());
   std::transform(expected_xs_nominal.cbegin(), expected_xs_nominal.cend(), expected_xs_up.begin(),
                  [](const std::vector<int> &val) {
                     std::vector<int> ret(val.size());
                     std::transform(val.cbegin(), val.cend(), ret.begin(),
                                    std::bind(std::plus<int>(), 2, std::placeholders::_1));
                     return ret;
                  });

   // Expected values for variation down
   std::vector<std::vector<int>> expected_xs_down(expected_xs_nominal.size());
   std::transform(expected_xs_nominal.cbegin(), expected_xs_nominal.cend(), expected_xs_down.begin(),
                  [](const std::vector<int> &val) {
                     std::vector<int> ret(val.size());
                     std::transform(val.cbegin(), val.cend(), ret.begin(),
                                    std::bind(std::plus<int>(), -1, std::placeholders::_1));
                     return ret;
                  });

   auto take_x = df_withcols.Take<ROOT::RVecI>("vecx");
   auto take_x_vars = ROOT::RDF::Experimental::VariationsFor(take_x);

   if (GetParam()) {
      // Easiest way to check for equality of outputs is to sort the expected
      // vectors and the output vectors from Take the same way
      std::vector<std::vector<int>> take_x_val_nominal;
      take_x_val_nominal.reserve(take_x_vars["nominal"].size());
      for (auto &&rvec : take_x_vars["nominal"]) {
         std::vector<int> vec;
         vec.reserve(rvec.size());
         for (auto &&v : rvec)
            vec.push_back(v);
         take_x_val_nominal.push_back(vec);
      }
      std::vector<std::vector<int>> take_x_val_up;
      take_x_val_up.reserve(take_x_vars["vecx:up"].size());
      for (auto &&rvec : take_x_vars["vecx:up"]) {
         std::vector<int> vec;
         vec.reserve(rvec.size());
         for (auto &&v : rvec)
            vec.push_back(v);
         take_x_val_up.push_back(vec);
      }
      std::vector<std::vector<int>> take_x_val_down;
      take_x_val_down.reserve(take_x_vars["vecx:down"].size());
      for (auto &&rvec : take_x_vars["vecx:down"]) {
         std::vector<int> vec;
         vec.reserve(rvec.size());
         for (auto &&v : rvec)
            vec.push_back(v);
         take_x_val_down.push_back(vec);
      }

      std::sort(take_x_val_nominal.begin(), take_x_val_nominal.end());
      std::sort(take_x_val_up.begin(), take_x_val_up.end());
      std::sort(take_x_val_down.begin(), take_x_val_down.end());

      std::sort(expected_xs_nominal.begin(), expected_xs_nominal.end());
      std::sort(expected_xs_up.begin(), expected_xs_up.end());
      std::sort(expected_xs_down.begin(), expected_xs_down.end());

      for (std::size_t i = 0; i < expected_xs_nominal.size(); i++) {
         expect_vec_eq(expected_xs_nominal[i], take_x_val_nominal[i]);
      }
      for (std::size_t i = 0; i < expected_xs_up.size(); i++) {
         expect_vec_eq(expected_xs_up[i], take_x_val_up[i]);
      }
      for (std::size_t i = 0; i < expected_xs_down.size(); i++) {
         expect_vec_eq(expected_xs_down[i], take_x_val_down[i]);
      }

   } else {

      for (std::size_t i = 0; i < expected_xs_nominal.size(); i++) {
         expect_vec_eq(expected_xs_nominal[i], take_x_vars["nominal"][i]);
      }
      for (std::size_t i = 0; i < expected_xs_up.size(); i++) {
         expect_vec_eq(expected_xs_up[i], take_x_vars["vecx:up"][i]);
      }
      for (std::size_t i = 0; i < expected_xs_down.size(); i++) {
         expect_vec_eq(expected_xs_down[i], take_x_vars["vecx:down"][i]);
      }
   }
}

struct RDFIndexedFriendsOrder : public ::testing::TestWithParam<std::pair<bool, bool>> {

   constexpr static auto fMainTreeName{"main_tree"};
   constexpr static auto fAuxTreeName{"aux_tree"};
   constexpr static auto fSmallAuxTreename{"aux_tree_small"};

   constexpr static std::uint32_t fNEntries{1000};
   constexpr static std::uint32_t fNEntriesPerFile{100};

   constexpr static std::uint32_t fNEntriesSmall{100};
   constexpr static std::uint32_t fNEntriesPerFileSmall{10};

   std::vector<std::uint32_t> fEvtNumberAuxSmall{};
   std::vector<std::string> fFileNames{};
   const unsigned int fNSlots;

   RDFIndexedFriendsOrder() : fNSlots(GetParam().first ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam().first)
         ROOT::EnableImplicitMT(fNSlots);
   }

   void SetUp() override
   {
      std::random_device rd;
      std::mt19937 g{rd()};

      std::vector<std::uint32_t> evtNumberMain(fNEntries);
      std::iota(evtNumberMain.begin(), evtNumberMain.end(), 0u);
      if (GetParam().second) {
         // Create same list of event numbers, but shuffled differently
         // This simulates the most extreme (and complex) case for horizontal
         // composition of two chains: global join.
         std::shuffle(evtNumberMain.begin(), evtNumberMain.end(), g);
      }

      std::vector<std::uint32_t> evtNumberAux(fNEntries);
      std::iota(evtNumberAux.begin(), evtNumberAux.end(), 0u);
      if (GetParam().second) {
         // Create same list of event numbers, but shuffled differently
         // This simulates the most extreme (and complex) case for horizontal
         // composition of two chains: global join.
         std::shuffle(evtNumberAux.begin(), evtNumberAux.end(), g);
      }

      fEvtNumberAuxSmall = std::vector<std::uint32_t>(fNEntriesSmall);
      std::sample(evtNumberMain.begin(), evtNumberMain.end(), fEvtNumberAuxSmall.begin(), fEvtNumberAuxSmall.size(), g);
      if (!GetParam().second) {
         // Sort for the case of aligned join
         std::sort(fEvtNumberAuxSmall.begin(), fEvtNumberAuxSmall.end());
         // If the index is not being built with the first entry with value zero, TChainIndex will print a spurious
         // warning
         fEvtNumberAuxSmall[0] = 0;
      }

      for (std::uint32_t chainOffset{}, smallChainOffset{}; chainOffset < fNEntries;
           chainOffset += fNEntriesPerFile, smallChainOffset += fNEntriesPerFileSmall) {

         std::string filename{"file_main_" + std::to_string(chainOffset) + ".root"};
         fFileNames.push_back(filename);

         TFile f(filename.c_str(), "RECREATE");

         TTree mainTree{fMainTreeName, fMainTreeName};
         std::uint32_t evtMain;
         std::uint32_t x;
         mainTree.Branch("evt_number", &evtMain);
         mainTree.Branch("x", &x);

         TTree auxTree{fAuxTreeName, fAuxTreeName};
         std::uint32_t evtAux;
         std::uint32_t y;
         auxTree.Branch("evt_number", &evtAux);
         auxTree.Branch("y", &y);

         for (std::uint32_t i = 0; i < fNEntriesPerFile; i++) {
            evtMain = evtNumberMain.at(chainOffset + i);
            x = evtMain;

            evtAux = evtNumberAux.at(chainOffset + i);
            y = evtAux;

            mainTree.Fill();
            auxTree.Fill();
         }

         // This tree is smaller, its event number branch is a random sample of
         // the main event number branch
         TTree auxTreeSmall{fSmallAuxTreename, fSmallAuxTreename};
         std::uint32_t evtAuxSmall;
         std::uint32_t z;
         auxTreeSmall.Branch("evt_number", &evtAuxSmall);
         auxTreeSmall.Branch("z", &z);

         for (std::uint32_t i = 0; i < fNEntriesPerFileSmall; i++) {
            evtAuxSmall = fEvtNumberAuxSmall.at(smallChainOffset + i);
            z = evtAuxSmall;

            auxTreeSmall.Fill();
         }

         f.Write();
      }
   }

   void TearDown() override
   {
      for (auto &&fileName : fFileNames)
         std::remove(fileName.c_str());
   }

   ~RDFIndexedFriendsOrder() override
   {
      if (GetParam().first)
         ROOT::DisableImplicitMT();
   }
};

TEST_P(RDFIndexedFriendsOrder, GlobalJoinSameCardinality)
{
   // The error regarding a wrong order of the index column values when building
   // the TChainIndex should only be displayed once, whether in single or multi
   // threaded mode.
   TChain mainChain{fMainTreeName};
   for (const auto &fn : fFileNames) {
      mainChain.Add(fn.c_str());
   }

   TChain auxChain{fAuxTreeName};
   for (const auto &fn : fFileNames) {
      auxChain.Add(fn.c_str());
   }
   {
      if (GetParam().second) {
         // The error messages should only be printed here, the first time the
         // index is built by the user
         ROOT::TestSupport::CheckDiagsRAII diagRAII;
         diagRAII.requiredDiag(kError, "TChainIndex::TChainIndex", "The indices in files of this chain aren't sorted.");
         diagRAII.requiredDiag(kError, "TTreePlayer::BuildIndex",
                               "Creating a TChainIndex unsuccessful - switching to TTreeIndex");
         auxChain.BuildIndex("evt_number");
      } else {
         auxChain.BuildIndex("evt_number");
      }
   }
   mainChain.AddFriend(&auxChain);

   // The error message from the TChainIndex should not be duplicated by RDataFrame
   ROOT::RDataFrame df{mainChain};

   auto takeX = df.FilterAvailable("y").Take<std::uint32_t>("x");

   std::vector<std::uint32_t> expectedX(fNEntries);
   std::iota(expectedX.begin(), expectedX.end(), 0u);
   auto valuesX = *takeX;
   std::sort(valuesX.begin(), valuesX.end());

   // Even after switching from TChainIndex to TTreeIndex, the matching should
   // happen as expected.
   expect_vec_eq(expectedX, valuesX);
}

TEST_P(RDFIndexedFriendsOrder, GlobalJoinDifferentCardinality)
{
   TChain mainChain{fMainTreeName};
   for (const auto &fn : fFileNames) {
      mainChain.Add(fn.c_str());
   }

   TChain auxChain{fSmallAuxTreename};
   for (const auto &fn : fFileNames) {
      auxChain.Add(fn.c_str());
   }
   {
      if (GetParam().second) {
         // The error messages should only be printed here, the first time the
         // index is built by the user
         ROOT::TestSupport::CheckDiagsRAII diagRAII;
         diagRAII.requiredDiag(kError, "TChainIndex::TChainIndex", "The indices in files of this chain aren't sorted.");
         diagRAII.requiredDiag(kError, "TTreePlayer::BuildIndex",
                               "Creating a TChainIndex unsuccessful - switching to TTreeIndex");
         auxChain.BuildIndex("evt_number");
      } else {
         auxChain.BuildIndex("evt_number");
      }
   }
   mainChain.AddFriend(&auxChain);

   // The cardinality is different: the auxiliary chain is smaller than the main chain.
   ROOT::RDataFrame df{mainChain};

   auto takeX = df.FilterAvailable("z").Take<std::uint32_t>("x");

   auto expectedX = fEvtNumberAuxSmall;
   std::sort(expectedX.begin(), expectedX.end());
   auto valuesX = *takeX;
   std::sort(valuesX.begin(), valuesX.end());

   // Even after switching from TChainIndex to TTreeIndex, the matching should
   // happen as expected.
   expect_vec_eq(expectedX, valuesX);
}

// run single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFMissingBranches, ::testing::Values(false));
INSTANTIATE_TEST_SUITE_P(Seq, RDFIndexNoMatchSimple,
                         ::testing::Values(std::make_pair(false, false), std::make_pair(false, true)));
INSTANTIATE_TEST_SUITE_P(Seq, RDFMissingBranchesVec, ::testing::Values(false));
INSTANTIATE_TEST_SUITE_P(Seq, RDFIndexedFriendsOrder,
                         ::testing::Values(std::make_pair(false, false), std::make_pair(false, true)));

// run multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFMissingBranches, ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(MT, RDFIndexNoMatchSimple,
                         ::testing::Values(std::make_pair(true, false), std::make_pair(true, true)));
INSTANTIATE_TEST_SUITE_P(MT, RDFMissingBranchesVec, ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(MT, RDFIndexedFriendsOrder,
                         ::testing::Values(std::make_pair(true, false), std::make_pair(true, true)));
#endif
