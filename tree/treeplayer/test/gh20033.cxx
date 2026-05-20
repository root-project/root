#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreePlayer.h>

#include <algorithm>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <fstream>

#include <gtest/gtest.h>

#include "CMSDASClasses.hxx"

// Meaning of the parameters: first 4 booleans indicate whether to use TTree or TChain for that
// step, the int indicates which entry to start from when calling Scan
struct GH20033Regression : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool, int>> {

   constexpr static std::array<const char *, 2> fStepZeroFileNames{"tree_gh20033_regression_stepzero_0.root",
                                                                   "tree_gh20033_regression_stepzero_1.root"};

   constexpr static auto fStepOneFileName{"tree_gh20033_regression_stepone.root"};
   constexpr static auto fStepTwoFileName{"tree_gh20033_regression_steptwo.root"};
   constexpr static auto fStepThreeFileName{"tree_gh20033_regression_stepthree.root"};
   constexpr static auto fStepFourFileName{"tree_gh20033_regression_stepfour.root"};

   constexpr static auto fStepZeroTreeName{"stepzerotree"};
   constexpr static auto fStepOneTreeName{"steponetree"};
   constexpr static auto fStepTwoTreeName{"steptwotree"};
   constexpr static auto fStepThreeTreeName{"stepthreetree"};
   constexpr static auto fStepFourTreeName{"stepfourtree"};

   constexpr static auto fTotalEntries{20};

   // In the test we will try calling TTree::Scan with different starting entries,
   // namely 0, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19
   const static std::unordered_map<int, int> fStartingEntriesToOutput;
   constexpr static std::array<const char *, 11> fScanExpectedOutputs{
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*        0 *            0 *            0 *
*        1 *            1 *            2 *
*        2 *            2 *            4 *
*        3 *            3 *            6 *
*        4 *            4 *            8 *
*        5 *            5 *           10 *
*        6 *            6 *           12 *
*        7 *            7 *           14 *
*        8 *            8 *           16 *
*        9 *            9 *           18 *
*       10 *           10 *           20 *
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*        3 *            3 *            6 *
*        4 *            4 *            8 *
*        5 *            5 *           10 *
*        6 *            6 *           12 *
*        7 *            7 *           14 *
*        8 *            8 *           16 *
*        9 *            9 *           18 *
*       10 *           10 *           20 *
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*        5 *            5 *           10 *
*        6 *            6 *           12 *
*        7 *            7 *           14 *
*        8 *            8 *           16 *
*        9 *            9 *           18 *
*       10 *           10 *           20 *
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*        7 *            7 *           14 *
*        8 *            8 *           16 *
*        9 *            9 *           18 *
*       10 *           10 *           20 *
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*        9 *            9 *           18 *
*       10 *           10 *           20 *
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*       10 *           10 *           20 *
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*       11 *           11 *           22 *
*       12 *           12 *           24 *
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*       13 *           13 *           26 *
*       14 *           14 *           28 *
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*       15 *           15 *           30 *
*       16 *           16 *           32 *
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*       17 *           17 *           34 *
*       18 *           18 *           36 *
*       19 *           19 *           38 *
******************************************
)Scan",
      R"Scan(******************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *
******************************************
*       19 *           19 *           38 *
******************************************
)Scan"};

   static void MakeStepZeroFile(const char *name, const char *treename, int first, int last)
   {
      auto file = std::make_unique<TFile>(name, "RECREATE");
      auto tree = std::make_unique<TTree>(treename, treename);

      int br1{};
      int br2{};
      tree->Branch("stepZeroBr1", &br1);
      tree->Branch("stepZeroBr2", &br2);

      for (br1 = first; br1 < last; ++br1) {
         br2 = 2 * br1;
         tree->Fill();
      }

      file->Write();
   }

   static void MakeOtherStepsFile(const char *name, const char *treename, int nEntries)
   {
      auto file = std::make_unique<TFile>(name, "RECREATE");
      auto tree = std::make_unique<TTree>(treename, treename);

      // empty entries, but crucially same number as the number of entries
      // in the step zero chain, to ensure alignment
      for (int i = 0; i < nEntries; ++i) {
         tree->Fill();
      }

      file->Write();
   }

   static void SetUpTestSuite()
   {
      MakeStepZeroFile(fStepZeroFileNames[0], fStepZeroTreeName, 0, 10);
      MakeStepZeroFile(fStepZeroFileNames[1], fStepZeroTreeName, 10, 20);
      MakeOtherStepsFile(fStepOneFileName, fStepOneTreeName, fTotalEntries);
      MakeOtherStepsFile(fStepTwoFileName, fStepTwoTreeName, fTotalEntries);
      MakeOtherStepsFile(fStepThreeFileName, fStepThreeTreeName, fTotalEntries);
      MakeOtherStepsFile(fStepFourFileName, fStepFourTreeName, fTotalEntries);
   }

   static void TearDownTestSuite()
   {
      for (const auto &fileName : fStepZeroFileNames)
         std::remove(fileName);

      std::remove(fStepOneFileName);
      std::remove(fStepTwoFileName);
      std::remove(fStepThreeFileName);
      std::remove(fStepFourFileName);
   }
};

const std::unordered_map<int, int> GH20033Regression::fStartingEntriesToOutput = {
   {0, 0}, {3, 1}, {5, 2}, {7, 3}, {9, 4}, {10, 5}, {11, 6}, {13, 7}, {15, 8}, {17, 9}, {19, 10}};

TEST_P(GH20033Regression, Test)
{
   // General idea: there is one TChain dataset which contains the actual
   // data to traverse, in this case two branches. this TChain is connected
   // to a hierarchy of friends. In general this could be arbitrarily long.
   // For the purpose of this test, this is the friendship chain:
   // - TChain "step zero"
   //   --> befriended by TTree "step one"
   //       (this TTree could either be standalone, or part of a TChain itself)
   //       --> befriended by TTree "step two"
   //           (this TTree could either be standalone, or part of a TChain itself)
   //
   // When the boundaries between trees of the TChain "step zero" is reached, the act
   // of switching over to a new tree should trigger an update notification across
   // the whole friendship chain, most importantly because the memory address provided
   // by the user is known by the top-most befriender (TChain "step two").
   // This test ensures smooth update notification triggers across the friendship chain,
   // irrespective of whether the intermediate datasets are TTree/TChain.

   auto &&[stepOneTChain, stepTwoTChain, stepThreeTChain, stepFourTChain, startingEntry] = GetParam();

   // The step zero dataset needs to be a TChain and have more than one TTree
   // to recreate the scenarios when one component of the friendship chain
   // is a TChain switching to a new TTree and thus triggering the update
   auto stepZeroDataset = std::make_unique<TChain>(fStepZeroTreeName);
   for (const auto &fileName : fStepZeroFileNames)
      stepZeroDataset->Add(fileName);

   // Step one dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepOneFileLifeLine{};
   std::unique_ptr<TTree> stepOneDataset{};
   if (stepOneTChain) {
      auto chain = std::make_unique<TChain>(fStepOneTreeName);
      chain->Add(fStepOneFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepZeroDataset.get());
      stepOneDataset = std::move(chain);
   } else {
      stepOneFileLifeLine = std::make_unique<TFile>(fStepOneFileName);
      stepOneDataset = std::unique_ptr<TTree>{stepOneFileLifeLine->Get<TTree>(fStepOneTreeName)};
      stepOneDataset->AddFriend(stepZeroDataset.get());
   }

   // Step two dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepTwoFileLifeLine{};
   std::unique_ptr<TTree> stepTwoDataset{};
   if (stepTwoTChain) {
      auto chain = std::make_unique<TChain>(fStepTwoTreeName);
      chain->Add(fStepTwoFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepOneDataset.get());
      stepTwoDataset = std::move(chain);
   } else {
      stepTwoFileLifeLine = std::make_unique<TFile>(fStepTwoFileName);
      stepTwoDataset = std::unique_ptr<TTree>{stepTwoFileLifeLine->Get<TTree>(fStepTwoTreeName)};
      stepTwoDataset->AddFriend(stepOneDataset.get());
   }

   // Step three dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepThreeFileLifeLine{};
   std::unique_ptr<TTree> stepThreeDataset{};
   if (stepThreeTChain) {
      auto chain = std::make_unique<TChain>(fStepThreeTreeName);
      chain->Add(fStepThreeFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepTwoDataset.get());
      stepThreeDataset = std::move(chain);
   } else {
      stepThreeFileLifeLine = std::make_unique<TFile>(fStepThreeFileName);
      stepThreeDataset = std::unique_ptr<TTree>{stepThreeFileLifeLine->Get<TTree>(fStepThreeTreeName)};
      stepThreeDataset->AddFriend(stepTwoDataset.get());
   }

   // Step four dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepFourFileLifeLine{};
   std::unique_ptr<TTree> stepFourDataset{};
   if (stepFourTChain) {
      auto chain = std::make_unique<TChain>(fStepFourTreeName);
      chain->Add(fStepFourFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepThreeDataset.get());
      stepFourDataset = std::move(chain);
   } else {
      stepFourFileLifeLine = std::make_unique<TFile>(fStepFourFileName);
      stepFourDataset = std::unique_ptr<TTree>{stepFourFileLifeLine->Get<TTree>(fStepFourTreeName)};
      stepFourDataset->AddFriend(stepThreeDataset.get());
   }

   // Set the branch addresses on the top-most befriender, i.e. step two dataset.
   // This is what establishes the memory addresses where the user wants data to be read into.
   // These addresses need to be propagated across the friendship chain, so that they can be
   // connected to the true TBranch addresses to fill the data as its being read from disk.
   int stepZeroBr1 = -1, stepZeroBr2 = -1;
   ASSERT_EQ(stepFourDataset->SetBranchAddress("stepZeroBr1", &stepZeroBr1), 0);
   ASSERT_EQ(stepFourDataset->SetBranchAddress("stepZeroBr2", &stepZeroBr2), 0);

   std::vector<int> expectedStepZeroBr1(fTotalEntries);
   std::vector<int> expectedStepZeroBr2(fTotalEntries);
   std::iota(expectedStepZeroBr1.begin(), expectedStepZeroBr1.end(), 0);
   std::generate(expectedStepZeroBr2.begin(), expectedStepZeroBr2.end(), [n = 0]() mutable {
      auto res = n * 2;
      n++;
      return res;
   });

   for (Long64_t i = 0; i < stepFourDataset->GetEntriesFast(); ++i) {
      stepFourDataset->GetEntry(i);
      EXPECT_EQ(expectedStepZeroBr1[i], stepZeroBr1);
      EXPECT_EQ(expectedStepZeroBr2[i], stepZeroBr2);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(stepFourDataset->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"tree_gh20033_morefriends_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         stepFourDataset->Scan("stepZeroBr1:stepZeroBr2", "", "colsize=12", TTree::kMaxEntries, startingEntry);

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         EXPECT_EQ(redirectOutput.str(), fScanExpectedOutputs[fStartingEntriesToOutput.at(startingEntry)]);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

INSTANTIATE_TEST_SUITE_P(
   Run, GH20033Regression,
   ::testing::Combine(::testing::Values(false, true), ::testing::Values(false, true), ::testing::Values(false, true),
                      ::testing::Values(false, true), ::testing::Values(0, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19)),
   // Extra parenthesis around lambda to avoid preprocessor errors, see
   // https://stackoverflow.com/questions/79438894/lambda-with-structured-bindings-inside-macro-call
   ([](const testing::TestParamInfo<GH20033Regression::ParamType> &paramInfo) {
      auto &&[stepOneTChain, stepTwoTChain, stepThreeTChain, stepFourTChain, startingEntry] = paramInfo.param;
      // googletest only accepts ASCII alphanumeric characters for labels
      std::string label{};
      if (stepOneTChain)
         label += "StepOneTChain_";
      else
         label += "StepOneTTree_";
      if (stepTwoTChain)
         label += "StepTwoTChain_";
      else
         label += "StepTwoTTree_";
      if (stepThreeTChain)
         label += "StepThreeTChain_";
      else
         label += "StepThreeTTree_";
      if (stepFourTChain)
         label += "StepFourTChain_";
      else
         label += "StepFourTTree_";
      label += std::to_string(startingEntry);
      return label;
   }));

// A similar situation as above, but now all TTree/TChain datasets in the hierarchy have the same
// branch, with different values. The std::string parameters represent the name of the expression
// to pass to TTree::Scan for the corresponding friend dataset branch
struct GH20033SameBranchName
   : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool, std::string, std::string, std::string,
                                                std::string, std::string, std::string>> {

   constexpr static std::array<const char *, 2> fStepZeroFileNames{"tree_gh20033_samebranchname_stepzero_0.root",
                                                                   "tree_gh20033_samebranchname_stepzero_1.root"};

   constexpr static auto fStepOneFileName{"tree_gh20033_samebranchname_stepone.root"};
   constexpr static auto fStepTwoFileName{"tree_gh20033_samebranchname_steptwo.root"};
   constexpr static auto fStepThreeFileName{"tree_gh20033_samebranchname_stepthree.root"};
   constexpr static auto fStepFourFileName{"tree_gh20033_samebranchname_stepfour.root"};

   constexpr static auto fStepZeroTreeName{"step0tree"};
   constexpr static auto fStepOneTreeName{"step1tree"};
   constexpr static auto fStepTwoTreeName{"step2tree"};
   constexpr static auto fStepThreeTreeName{"step3tree"};
   constexpr static auto fStepFourTreeName{"step4tree"};

   // The names of the chains use the token 'chai' for simplicity, it has same number of characters as 'tree', to make
   // it easier to build the expected TTree::Scan output header later on in the test itself.
   constexpr static auto fStepZeroChainName{"step0chai"};
   constexpr static auto fStepOneChainName{"step1chai"};
   constexpr static auto fStepTwoChainName{"step2chai"};
   constexpr static auto fStepThreeChainName{"step3chai"};
   constexpr static auto fStepFourChainName{"step4chai"};

   constexpr static auto fTotalEntries{4};

   static void WriteData(const char *name, const char *treename, int first, int last)
   {
      auto file = std::make_unique<TFile>(name, "RECREATE");
      auto tree = std::make_unique<TTree>(treename, treename);

      int val{};
      tree->Branch("value", &val);

      DAS::GenEvent evt{};
      evt.weights = std::vector<DAS::Weight>{{0, 1}};
      tree->Branch("genEvent", &evt);

      for (val = first; val < last; ++val) {
         evt.weights[0].v = val;
         evt.weights[0].i = val;
         tree->Fill();
      }

      file->Write();
   }

   GH20033SameBranchName()
   {
      auto &&[_1, _2, _3, _4, stepZeroScan, stepOneScan, stepTwoScan, stepThreeScan, stepFourScan, _5] = GetParam();

      // All scan strings are empty, we will skip the test, avoid useless work here
      if (stepZeroScan.empty() && stepOneScan.empty() && stepTwoScan.empty() && stepThreeScan.empty() &&
          stepFourScan.empty())
         return;

      // We want to write out TTrees in files which might have either the same name of the TChain that will
      // wrap them, or their own different name than the one of the TChain that will wrap them later in the
      // test.
      WriteData(fStepZeroFileNames[0], (stepZeroScan.empty() ? fStepZeroTreeName : stepZeroScan.c_str()), 0,
                fTotalEntries / 2);
      WriteData(fStepZeroFileNames[1], (stepZeroScan.empty() ? fStepZeroTreeName : stepZeroScan.c_str()),
                fTotalEntries / 2, fTotalEntries);
      WriteData(fStepOneFileName, (stepOneScan.empty() ? fStepOneTreeName : stepOneScan.c_str()), 100,
                100 + fTotalEntries);
      WriteData(fStepTwoFileName, (stepTwoScan.empty() ? fStepTwoTreeName : stepTwoScan.c_str()), 200,
                200 + fTotalEntries);
      WriteData(fStepThreeFileName, (stepThreeScan.empty() ? fStepThreeTreeName : stepThreeScan.c_str()), 300,
                300 + fTotalEntries);
      WriteData(fStepFourFileName, (stepFourScan.empty() ? fStepFourTreeName : stepFourScan.c_str()), 400,
                400 + fTotalEntries);
   }

   ~GH20033SameBranchName()
   {
      for (const auto &fileName : fStepZeroFileNames)
         std::remove(fileName);

      std::remove(fStepOneFileName);
      std::remove(fStepTwoFileName);
      std::remove(fStepThreeFileName);
      std::remove(fStepFourFileName);
   }
};

TEST_P(GH20033SameBranchName, Test)
{
   // General idea: all the datasets in the hierarchy of friendships share the same
   // branch name "val". Each dataset will have different values to distinguish itself.
   // Try all different combinations of TTree::Scan such that all different branches
   // in the friend hierarchy are tested, specifying the corresponding dataset name

   auto &&[stepOneTChain, stepTwoTChain, stepThreeTChain, stepFourTChain, stepZeroScan, stepOneScan, stepTwoScan,
           stepThreeScan, stepFourScan, expressionName] = GetParam();
   if (stepZeroScan.empty() && stepOneScan.empty() && stepTwoScan.empty() && stepThreeScan.empty() &&
       stepFourScan.empty())
      GTEST_SKIP() << "Skipping test with empty TTree::Scan expression";

   // Precompute which was the name of the TTree written to disk for this test run
   const char *stepZeroToDiskName = stepZeroScan.empty() ? fStepZeroTreeName : stepZeroScan.c_str();
   const char *stepOneToDiskName = stepOneScan.empty() ? fStepOneTreeName : stepOneScan.c_str();
   const char *stepTwoToDiskName = stepTwoScan.empty() ? fStepTwoTreeName : stepTwoScan.c_str();
   const char *stepThreeToDiskName = stepThreeScan.empty() ? fStepThreeTreeName : stepThreeScan.c_str();
   const char *stepFourToDiskName = stepFourScan.empty() ? fStepFourTreeName : stepFourScan.c_str();

   // The step zero dataset needs to be a TChain and have more than one TTree
   // to recreate the scenarios when one component of the friendship chain
   // is a TChain switching to a new TTree and thus triggering the update
   auto stepZeroDataset = std::make_unique<TChain>(fStepZeroChainName);
   for (const auto &fileName : fStepZeroFileNames)
      stepZeroDataset->Add((std::string(fileName) + "?#" + stepZeroToDiskName).c_str());

   // Step one dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepOneFileLifeLine{};
   std::unique_ptr<TTree> stepOneDataset{};
   if (stepOneTChain) {
      auto chain = std::make_unique<TChain>(fStepOneChainName);
      chain->Add((std::string(fStepOneFileName) + "?#" + stepOneToDiskName).c_str());
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepZeroDataset.get());
      stepOneDataset = std::move(chain);
   } else {
      stepOneFileLifeLine = std::make_unique<TFile>(fStepOneFileName);
      stepOneDataset = std::unique_ptr<TTree>{stepOneFileLifeLine->Get<TTree>(stepOneToDiskName)};
      stepOneDataset->AddFriend(stepZeroDataset.get());
   }

   // Step two dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepTwoFileLifeLine{};
   std::unique_ptr<TTree> stepTwoDataset{};
   if (stepTwoTChain) {
      auto chain = std::make_unique<TChain>(fStepTwoChainName);
      chain->Add((std::string(fStepTwoFileName) + "?#" + stepTwoToDiskName).c_str());
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepOneDataset.get());
      stepTwoDataset = std::move(chain);
   } else {
      stepTwoFileLifeLine = std::make_unique<TFile>(fStepTwoFileName);
      stepTwoDataset = std::unique_ptr<TTree>{stepTwoFileLifeLine->Get<TTree>(stepTwoToDiskName)};
      stepTwoDataset->AddFriend(stepOneDataset.get());
   }

   // Step three dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepThreeFileLifeLine{};
   std::unique_ptr<TTree> stepThreeDataset{};
   if (stepThreeTChain) {
      auto chain = std::make_unique<TChain>(fStepThreeChainName);
      chain->Add((std::string(fStepThreeFileName) + "?#" + stepThreeToDiskName).c_str());
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepTwoDataset.get());
      stepThreeDataset = std::move(chain);
   } else {
      stepThreeFileLifeLine = std::make_unique<TFile>(fStepThreeFileName);
      stepThreeDataset = std::unique_ptr<TTree>{stepThreeFileLifeLine->Get<TTree>(stepThreeToDiskName)};
      stepThreeDataset->AddFriend(stepTwoDataset.get());
   }

   // Step four dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepFourFileLifeLine{};
   std::unique_ptr<TTree> stepFourDataset{};
   if (stepFourTChain) {
      auto chain = std::make_unique<TChain>(fStepFourChainName);
      chain->Add((std::string(fStepFourFileName) + "?#" + stepFourToDiskName).c_str());
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepThreeDataset.get());
      stepFourDataset = std::move(chain);
   } else {
      stepFourFileLifeLine = std::make_unique<TFile>(fStepFourFileName);
      stepFourDataset = std::unique_ptr<TTree>{stepFourFileLifeLine->Get<TTree>(stepFourToDiskName)};
      stepFourDataset->AddFriend(stepThreeDataset.get());
   }

   // Build the TTree::Scan call
   std::string scanExpression{};
   // step zero is always a TChain, we can safely use the name of the chain
   if (!stepZeroScan.empty())
      scanExpression += "step0chai." + expressionName + ":";

   // Other steps can be either accessed via TTree or TChain
   // In the case of TTree, the scan expression can only
   // have the name of the TTree written on disk. In the
   // case of TChain, the scan expression has the name of the
   // TChain, which could be different than the name of the TTree
   // it wraps.
   if (!stepOneScan.empty()) {
      if (stepOneTChain)
         scanExpression += "step1chai." + expressionName + ":";
      else
         scanExpression += std::string(stepOneToDiskName) + "." + expressionName + ":";
   }

   if (!stepTwoScan.empty()) {
      if (stepTwoTChain)
         scanExpression += "step2chai." + expressionName + ":";
      else
         scanExpression += std::string(stepTwoToDiskName) + "." + expressionName + ":";
   }

   if (!stepThreeScan.empty()) {
      if (stepThreeTChain)
         scanExpression += "step3chai." + expressionName + ":";
      else
         scanExpression += std::string(stepThreeToDiskName) + "." + expressionName + ":";
   }

   if (!stepFourScan.empty()) {
      if (stepFourTChain)
         scanExpression += "step4chai." + expressionName + ":";
      else
         scanExpression += std::string(stepFourToDiskName) + "." + expressionName + ":";
   }

   // The scan expression can never be empty, we have already dealt with that case by skipping the test at the beginning
   if (scanExpression.back() == ':')
      scanExpression.pop_back();

   // Build the expected TTree::Scan output
   std::string expectedScan{};
   const auto rowHeaderNChars{12};   // 10 is the Row header, +2 column delimiters
   const auto colNChars{15 + 2 + 1}; // colsize=15, +2 margin, +1 right column delimiter
   // Header, opening row
   std::string starRow(rowHeaderNChars, '*');
   if (!stepZeroScan.empty())
      starRow += std::string(colNChars, '*');
   if (!stepOneScan.empty())
      starRow += std::string(colNChars, '*');
   if (!stepTwoScan.empty())
      starRow += std::string(colNChars, '*');
   if (!stepThreeScan.empty())
      starRow += std::string(colNChars, '*');
   if (!stepFourScan.empty())
      starRow += std::string(colNChars, '*');
   starRow += "\n";

   expectedScan += starRow;

   // Header, branch titles
   expectedScan += "*    Row   *";
   // We keep the same colsize in the output scan to make the building of the expected output easier. Cut the longer
   // expression name to the same length as the other branch name.
   std::string expressionToken{(expressionName == "genEvent.Weight()" ? "genEv" : "value")};
   if (!stepZeroScan.empty())
      expectedScan += " step0chai." + expressionToken + " *";
   if (!stepOneScan.empty())
      expectedScan += " " + std::string(stepOneTChain ? "step1chai" : stepOneToDiskName) + "." + expressionToken + " *";
   if (!stepTwoScan.empty())
      expectedScan += " " + std::string(stepTwoTChain ? "step2chai" : stepTwoToDiskName) + "." + expressionToken + " *";
   if (!stepThreeScan.empty())
      expectedScan +=
         " " + std::string(stepThreeTChain ? "step3chai" : stepThreeToDiskName) + "." + expressionToken + " *";
   if (!stepFourScan.empty())
      expectedScan +=
         " " + std::string(stepFourTChain ? "step4chai" : stepFourToDiskName) + "." + expressionToken + " *";
   expectedScan += "\n";

   // Header, closing row
   expectedScan += starRow;

   // main body rows
   for (std::string n : {"0", "1", "2", "3"}) {
      expectedScan += "*        " + n + " *";
      if (!stepZeroScan.empty())
         expectedScan += "               " + n + " *";
      if (!stepOneScan.empty())
         expectedScan += "             10" + n + " *";
      if (!stepTwoScan.empty())
         expectedScan += "             20" + n + " *";
      if (!stepThreeScan.empty())
         expectedScan += "             30" + n + " *";
      if (!stepFourScan.empty())
         expectedScan += "             40" + n + " *";
      expectedScan += "\n";
   }

   // Ending row
   expectedScan += starRow;

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(stepFourDataset->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"tree_gh20033_samebranchname_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         stepFourDataset->Scan(scanExpression.c_str(), "", "colsize=15");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         EXPECT_EQ(redirectOutput.str(), expectedScan);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

INSTANTIATE_TEST_SUITE_P(
   Run, GH20033SameBranchName,
   ::testing::Combine(
      // Whether to use TTree or TChain to represent one of the dataset layerss
      ::testing::Values(false, true), ::testing::Values(false, true), ::testing::Values(false, true),
      ::testing::Values(false, true),
      // Components of the TTree::Scan call
      ::testing::Values("", "step0tree", "step0chai"), ::testing::Values("", "step1tree", "step1chai"),
      ::testing::Values("", "step2tree", "step2chai"), ::testing::Values("", "step3tree", "step3chai"),
      ::testing::Values("", "step4tree", "step4chai"),
      // Expression to evaluate
      ::testing::Values("value", "genEvent.Weight()")),
   // Extra parenthesis around lambda to avoid preprocessor errors, see
   // https://stackoverflow.com/questions/79438894/lambda-with-structured-bindings-inside-macro-call
   ([](const testing::TestParamInfo<GH20033SameBranchName::ParamType> &paramInfo) {
      auto &&[stepOneTChain, stepTwoTChain, stepThreeTChain, stepFourTChain, stepZeroScan, stepOneScan, stepTwoScan,
              stepThreeScan, stepFourScan, expressionName] = paramInfo.param;
      // googletest only accepts ASCII alphanumeric characters for labels
      std::string label{};
      if (stepOneTChain)
         label += "StepOneTChain_";
      else
         label += "StepOneTTree_";
      if (stepTwoTChain)
         label += "StepTwoTChain_";
      else
         label += "StepTwoTTree_";
      if (stepThreeTChain)
         label += "StepThreeTChain_";
      else
         label += "StepThreeTTree_";
      if (stepFourTChain)
         label += "StepFourTChain_";
      else
         label += "StepFourTTree_";

      if (!stepZeroScan.empty())
         label += "StepZeroScan_" + stepZeroScan + "_";
      else
         label += "StepZeroNoScan_";
      if (!stepOneScan.empty())
         label += "StepOneScan_" + stepOneScan + "_";
      else
         label += "StepOneNoScan_";
      if (!stepTwoScan.empty())
         label += "StepTwoScan_" + stepTwoScan + "_";
      else
         label += "StepTwoNoScan_";
      if (!stepThreeScan.empty())
         label += "StepThreeScan_" + stepThreeScan + "_";
      else
         label += "StepThreeNoScan_";
      if (!stepFourScan.empty())
         label += "StepFourScan_" + stepFourScan + "_";
      else
         label += "StepFourNoScan_";

      if (expressionName == "value")
         label += "value";
      else
         label += "genEventWeight";

      return label;
   }));

// Another scenario where all datasets in the hierarchy have the same branch name but the different branches from the
// different datasets are accessed together in the same parts of a TTree::Scan expression. This checks that the various
// branch addresses are properly set and they don't get mixed up.
struct GH20033ComplexScanExpression : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool>> {

   constexpr static std::array<const char *, 3> fStepZeroFileNames{
      "tree_gh20033_complexscanexpression_stepzero_0.root", "tree_gh20033_complexscanexpression_stepzero_1.root",
      "tree_gh20033_complexscanexpression_stepzero_2.root"};

   constexpr static auto fStepOneFileName{"tree_gh20033_complexscanexpression_stepone.root"};
   constexpr static auto fStepTwoFileName{"tree_gh20033_complexscanexpression_steptwo.root"};
   constexpr static auto fStepThreeFileName{"tree_gh20033_complexscanexpression_stepthree.root"};
   constexpr static auto fStepFourFileName{"tree_gh20033_complexscanexpression_stepfour.root"};

   constexpr static auto fStepZeroTreeName{"stepzerotree"};
   constexpr static auto fStepOneTreeName{"steponetree"};
   constexpr static auto fStepTwoTreeName{"steptwotree"};
   constexpr static auto fStepThreeTreeName{"stepthreetree"};
   constexpr static auto fStepFourTreeName{"stepfourtree"};

   constexpr static auto fTotalEntries{10};

   static void WriteData(const char *name, const char *treename, int first, int last)
   {
      auto file = std::make_unique<TFile>(name, "RECREATE");
      auto tree = std::make_unique<TTree>(treename, treename);

      int val{};
      tree->Branch("value", &val);

      for (val = first; val < last; ++val) {
         tree->Fill();
      }

      file->Write();
   }

   static void SetUpTestSuite()
   {
      WriteData(fStepZeroFileNames[0], fStepZeroTreeName, 0, 4);
      WriteData(fStepZeroFileNames[1], fStepZeroTreeName, 4, 7);
      WriteData(fStepZeroFileNames[2], fStepZeroTreeName, 7, 10);
      WriteData(fStepOneFileName, fStepOneTreeName, 100, 100 + fTotalEntries);
      WriteData(fStepTwoFileName, fStepTwoTreeName, 200, 200 + fTotalEntries);
      WriteData(fStepThreeFileName, fStepThreeTreeName, 300, 300 + fTotalEntries);
      WriteData(fStepFourFileName, fStepFourTreeName, 400, 400 + fTotalEntries);
   }

   static void TearDownTestSuite()
   {
      for (const auto &fileName : fStepZeroFileNames)
         std::remove(fileName);

      std::remove(fStepOneFileName);
      std::remove(fStepTwoFileName);
      std::remove(fStepThreeFileName);
      std::remove(fStepFourFileName);
   }
};

TEST_P(GH20033ComplexScanExpression, Test)
{

   auto &&[stepOneTChain, stepTwoTChain, stepThreeTChain, stepFourTChain] = GetParam();

   // The step zero dataset needs to be a TChain and have more than one TTree
   // to recreate the scenarios when one component of the friendship chain
   // is a TChain switching to a new TTree and thus triggering the update
   auto stepZeroDataset = std::make_unique<TChain>(fStepZeroTreeName);
   for (const auto &fileName : fStepZeroFileNames)
      stepZeroDataset->Add(fileName);

   // Step one dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepOneFileLifeLine{};
   std::unique_ptr<TTree> stepOneDataset{};
   if (stepOneTChain) {
      auto chain = std::make_unique<TChain>(fStepOneTreeName);
      chain->Add(fStepOneFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepZeroDataset.get());
      stepOneDataset = std::move(chain);
   } else {
      stepOneFileLifeLine = std::make_unique<TFile>(fStepOneFileName);
      stepOneDataset = std::unique_ptr<TTree>{stepOneFileLifeLine->Get<TTree>(fStepOneTreeName)};
      stepOneDataset->AddFriend(stepZeroDataset.get());
   }

   // Step two dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepTwoFileLifeLine{};
   std::unique_ptr<TTree> stepTwoDataset{};
   if (stepTwoTChain) {
      auto chain = std::make_unique<TChain>(fStepTwoTreeName);
      chain->Add(fStepTwoFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepOneDataset.get());
      stepTwoDataset = std::move(chain);
   } else {
      stepTwoFileLifeLine = std::make_unique<TFile>(fStepTwoFileName);
      stepTwoDataset = std::unique_ptr<TTree>{stepTwoFileLifeLine->Get<TTree>(fStepTwoTreeName)};
      stepTwoDataset->AddFriend(stepOneDataset.get());
   }

   // Step three dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepThreeFileLifeLine{};
   std::unique_ptr<TTree> stepThreeDataset{};
   if (stepThreeTChain) {
      auto chain = std::make_unique<TChain>(fStepThreeTreeName);
      chain->Add(fStepThreeFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepTwoDataset.get());
      stepThreeDataset = std::move(chain);
   } else {
      stepThreeFileLifeLine = std::make_unique<TFile>(fStepThreeFileName);
      stepThreeDataset = std::unique_ptr<TTree>{stepThreeFileLifeLine->Get<TTree>(fStepThreeTreeName)};
      stepThreeDataset->AddFriend(stepTwoDataset.get());
   }

   // Step four dataset, can be a TChain or a standalone TTree.
   // In the former case, the first tree of the TChain is used
   // to befriend the chain from the previous step.
   std::unique_ptr<TFile> stepFourFileLifeLine{};
   std::unique_ptr<TTree> stepFourDataset{};
   if (stepFourTChain) {
      auto chain = std::make_unique<TChain>(fStepFourTreeName);
      chain->Add(fStepFourFileName);
      chain->GetEntry(0);
      chain->GetTree()->AddFriend(stepThreeDataset.get());
      stepFourDataset = std::move(chain);
   } else {
      stepFourFileLifeLine = std::make_unique<TFile>(fStepFourFileName);
      stepFourDataset = std::unique_ptr<TTree>{stepFourFileLifeLine->Get<TTree>(fStepFourTreeName)};
      stepFourDataset->AddFriend(stepThreeDataset.get());
   }

   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(stepFourDataset->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"tree_gh20033_complexscanexpression_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         stepFourDataset->Scan("stepfourtree.value * stepzerotree.value:steponetree.value + steptwotree.value:"
                               "stepthreetree.value - stepzerotree.value",
                               "stepthreetree.value > 303", "colsize=50");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         EXPECT_EQ(
            redirectOutput.str(),
            R"Scan(***************************************************************************************************************************************************************************
*    Row   *            stepfourtree.value * stepzerotree.value *              steponetree.value + steptwotree.value *           stepthreetree.value - stepzerotree.value *
***************************************************************************************************************************************************************************
*        4 *                                               1616 *                                                308 *                                                300 *
*        5 *                                               2025 *                                                310 *                                                300 *
*        6 *                                               2436 *                                                312 *                                                300 *
*        7 *                                               2849 *                                                314 *                                                300 *
*        8 *                                               3264 *                                                316 *                                                300 *
*        9 *                                               3681 *                                                318 *                                                300 *
***************************************************************************************************************************************************************************
)Scan");
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

INSTANTIATE_TEST_SUITE_P(
   Run, GH20033ComplexScanExpression,
   ::testing::Combine(::testing::Values(false, true), ::testing::Values(false, true), ::testing::Values(false, true),
                      ::testing::Values(false, true)),
   // Extra parenthesis around lambda to avoid preprocessor errors, see
   // https://stackoverflow.com/questions/79438894/lambda-with-structured-bindings-inside-macro-call
   ([](const testing::TestParamInfo<GH20033ComplexScanExpression::ParamType> &paramInfo) {
      auto &&[stepOneTChain, stepTwoTChain, stepThreeTChain, stepFourTChain] = paramInfo.param;
      // googletest only accepts ASCII alphanumeric characters for labels
      std::string label{};
      if (stepOneTChain)
         label += "StepOneTChain_";
      else
         label += "StepOneTTree_";
      if (stepTwoTChain)
         label += "StepTwoTChain_";
      else
         label += "StepTwoTTree_";
      if (stepThreeTChain)
         label += "StepThreeTChain_";
      else
         label += "StepThreeTTree_";
      if (stepFourTChain)
         label += "StepFourTChain";
      else
         label += "StepFourTTree";
      return label;
   }));
