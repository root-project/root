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
