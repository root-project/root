#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreePlayer.h>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <fstream>

#include <gtest/gtest.h>

struct GH16805Regression : public ::testing::Test {

   const static inline std::vector<std::string> fStepZeroFileNames{"gh16805_regression_stepzero_0.root",
                                                                   "gh16805_regression_stepzero_1.root"};
   const static inline std::vector<std::pair<int, int>> fStepZeroEntryRanges{{0, 10}, {10, 20}};
   constexpr static auto fStepZeroTreeName{"stepzero"};

   const static inline std::vector<std::string> fTopLevelFriendFileNames{"gh16805_regression_toplevelfriend_0.root",
                                                                         "gh16805_regression_toplevelfriend_1.root",
                                                                         "gh16805_regression_toplevelfriend_2.root"};
   const static inline std::vector<std::pair<int, int>> fTopLevelFriendEntryRanges{{200, 207}, {207, 214}, {214, 220}};
   constexpr static auto fTopLevelFriendTreeName{"topLevelFriend"};

   constexpr static auto fStepOneFileName{"gh16805_regression_stepone.root"};
   constexpr static std::pair<int, int> fStepOneEntryRange{100, 120};
   constexpr static auto fStepOneTreeName{"stepone"};

   static void writeStepZero(const char *treeName, const char *fileName, int begin, int end)
   {
      auto file = std::make_unique<TFile>(fileName, "recreate");
      auto tree = std::make_unique<TTree>(treeName, treeName);
      int stepZeroBr1, stepZeroBr2 = 0;
      tree->Branch("stepZeroBr1", &stepZeroBr1);
      tree->Branch("stepZeroBr2", &stepZeroBr2);

      for (stepZeroBr1 = begin, stepZeroBr2 = begin * 2; stepZeroBr1 < end;
           ++stepZeroBr1, stepZeroBr2 = stepZeroBr1 * 2)
         tree->Fill();
      file->Write();
   }

   static void writeStepOne(const char *treeName, const char *fileName, int begin, int end)
   {
      auto file = std::make_unique<TFile>(fileName, "recreate");
      auto tree = std::make_unique<TTree>(treeName, treeName);
      int br1 = -1;
      tree->Branch("stepOneBr1", &br1);

      for (br1 = begin; br1 < end; ++br1)
         tree->Fill();

      file->Write();
   }

   static void writeTopLevelFriend(const char *treeName, const char *fileName, int begin, int end)
   {
      auto file = std::make_unique<TFile>(fileName, "recreate");
      auto tree = std::make_unique<TTree>(treeName, treeName);
      int friendBr1, friendBr2 = 0;
      tree->Branch("friendBr1", &friendBr1);
      tree->Branch("friendBr2", &friendBr2);

      for (friendBr1 = begin, friendBr2 = begin * 2; friendBr1 < end; ++friendBr1, friendBr2 = friendBr1 * 2)
         tree->Fill();
      file->Write();
   }

   static void SetUpTestSuite()
   {
      for (decltype(fStepZeroFileNames.size()) i = 0; i < fStepZeroFileNames.size(); i++) {
         writeStepZero(fStepZeroTreeName, fStepZeroFileNames[i].c_str(), fStepZeroEntryRanges[i].first,
                       fStepZeroEntryRanges[i].second);
      }

      for (decltype(fTopLevelFriendFileNames.size()) i = 0; i < fTopLevelFriendFileNames.size(); i++) {
         writeTopLevelFriend(fTopLevelFriendTreeName, fTopLevelFriendFileNames[i].c_str(),
                             fTopLevelFriendEntryRanges[i].first, fTopLevelFriendEntryRanges[i].second);
      }
      writeStepOne(fStepOneTreeName, fStepOneFileName, fStepOneEntryRange.first, fStepOneEntryRange.second);
   }

   static void TearDownTestSuite()
   {
      for (const auto &f : fStepZeroFileNames)
         std::remove(f.c_str());

      for (const auto &f : fTopLevelFriendFileNames)
         std::remove(f.c_str());

      std::remove(fStepOneFileName);
   }
};

TEST_F(GH16805Regression, Test)
{
   // Tests that a chain is able to correctly update branch addresses of its friends,
   // whether the friends are attached to the chain itself or to its first tree.

   auto stepZeroChain = std::make_unique<TChain>(fStepZeroTreeName);
   for (const auto &f : fStepZeroFileNames)
      stepZeroChain->Add(f.c_str());

   auto stepOneChain = std::make_unique<TChain>(fStepOneTreeName);
   stepOneChain->Add(fStepOneFileName);

   auto topLevelFriendChain = std::make_unique<TChain>(fTopLevelFriendTreeName);
   for (const auto &f : fTopLevelFriendFileNames)
      topLevelFriendChain->Add(f.c_str());
   stepOneChain->AddFriend(topLevelFriendChain.get());

   // Load the first tree in the chain
   stepOneChain->LoadTree(0);
   auto stepOneFirstTree = stepOneChain->GetTree()->GetTree();
   ASSERT_NE(stepOneFirstTree, nullptr);

   stepOneFirstTree->AddFriend(stepZeroChain.get());

   // Now iterate over the chain and check the contents of the branches
   // inherited from the friends.
   int stepZeroBr1 = -1, stepZeroBr2 = -1, stepOneBr1 = -1, friendBr1 = -1, friendBr2 = -1;
   ASSERT_EQ(stepOneChain->SetBranchAddress("stepZeroBr1", &stepZeroBr1), 0);
   ASSERT_EQ(stepOneChain->SetBranchAddress("stepZeroBr2", &stepZeroBr2), 0);
   ASSERT_EQ(stepOneChain->SetBranchAddress("stepOneBr1", &stepOneBr1), 0);
   ASSERT_EQ(stepOneChain->SetBranchAddress("friendBr1", &friendBr1), 0);
   ASSERT_EQ(stepOneChain->SetBranchAddress("friendBr2", &friendBr2), 0);

   std::vector<int> expectedStepZeroBr1(20);
   std::vector<int> expectedStepZeroBr2(20);
   std::vector<int> expectedStepOneBr1(20);
   std::vector<int> expectedTopLevelFriendBr1(20);
   std::vector<int> expectedTopLevelFriendBr2(20);

   std::iota(expectedStepZeroBr1.begin(), expectedStepZeroBr1.end(), 0);
   std::generate(expectedStepZeroBr2.begin(), expectedStepZeroBr2.end(), [n = 0]() mutable {
      auto res = n * 2;
      n++;
      return res;
   });
   std::iota(expectedStepOneBr1.begin(), expectedStepOneBr1.end(), 100);
   std::iota(expectedTopLevelFriendBr1.begin(), expectedTopLevelFriendBr1.end(), 200);
   std::generate(expectedTopLevelFriendBr2.begin(), expectedTopLevelFriendBr2.end(), [n = 200]() mutable {
      auto res = n * 2;
      n++;
      return res;
   });

   for (Long64_t i = 0; i < stepOneChain->GetEntriesFast(); ++i) {
      stepOneChain->GetEntry(i);
      EXPECT_EQ(expectedStepZeroBr1[i], stepZeroBr1);
      EXPECT_EQ(expectedStepZeroBr2[i], stepZeroBr2);
      EXPECT_EQ(expectedStepOneBr1[i], stepOneBr1);
      EXPECT_EQ(expectedTopLevelFriendBr1[i], friendBr1);
      EXPECT_EQ(expectedTopLevelFriendBr2[i], friendBr2);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(stepOneChain->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16805_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         stepOneChain->Scan("stepZeroBr1:stepZeroBr2:stepOneBr1:friendBr1:friendBr2", "", "colsize=12");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(***************************************************************************************
*    Row   *  stepZeroBr1 *  stepZeroBr2 *   stepOneBr1 *    friendBr1 *    friendBr2 *
***************************************************************************************
*        0 *            0 *            0 *          100 *          200 *          400 *
*        1 *            1 *            2 *          101 *          201 *          402 *
*        2 *            2 *            4 *          102 *          202 *          404 *
*        3 *            3 *            6 *          103 *          203 *          406 *
*        4 *            4 *            8 *          104 *          204 *          408 *
*        5 *            5 *           10 *          105 *          205 *          410 *
*        6 *            6 *           12 *          106 *          206 *          412 *
*        7 *            7 *           14 *          107 *          207 *          414 *
*        8 *            8 *           16 *          108 *          208 *          416 *
*        9 *            9 *           18 *          109 *          209 *          418 *
*       10 *           10 *           20 *          110 *          210 *          420 *
*       11 *           11 *           22 *          111 *          211 *          422 *
*       12 *           12 *           24 *          112 *          212 *          424 *
*       13 *           13 *           26 *          113 *          213 *          426 *
*       14 *           14 *           28 *          114 *          214 *          428 *
*       15 *           15 *           30 *          115 *          215 *          430 *
*       16 *           16 *           32 *          116 *          216 *          432 *
*       17 *           17 *           34 *          117 *          217 *          434 *
*       18 *           18 *           36 *          118 *          218 *          436 *
*       19 *           19 *           38 *          119 *          219 *          438 *
***************************************************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}
