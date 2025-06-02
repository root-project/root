#include <cstdio>
#include <fstream>
#include <numeric>

#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>
#include <TTreePlayer.h>
#include <TEntryList.h>
#include <TDirectory.h>

#include "gtest/gtest.h"
#include "ROOT/TestSupport.hxx"

#include <iostream>
// Test regression for https://github.com/root-project/root/issues/16804
struct RegresssionGH16804 : public ::testing::Test {

   constexpr static inline std::array<const char *, 2> fFriendFileNames{"gh16804_friend_0.root",
                                                                        "gh16804_friend_1.root"};
   constexpr static inline std::array<const char *, 4> fOtherFriendFileNames{
      "gh16804_otherfriend_0.root", "gh16804_otherfriend_1.root", "gh16804_otherfriend_2.root",
      "gh16804_otherfriend_3.root"};
   constexpr static inline const char *fMainFileName{"gh16804_main.root"};
   constexpr static inline const char *fMainTreeName{"mainTree"};
   constexpr static inline const char *fFriendTreeName{"friendTree"};
   constexpr static inline const char *fOtherFriendTreeName{"otherFriendTree"};

   static void CreateMainTree()
   {
      int begin{};
      int end{20};

      auto file = std::make_unique<TFile>(fMainFileName, "RECREATE");
      auto tree = std::make_unique<TTree>(fMainTreeName, fMainTreeName);

      int mainBranch{};
      tree->Branch("mainBranch", &mainBranch);

      // Sequential entries in reverse order from 19 (inclusive) to 0 (inclusive)
      for (mainBranch = end - 1; mainBranch > begin - 1; mainBranch--)
         tree->Fill();

      file->Write();
   }

   static void CreateFriendTrees()
   {
      int begin{};
      int end{10};
      for (const auto &fn : fFriendFileNames) {
         auto file = std::make_unique<TFile>(fn, "RECREATE");
         auto tree = std::make_unique<TTree>(fFriendTreeName, fFriendTreeName);

         int index{};
         int value{100 + begin};
         tree->Branch("index", &index);
         tree->Branch("value", &value);

         for (index = begin; index < end; ++index) {
            tree->Fill();
            value++;
         }

         file->Write();
         begin += 10;
         end += 10;
      }
   }

   static void CreateOtherFriendTrees()
   {
      int begin{200};
      int end{205};
      for (const auto &fn : fOtherFriendFileNames) {
         auto file = std::make_unique<TFile>(fn, "RECREATE");
         auto tree = std::make_unique<TTree>(fOtherFriendTreeName, fOtherFriendTreeName);

         int a{};
         int b{100 + begin};
         tree->Branch("a", &a);
         tree->Branch("b", &b);

         for (a = begin; a < end; a++) {
            tree->Fill();
            b++;
         }

         file->Write();
         begin += 5;
         end += 5;
      }
   }

   static void SetUpTestSuite()
   {
      CreateMainTree();
      CreateFriendTrees();
      CreateOtherFriendTrees();
   }

   static void TearDownTestSuite()
   {
      for (const auto &fn : fFriendFileNames)
         std::remove(fn);
      for (const auto &fn : fOtherFriendFileNames)
         std::remove(fn);
      std::remove(fMainFileName);
   }
};

TEST_F(RegresssionGH16804, MainTTreeFriendTChain)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(&friendChain);

   int index = -1, value = -1, mainBranch = -1;
   auto mainBranchRet = mainTree->SetBranchAddress("mainBranch", &mainBranch);
   auto indexRet = mainTree->SetBranchAddress("index", &index);
   auto valueRet = mainTree->SetBranchAddress("value", &value);
   EXPECT_GE(mainBranchRet, 0);
   EXPECT_GE(indexRet, 0);
   EXPECT_GE(valueRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedIndex(20);
   std::vector<int> expectedValue(20);

   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedIndex.begin(), expectedIndex.end(), 0);
   std::iota(expectedValue.begin(), expectedValue.end(), 100);

   auto nEntries = mainTree->GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainTree->GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedIndex[i], index);
      EXPECT_EQ(expectedValue[i], value);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainTree->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainTree->Scan("mainBranch:index:value");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************
*    Row   * mainBranc *     index *     value *
************************************************
*        0 *        19 *         0 *       100 *
*        1 *        18 *         1 *       101 *
*        2 *        17 *         2 *       102 *
*        3 *        16 *         3 *       103 *
*        4 *        15 *         4 *       104 *
*        5 *        14 *         5 *       105 *
*        6 *        13 *         6 *       106 *
*        7 *        12 *         7 *       107 *
*        8 *        11 *         8 *       108 *
*        9 *        10 *         9 *       109 *
*       10 *         9 *        10 *       110 *
*       11 *         8 *        11 *       111 *
*       12 *         7 *        12 *       112 *
*       13 *         6 *        13 *       113 *
*       14 *         5 *        14 *       114 *
*       15 *         4 *        15 *       115 *
*       16 *         3 *        16 *       116 *
*       17 *         2 *        17 *       117 *
*       18 *         1 *        18 *       118 *
*       19 *         0 *        19 *       119 *
************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegresssionGH16804, WrongBranchNameTTreeFriendTChain)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(&friendChain);

   int wrong = -1;
   auto wrongBranchRetBeforeLoadTree = mainTree->SetBranchAddress("wrong", &wrong);
   // SetBranchAddress did not find the branch in the mainTree, then tried
   // to find it in the list of friends. The friend TChain hasn't loaded its
   // tree yet, so it returns kNoCheck==5
   EXPECT_EQ(wrongBranchRetBeforeLoadTree, 5);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TChain::SetBranchAddress", "unknown branch -> wrong");
   diagRAII.requiredDiag(kError, "TTree::SetBranchAddress", "unknown branch -> wrong");

   mainTree->LoadTree(0);
   auto wrongBranchRetAfterLoadTree = mainTree->SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRetAfterLoadTree, -5);
}

TEST_F(RegresssionGH16804, WrongBranchNameTTreeFriendTTree)
{
   auto friendFile = std::make_unique<TFile>(fFriendFileNames[0]);
   auto friendTree = friendFile->Get<TTree>(fFriendTreeName);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(friendTree);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TTree::SetBranchAddress", "unknown branch -> wrong");
   int wrong = -1;
   auto wrongBranchRetBeforeLoadTree = mainTree->SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRetBeforeLoadTree, -5);
}

TEST_F(RegresssionGH16804, WrongBranchNameTChainFriendTChain)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);

   TChain mainChain{fMainTreeName};
   mainChain.Add(fMainFileName);
   mainChain.AddFriend(&friendChain);

   int wrong = -1;
   auto wrongBranchRetBeforeLoadTree = mainChain.SetBranchAddress("wrong", &wrong);
   // SetBranchAddress did not find the branch in the mainTree, then tried
   // to find it in the list of friends. The friend TChain hasn't loaded its
   // tree yet, so it returns kNoCheck==5
   EXPECT_EQ(wrongBranchRetBeforeLoadTree, 5);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TChain::SetBranchAddress", "unknown branch -> wrong");

   mainChain.LoadTree(0);
   auto wrongBranchRetAfterLoadTree = mainChain.SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRetAfterLoadTree, -5);
}

TEST_F(RegresssionGH16804, WrongBranchNameTChainFriendTTree)
{
   auto friendFile = std::make_unique<TFile>(fMainFileName);
   auto friendTree = friendFile->Get<TTree>(fMainTreeName);

   TChain mainChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      mainChain.Add(fn);
   mainChain.AddFriend(friendTree);

   int wrong = -1;
   auto wrongBranchRetBeforeLoadTree = mainChain.SetBranchAddress("wrong", &wrong);
   // SetBranchAddress did not find the branch in the mainTree, then tried
   // to find it in the list of friends. The friend TChain hasn't loaded its
   // tree yet, so it returns kNoCheck==5
   EXPECT_EQ(wrongBranchRetBeforeLoadTree, 5);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TChain::SetBranchAddress", "unknown branch -> wrong");

   mainChain.LoadTree(0);
   auto wrongBranchRetAfterLoadTree = mainChain.SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRetAfterLoadTree, -5);
}

TEST_F(RegresssionGH16804, WrongBranchNameTTreeTwoFriendTChains)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);
   TChain otherFriendChain{fOtherFriendTreeName};
   for (const auto &fn : fOtherFriendFileNames)
      otherFriendChain.Add(fn);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(&friendChain);
   mainTree->AddFriend(&otherFriendChain);

   int wrong = -1;
   auto wrongBranchRetBeforeLoadTree = mainTree->SetBranchAddress("wrong", &wrong);
   // SetBranchAddress did not find the branch in the mainTree, then tried
   // to find it in the list of friends. The friend TChains haven not their
   // trees yet, so the return value is kNoCheck==5
   EXPECT_EQ(wrongBranchRetBeforeLoadTree, 5);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TChain::SetBranchAddress", "unknown branch -> wrong");
   diagRAII.requiredDiag(kError, "TTree::SetBranchAddress", "unknown branch -> wrong");

   mainTree->LoadTree(0);
   auto wrongBranchRetAfterLoadTree = mainTree->SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRetAfterLoadTree, -5);
}

TEST_F(RegresssionGH16804, TChainFriendTChain)
{
   TChain mainChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      mainChain.Add(fn);

   TChain friendChain{fOtherFriendTreeName};
   for (const auto &fn : fOtherFriendFileNames)
      friendChain.Add(fn);

   mainChain.AddFriend(&friendChain);

   int index = -1;
   int value = -1;
   int a = -1;
   int b = -1;

   auto indexRet = mainChain.SetBranchAddress("index", &index);
   auto valueRet = mainChain.SetBranchAddress("value", &value);
   auto aRet = mainChain.SetBranchAddress("a", &a);
   auto bRet = mainChain.SetBranchAddress("b", &b);
   EXPECT_GE(indexRet, 0);
   EXPECT_GE(valueRet, 0);
   EXPECT_GE(aRet, 0);
   EXPECT_GE(bRet, 0);

   std::vector<int> expectedIndex(20);
   std::vector<int> expectedValue(20);
   std::vector<int> expectedA(20);
   std::vector<int> expectedB(20);

   std::iota(expectedIndex.begin(), expectedIndex.end(), 0);
   std::iota(expectedValue.begin(), expectedValue.end(), 100);
   std::iota(expectedA.begin(), expectedA.end(), 200);
   std::iota(expectedB.begin(), expectedB.end(), 300);

   for (Long64_t i = 0; i < mainChain.GetEntriesFast(); ++i) {
      mainChain.GetEntry(i);
      EXPECT_EQ(expectedIndex[i], index);
      EXPECT_EQ(expectedValue[i], value);
      EXPECT_EQ(expectedA[i], a);
      EXPECT_EQ(expectedB[i], b);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainChain.GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_chainfriendchain_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainChain.Scan("index:value:a:b");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************************
*    Row   *     index *     value *         a *         b *
************************************************************
*        0 *         0 *       100 *       200 *       300 *
*        1 *         1 *       101 *       201 *       301 *
*        2 *         2 *       102 *       202 *       302 *
*        3 *         3 *       103 *       203 *       303 *
*        4 *         4 *       104 *       204 *       304 *
*        5 *         5 *       105 *       205 *       305 *
*        6 *         6 *       106 *       206 *       306 *
*        7 *         7 *       107 *       207 *       307 *
*        8 *         8 *       108 *       208 *       308 *
*        9 *         9 *       109 *       209 *       309 *
*       10 *        10 *       110 *       210 *       310 *
*       11 *        11 *       111 *       211 *       311 *
*       12 *        12 *       112 *       212 *       312 *
*       13 *        13 *       113 *       213 *       313 *
*       14 *        14 *       114 *       214 *       314 *
*       15 *        15 *       115 *       215 *       315 *
*       16 *        16 *       116 *       216 *       316 *
*       17 *        17 *       117 *       217 *       317 *
*       18 *        18 *       118 *       218 *       318 *
*       19 *        19 *       119 *       219 *       319 *
************************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegresssionGH16804, TTreeTwoFriendTChains)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);
   TChain otherFriendChain{fOtherFriendTreeName};
   for (const auto &fn : fOtherFriendFileNames)
      otherFriendChain.Add(fn);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(&friendChain);
   mainTree->AddFriend(&otherFriendChain);

   int index = -1, value = -1, mainBranch = -1, a = -1, b = -1;
   auto mainBranchRet = mainTree->SetBranchAddress("mainBranch", &mainBranch);
   auto indexRet = mainTree->SetBranchAddress("index", &index);
   auto valueRet = mainTree->SetBranchAddress("value", &value);
   auto aRet = mainTree->SetBranchAddress("a", &a);
   auto bRet = mainTree->SetBranchAddress("b", &b);
   EXPECT_GE(mainBranchRet, 0);
   EXPECT_GE(indexRet, 0);
   EXPECT_GE(valueRet, 0);
   EXPECT_GE(aRet, 0);
   EXPECT_GE(bRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedIndex(20);
   std::vector<int> expectedValue(20);
   std::vector<int> expectedA(20);
   std::vector<int> expectedB(20);
   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedIndex.begin(), expectedIndex.end(), 0);
   std::iota(expectedValue.begin(), expectedValue.end(), 100);
   std::iota(expectedA.begin(), expectedA.end(), 200);
   std::iota(expectedB.begin(), expectedB.end(), 300);

   auto nEntries = mainTree->GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainTree->GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedIndex[i], index);
      EXPECT_EQ(expectedValue[i], value);
      EXPECT_EQ(expectedA[i], a);
      EXPECT_EQ(expectedB[i], b);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainTree->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_ttreetwofriendtchains_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainTree->Scan("mainBranch:index:value:a:b");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************************************
*    Row   * mainBranc *     index *     value *         a *         b *
************************************************************************
*        0 *        19 *         0 *       100 *       200 *       300 *
*        1 *        18 *         1 *       101 *       201 *       301 *
*        2 *        17 *         2 *       102 *       202 *       302 *
*        3 *        16 *         3 *       103 *       203 *       303 *
*        4 *        15 *         4 *       104 *       204 *       304 *
*        5 *        14 *         5 *       105 *       205 *       305 *
*        6 *        13 *         6 *       106 *       206 *       306 *
*        7 *        12 *         7 *       107 *       207 *       307 *
*        8 *        11 *         8 *       108 *       208 *       308 *
*        9 *        10 *         9 *       109 *       209 *       309 *
*       10 *         9 *        10 *       110 *       210 *       310 *
*       11 *         8 *        11 *       111 *       211 *       311 *
*       12 *         7 *        12 *       112 *       212 *       312 *
*       13 *         6 *        13 *       113 *       213 *       313 *
*       14 *         5 *        14 *       114 *       214 *       314 *
*       15 *         4 *        15 *       115 *       215 *       315 *
*       16 *         3 *        16 *       116 *       216 *       316 *
*       17 *         2 *        17 *       117 *       217 *       317 *
*       18 *         1 *        18 *       118 *       218 *       318 *
*       19 *         0 *        19 *       119 *       219 *       319 *
************************************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegresssionGH16804, TChainTwoFriendTChains)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);
   TChain otherFriendChain{fOtherFriendTreeName};
   for (const auto &fn : fOtherFriendFileNames)
      otherFriendChain.Add(fn);

   TChain mainChain{fMainTreeName};
   mainChain.Add(fMainFileName);

   mainChain.AddFriend(&friendChain);
   mainChain.AddFriend(&otherFriendChain);

   int index = -1, value = -1, mainBranch = -1, a = -1, b = -1;
   auto mainBranchRet = mainChain.SetBranchAddress("mainBranch", &mainBranch);
   auto indexRet = mainChain.SetBranchAddress("index", &index);
   auto valueRet = mainChain.SetBranchAddress("value", &value);
   auto aRet = mainChain.SetBranchAddress("a", &a);
   auto bRet = mainChain.SetBranchAddress("b", &b);
   EXPECT_GE(mainBranchRet, 0);
   EXPECT_GE(indexRet, 0);
   EXPECT_GE(valueRet, 0);
   EXPECT_GE(aRet, 0);
   EXPECT_GE(bRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedIndex(20);
   std::vector<int> expectedValue(20);
   std::vector<int> expectedA(20);
   std::vector<int> expectedB(20);
   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedIndex.begin(), expectedIndex.end(), 0);
   std::iota(expectedValue.begin(), expectedValue.end(), 100);
   std::iota(expectedA.begin(), expectedA.end(), 200);
   std::iota(expectedB.begin(), expectedB.end(), 300);

   for (Long64_t i = 0; i < mainChain.GetEntriesFast(); ++i) {
      mainChain.GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedIndex[i], index);
      EXPECT_EQ(expectedValue[i], value);
      EXPECT_EQ(expectedA[i], a);
      EXPECT_EQ(expectedB[i], b);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainChain.GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_tchaintwofriendtchains_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainChain.Scan("mainBranch:index:value:a:b");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************************************
*    Row   * mainBranc *     index *     value *         a *         b *
************************************************************************
*        0 *        19 *         0 *       100 *       200 *       300 *
*        1 *        18 *         1 *       101 *       201 *       301 *
*        2 *        17 *         2 *       102 *       202 *       302 *
*        3 *        16 *         3 *       103 *       203 *       303 *
*        4 *        15 *         4 *       104 *       204 *       304 *
*        5 *        14 *         5 *       105 *       205 *       305 *
*        6 *        13 *         6 *       106 *       206 *       306 *
*        7 *        12 *         7 *       107 *       207 *       307 *
*        8 *        11 *         8 *       108 *       208 *       308 *
*        9 *        10 *         9 *       109 *       209 *       309 *
*       10 *         9 *        10 *       110 *       210 *       310 *
*       11 *         8 *        11 *       111 *       211 *       311 *
*       12 *         7 *        12 *       112 *       212 *       312 *
*       13 *         6 *        13 *       113 *       213 *       313 *
*       14 *         5 *        14 *       114 *       214 *       314 *
*       15 *         4 *        15 *       115 *       215 *       315 *
*       16 *         3 *        16 *       116 *       216 *       316 *
*       17 *         2 *        17 *       117 *       217 *       317 *
*       18 *         1 *        18 *       118 *       218 *       318 *
*       19 *         0 *        19 *       119 *       219 *       319 *
************************************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}
