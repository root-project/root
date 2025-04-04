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
struct RegressionGH16804 : public ::testing::Test {

   constexpr static inline std::array<const char *, 2> fFriendFileNames{"gh16804_friend_0.root",
                                                                        "gh16804_friend_1.root"};
   constexpr static inline std::array<const char *, 4> fOtherFriendFileNames{
      "gh16804_otherfriend_0.root", "gh16804_otherfriend_1.root", "gh16804_otherfriend_2.root",
      "gh16804_otherfriend_3.root"};
   constexpr static inline const char *fMainFileName{"gh16804_main.root"};
   constexpr static inline const char *fMainTreeName{"mainTree"};
   constexpr static inline const char *fFriendTreeName{"friendTree"};
   constexpr static inline const char *fOtherFriendTreeName{"otherFriendTree"};
   constexpr static inline const char *fFriend20EntriesFileName1{"gh16804_friend_20entries_1.root"};
   constexpr static inline const char *fFriend20EntriesTreeName1{"friendTree20Entries_1"};
   constexpr static inline const char *fFriend20EntriesFileName2{"gh16804_friend_20entries_2.root"};
   constexpr static inline const char *fFriend20EntriesTreeName2{"friendTree20Entries_2"};

   static void CreateTree20Entries(const char *treeName, const char *fileName, const char *branchName)
   {
      int begin{};
      int end{20};

      auto file = std::make_unique<TFile>(fileName, "RECREATE");
      auto tree = std::make_unique<TTree>(treeName, treeName);

      int x{};
      tree->Branch(branchName, &x);

      // Sequential entries in reverse order from 19 (inclusive) to 0 (inclusive)
      for (x = end - 1; x > begin - 1; x--)
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
      CreateTree20Entries(fMainTreeName, fMainFileName, "mainBranch");
      CreateTree20Entries(fFriend20EntriesTreeName1, fFriend20EntriesFileName1, "friendBranch");
      CreateTree20Entries(fFriend20EntriesTreeName2, fFriend20EntriesFileName2, "yetAnotherFriendBranch");
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
      std::remove(fFriend20EntriesFileName1);
      std::remove(fFriend20EntriesFileName2);
   }
};

TEST_F(RegressionGH16804, GetBranchWrongName)
{
   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);

   ASSERT_EQ(mainTree->GetBranch("wrong"), nullptr);
}

TEST_F(RegressionGH16804, WrongBranchNameTTreeFriendTChain)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(&friendChain);

   int wrong = -1;
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TTree::SetBranchAddress", "unknown branch -> wrong");
   // SetBranchAddress loads the first tree in any friend TChain if it wasn't loaded before,
   // so the full dataset schema is known.
   auto wrongBranchRet = mainTree->SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRet, -5);
}

TEST_F(RegressionGH16804, WrongBranchNameTTreeFriendTTree)
{
   auto friendFile = std::make_unique<TFile>(fFriendFileNames[0]);
   auto friendTree = friendFile->Get<TTree>(fFriendTreeName);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(friendTree);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "TTree::SetBranchAddress", "unknown branch -> wrong");
   int wrong = -1;
   auto wrongBranchRet = mainTree->SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRet, -5);
}

TEST_F(RegressionGH16804, WrongBranchNameTChainFriendTChain)
{
   TChain friendChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      friendChain.Add(fn);

   TChain mainChain{fMainTreeName};
   mainChain.Add(fMainFileName);
   mainChain.AddFriend(&friendChain);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   // Branch name is not found in main chain nor in friend chain, SetBranchStatus should print error
   // SetBranchAddress loads the first tree in any friend TChain if it wasn't loaded before,
   // so the full dataset schema is known.
   diagRAII.requiredDiag(kError, "TChain::SetBranchAddress", "unknown branch -> wrong");
   int wrong = -1;
   auto wrongBranchRet = mainChain.SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRet, -5);
}

TEST_F(RegressionGH16804, WrongBranchNameTChainFriendTTree)
{
   auto friendFile = std::make_unique<TFile>(fMainFileName);
   auto friendTree = friendFile->Get<TTree>(fMainTreeName);

   TChain mainChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      mainChain.Add(fn);
   mainChain.AddFriend(friendTree);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   // Branch name is not found in main chain nor in friend chain, SetBranchStatus should print error
   diagRAII.requiredDiag(kError, "TChain::SetBranchAddress", "unknown branch -> wrong");
   int wrong = -1;
   auto wrongBranchRet = mainChain.SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRet, -5);
}

TEST_F(RegressionGH16804, WrongBranchNameTTreeTwoFriendTChains)
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

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   // Branch name is not found in main tree nor in friend chain, SetBranchStatus should print error
   // SetBranchAddress loads the first tree in any friend TChain if it wasn't loaded before,
   // so the full dataset schema is known.
   diagRAII.requiredDiag(kError, "TTree::SetBranchAddress", "unknown branch -> wrong");
   int wrong = -1;
   auto wrongBranchRet = mainTree->SetBranchAddress("wrong", &wrong);
   EXPECT_EQ(wrongBranchRet, -5);
}

TEST_F(RegressionGH16804, TTreeFriendTChain)
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

TEST_F(RegressionGH16804, TChainFriendTTree)
{
   TChain mainChain{fFriendTreeName};
   for (const auto &fn : fFriendFileNames)
      mainChain.Add(fn);

   auto friendFile = std::make_unique<TFile>(fMainFileName);
   auto friendTree = friendFile->Get<TTree>(fMainTreeName);
   mainChain.AddFriend(friendTree);

   int index = -1, value = -1, mainBranch = -1;
   auto mainBranchRet = mainChain.SetBranchAddress("mainBranch", &mainBranch);
   auto indexRet = mainChain.SetBranchAddress("index", &index);
   auto valueRet = mainChain.SetBranchAddress("value", &value);
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

   for (Long64_t i = 0; i < mainChain.GetEntriesFast(); ++i) {
      mainChain.GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedIndex[i], index);
      EXPECT_EQ(expectedValue[i], value);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainChain.GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_tchain_friend_ttree_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainChain.Scan("mainBranch:index:value");

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

TEST_F(RegressionGH16804, TTreeFriendTTree)
{
   auto friendFile = std::make_unique<TFile>(fFriend20EntriesFileName1);
   auto friendTree = friendFile->Get<TTree>(fFriend20EntriesTreeName1);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(friendTree);

   int mainBranch = -1, friendBranch = -1;
   auto mainBranchRet = mainTree->SetBranchAddress("mainBranch", &mainBranch);
   auto friendBranchRet = mainTree->SetBranchAddress("friendBranch", &friendBranch);
   ASSERT_EQ(mainBranchRet, 0);
   ASSERT_EQ(friendBranchRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedFriendBranch(20);

   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedFriendBranch.begin(), expectedFriendBranch.end(), 0);
   std::reverse(expectedFriendBranch.begin(), expectedFriendBranch.end());

   auto nEntries = mainTree->GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainTree->GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedFriendBranch[i], friendBranch);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainTree->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_ttree_friend_ttree_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainTree->Scan("mainBranch:friendBranch");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************
*    Row   * mainBranc * friendBra *
************************************
*        0 *        19 *        19 *
*        1 *        18 *        18 *
*        2 *        17 *        17 *
*        3 *        16 *        16 *
*        4 *        15 *        15 *
*        5 *        14 *        14 *
*        6 *        13 *        13 *
*        7 *        12 *        12 *
*        8 *        11 *        11 *
*        9 *        10 *        10 *
*       10 *         9 *         9 *
*       11 *         8 *         8 *
*       12 *         7 *         7 *
*       13 *         6 *         6 *
*       14 *         5 *         5 *
*       15 *         4 *         4 *
*       16 *         3 *         3 *
*       17 *         2 *         2 *
*       18 *         1 *         1 *
*       19 *         0 *         0 *
************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegressionGH16804, TChainFriendTChain)
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

TEST_F(RegressionGH16804, TTreeTwoFriendTChains)
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

TEST_F(RegressionGH16804, TChainTwoFriendTChains)
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

TEST_F(RegressionGH16804, TTreeTwoFriendTTrees)
{
   auto friendFile1 = std::make_unique<TFile>(fFriend20EntriesFileName1);
   auto friendTree1 = friendFile1->Get<TTree>(fFriend20EntriesTreeName1);
   auto friendFile2 = std::make_unique<TFile>(fFriend20EntriesFileName2);
   auto friendTree2 = friendFile2->Get<TTree>(fFriend20EntriesTreeName2);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(friendTree1);
   mainTree->AddFriend(friendTree2);

   int mainBranch = -1, friendBranch = -1, yetAnotherFriendBranch = -1;
   auto mainBranchRet = mainTree->SetBranchAddress("mainBranch", &mainBranch);
   auto yetAnotherFriendBranchRet = mainTree->SetBranchAddress("yetAnotherFriendBranch", &yetAnotherFriendBranch);
   auto friendBranchRet = mainTree->SetBranchAddress("friendBranch", &friendBranch);
   ASSERT_EQ(mainBranchRet, 0);
   ASSERT_EQ(friendBranchRet, 0);
   ASSERT_EQ(yetAnotherFriendBranchRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedFriendBranch(20);
   std::vector<int> expectedyetAnotherFriendBranch(20);

   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedFriendBranch.begin(), expectedFriendBranch.end(), 0);
   std::reverse(expectedFriendBranch.begin(), expectedFriendBranch.end());
   std::iota(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end(), 0);
   std::reverse(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end());

   auto nEntries = mainTree->GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainTree->GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedyetAnotherFriendBranch[i], yetAnotherFriendBranch);
      EXPECT_EQ(expectedFriendBranch[i], friendBranch);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainTree->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_ttree_two_friend_ttrees_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainTree->Scan("mainBranch:friendBranch:yetAnotherFriendBranch");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************
*    Row   * mainBranc * friendBra * yetAnothe *
************************************************
*        0 *        19 *        19 *        19 *
*        1 *        18 *        18 *        18 *
*        2 *        17 *        17 *        17 *
*        3 *        16 *        16 *        16 *
*        4 *        15 *        15 *        15 *
*        5 *        14 *        14 *        14 *
*        6 *        13 *        13 *        13 *
*        7 *        12 *        12 *        12 *
*        8 *        11 *        11 *        11 *
*        9 *        10 *        10 *        10 *
*       10 *         9 *         9 *         9 *
*       11 *         8 *         8 *         8 *
*       12 *         7 *         7 *         7 *
*       13 *         6 *         6 *         6 *
*       14 *         5 *         5 *         5 *
*       15 *         4 *         4 *         4 *
*       16 *         3 *         3 *         3 *
*       17 *         2 *         2 *         2 *
*       18 *         1 *         1 *         1 *
*       19 *         0 *         0 *         0 *
************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegressionGH16804, TTreeOneFriendTTreeOneFriendTChain)
{
   auto friendFile1 = std::make_unique<TFile>(fFriend20EntriesFileName1);
   auto friendTree1 = friendFile1->Get<TTree>(fFriend20EntriesTreeName1);

   TChain friendChain{fFriend20EntriesTreeName2};
   friendChain.Add(fFriend20EntriesFileName2);

   auto mainFile = std::make_unique<TFile>(fMainFileName);
   auto mainTree = mainFile->Get<TTree>(fMainTreeName);
   mainTree->AddFriend(friendTree1);
   mainTree->AddFriend(&friendChain);

   int mainBranch = -1, friendBranch = -1, yetAnotherFriendBranch = -1;
   auto mainBranchRet = mainTree->SetBranchAddress("mainBranch", &mainBranch);
   auto yetAnotherFriendBranchRet = mainTree->SetBranchAddress("yetAnotherFriendBranch", &yetAnotherFriendBranch);
   auto friendBranchRet = mainTree->SetBranchAddress("friendBranch", &friendBranch);
   ASSERT_EQ(mainBranchRet, 0);
   ASSERT_EQ(friendBranchRet, 0);
   ASSERT_EQ(yetAnotherFriendBranchRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedFriendBranch(20);
   std::vector<int> expectedyetAnotherFriendBranch(20);

   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedFriendBranch.begin(), expectedFriendBranch.end(), 0);
   std::reverse(expectedFriendBranch.begin(), expectedFriendBranch.end());
   std::iota(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end(), 0);
   std::reverse(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end());

   auto nEntries = mainTree->GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainTree->GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedyetAnotherFriendBranch[i], yetAnotherFriendBranch);
      EXPECT_EQ(expectedFriendBranch[i], friendBranch);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainTree->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_ttree_two_friend_ttrees_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainTree->Scan("mainBranch:friendBranch:yetAnotherFriendBranch");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************
*    Row   * mainBranc * friendBra * yetAnothe *
************************************************
*        0 *        19 *        19 *        19 *
*        1 *        18 *        18 *        18 *
*        2 *        17 *        17 *        17 *
*        3 *        16 *        16 *        16 *
*        4 *        15 *        15 *        15 *
*        5 *        14 *        14 *        14 *
*        6 *        13 *        13 *        13 *
*        7 *        12 *        12 *        12 *
*        8 *        11 *        11 *        11 *
*        9 *        10 *        10 *        10 *
*       10 *         9 *         9 *         9 *
*       11 *         8 *         8 *         8 *
*       12 *         7 *         7 *         7 *
*       13 *         6 *         6 *         6 *
*       14 *         5 *         5 *         5 *
*       15 *         4 *         4 *         4 *
*       16 *         3 *         3 *         3 *
*       17 *         2 *         2 *         2 *
*       18 *         1 *         1 *         1 *
*       19 *         0 *         0 *         0 *
************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegressionGH16804, TChainTwoFriendTTrees)
{
   auto friendFile1 = std::make_unique<TFile>(fFriend20EntriesFileName1);
   auto friendTree1 = friendFile1->Get<TTree>(fFriend20EntriesTreeName1);
   auto friendFile2 = std::make_unique<TFile>(fFriend20EntriesFileName2);
   auto friendTree2 = friendFile2->Get<TTree>(fFriend20EntriesTreeName2);

   TChain mainChain{fMainTreeName};
   mainChain.Add(fMainFileName);
   mainChain.AddFriend(friendTree1);
   mainChain.AddFriend(friendTree2);

   int mainBranch = -1, friendBranch = -1, yetAnotherFriendBranch = -1;
   auto mainBranchRet = mainChain.SetBranchAddress("mainBranch", &mainBranch);
   auto yetAnotherFriendBranchRet = mainChain.SetBranchAddress("yetAnotherFriendBranch", &yetAnotherFriendBranch);
   auto friendBranchRet = mainChain.SetBranchAddress("friendBranch", &friendBranch);
   ASSERT_EQ(mainBranchRet, 0);
   ASSERT_EQ(friendBranchRet, 0);
   ASSERT_EQ(yetAnotherFriendBranchRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedFriendBranch(20);
   std::vector<int> expectedyetAnotherFriendBranch(20);

   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedFriendBranch.begin(), expectedFriendBranch.end(), 0);
   std::reverse(expectedFriendBranch.begin(), expectedFriendBranch.end());
   std::iota(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end(), 0);
   std::reverse(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end());

   auto nEntries = mainChain.GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainChain.GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedyetAnotherFriendBranch[i], yetAnotherFriendBranch);
      EXPECT_EQ(expectedFriendBranch[i], friendBranch);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainChain.GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_tchain_two_friend_ttrees_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainChain.Scan("mainBranch:friendBranch:yetAnotherFriendBranch");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************
*    Row   * mainBranc * friendBra * yetAnothe *
************************************************
*        0 *        19 *        19 *        19 *
*        1 *        18 *        18 *        18 *
*        2 *        17 *        17 *        17 *
*        3 *        16 *        16 *        16 *
*        4 *        15 *        15 *        15 *
*        5 *        14 *        14 *        14 *
*        6 *        13 *        13 *        13 *
*        7 *        12 *        12 *        12 *
*        8 *        11 *        11 *        11 *
*        9 *        10 *        10 *        10 *
*       10 *         9 *         9 *         9 *
*       11 *         8 *         8 *         8 *
*       12 *         7 *         7 *         7 *
*       13 *         6 *         6 *         6 *
*       14 *         5 *         5 *         5 *
*       15 *         4 *         4 *         4 *
*       16 *         3 *         3 *         3 *
*       17 *         2 *         2 *         2 *
*       18 *         1 *         1 *         1 *
*       19 *         0 *         0 *         0 *
************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

TEST_F(RegressionGH16804, TChainOneFriendTTreeOneFriendTChain)
{
   auto friendFile1 = std::make_unique<TFile>(fFriend20EntriesFileName1);
   auto friendTree1 = friendFile1->Get<TTree>(fFriend20EntriesTreeName1);

   TChain friendChain{fFriend20EntriesTreeName2};
   friendChain.Add(fFriend20EntriesFileName2);

   TChain mainChain{fMainTreeName};
   mainChain.Add(fMainFileName);
   mainChain.AddFriend(&friendChain);
   mainChain.AddFriend(friendTree1);

   int mainBranch = -1, friendBranch = -1, yetAnotherFriendBranch = -1;
   auto mainBranchRet = mainChain.SetBranchAddress("mainBranch", &mainBranch);
   auto yetAnotherFriendBranchRet = mainChain.SetBranchAddress("yetAnotherFriendBranch", &yetAnotherFriendBranch);
   auto friendBranchRet = mainChain.SetBranchAddress("friendBranch", &friendBranch);
   ASSERT_EQ(mainBranchRet, 0);
   ASSERT_EQ(friendBranchRet, 0);
   ASSERT_EQ(yetAnotherFriendBranchRet, 0);

   std::vector<int> expectedMainBranch(20);
   std::vector<int> expectedFriendBranch(20);
   std::vector<int> expectedyetAnotherFriendBranch(20);

   std::iota(expectedMainBranch.begin(), expectedMainBranch.end(), 0);
   std::reverse(expectedMainBranch.begin(), expectedMainBranch.end());
   std::iota(expectedFriendBranch.begin(), expectedFriendBranch.end(), 0);
   std::reverse(expectedFriendBranch.begin(), expectedFriendBranch.end());
   std::iota(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end(), 0);
   std::reverse(expectedyetAnotherFriendBranch.begin(), expectedyetAnotherFriendBranch.end());

   auto nEntries = mainChain.GetEntriesFast();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      mainChain.GetEntry(i);
      EXPECT_EQ(expectedMainBranch[i], mainBranch);
      EXPECT_EQ(expectedyetAnotherFriendBranch[i], yetAnotherFriendBranch);
      EXPECT_EQ(expectedFriendBranch[i], friendBranch);
   }

   // Now test with TTree::Scan
   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(mainChain.GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"regression_16804_tchain_one_friend_tree_one_friend_chain_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         mainChain.Scan("mainBranch:friendBranch:yetAnotherFriendBranch");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************************************
*    Row   * mainBranc * friendBra * yetAnothe *
************************************************
*        0 *        19 *        19 *        19 *
*        1 *        18 *        18 *        18 *
*        2 *        17 *        17 *        17 *
*        3 *        16 *        16 *        16 *
*        4 *        15 *        15 *        15 *
*        5 *        14 *        14 *        14 *
*        6 *        13 *        13 *        13 *
*        7 *        12 *        12 *        12 *
*        8 *        11 *        11 *        11 *
*        9 *        10 *        10 *        10 *
*       10 *         9 *         9 *         9 *
*       11 *         8 *         8 *         8 *
*       12 *         7 *         7 *         7 *
*       13 *         6 *         6 *         6 *
*       14 *         5 *         5 *         5 *
*       15 *         4 *         4 *         4 *
*       16 *         3 *         3 *         3 *
*       17 *         2 *         2 *         2 *
*       18 *         1 *         1 *         1 *
*       19 *         0 *         0 *         0 *
************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}
