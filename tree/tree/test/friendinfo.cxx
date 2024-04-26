#include <algorithm> // std::reverse
#include <memory>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "ROOT/InternalTreeUtils.hxx"
#include "ROOT/RFriendInfo.hxx"

#include "gtest/gtest.h"

namespace {

template <typename T>
void EXPECT_VEC_EQ(const std::vector<T> &v1, const std::vector<T> &v2)
{
   ASSERT_EQ(v1.size(), v2.size());
   for (std::size_t i = 0ul; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]);
   }
}

void FillTree(const std::string &fileName, const std::string &treeName, int nEntries)
{
   TFile f{fileName.c_str(), "RECREATE"};
   if (f.IsZombie()) {
      throw std::runtime_error("Could not create file for the test!");
   }
   TTree t{treeName.c_str(), treeName.c_str()};

   int b;
   t.Branch("b1", &b);

   for (int i = 0; i < nEntries; ++i) {
      b = i * 10;
      t.Fill();
   }

   const auto writtenBytes{t.Write()};
   if (writtenBytes == 0) {
      throw std::runtime_error("Could not write a tree for the test!");
   }
   f.Close();
}

constexpr std::size_t nEntriesInMainTree{20};
constexpr std::size_t nEntriesInFriendTree{5};
const std::string mainTreeName{"rfriendinfo_test_main"};
const std::string mainFileName{"rfriendinfo_test_main.root"};
const std::vector<std::string> friendTreeNames{
   "rfriendinfo_test_friend_0",
   "rfriendinfo_test_friend_1",
   "rfriendinfo_test_friend_2",
   "rfriendinfo_test_friend_3",
};
const std::vector<std::string> friendFileNames{
   "rfriendinfo_test_friend_0.root",
   "rfriendinfo_test_friend_1.root",
   "rfriendinfo_test_friend_2.root",
   "rfriendinfo_test_friend_3.root",
};

class RFriendInfoTest : public ::testing::TestWithParam<bool> {
public:
   static void SetUpTestSuite()
   {
      FillTree(mainFileName, mainTreeName, nEntriesInMainTree);
      for (std::size_t i = 0ul; i < friendFileNames.size(); i++) {
         FillTree(friendFileNames[i], friendTreeNames[i], nEntriesInFriendTree);
      }
   }

   static void TearDownTestSuite()
   {
      gSystem->Unlink(mainFileName.c_str());
      for (const auto &fileName : friendFileNames) {
         gSystem->Unlink(fileName.c_str());
      }
   }
};

TEST_P(RFriendInfoTest, GetFriendInfoMainChainFriendChain)
{

   TChain mainChain{mainTreeName.c_str()};
   mainChain.Add(mainFileName.c_str());

   TChain friendChain{};
   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      friendChain.Add((friendFileNames[i] + "?#" + friendTreeNames[i]).c_str());
   }

   mainChain.AddFriend(&friendChain);

   auto retrieveEntries = GetParam();
   auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(mainChain, retrieveEntries);
   const auto nFriends = friendInfo.fFriendFileNames.size();
   ASSERT_EQ(nFriends, 1);
   EXPECT_EQ(nFriends, friendInfo.fFriendChainSubNames.size());
   EXPECT_EQ(nFriends, friendInfo.fNEntriesPerTreePerFriend.size());
   for (std::size_t i = 0ul; i < nFriends; i++) {
      const auto nFilesInThisFriend = friendInfo.fFriendFileNames[i].size();
      ASSERT_EQ(nFilesInThisFriend, 4);
      EXPECT_EQ(nFilesInThisFriend, friendInfo.fFriendChainSubNames[i].size());
      EXPECT_EQ(nFilesInThisFriend, friendInfo.fNEntriesPerTreePerFriend[i].size());
   }
}

TEST_P(RFriendInfoTest, GetFriendInfoMainChainTwoFriendChains)
{

   TChain mainChain{mainTreeName.c_str()};
   mainChain.Add(mainFileName.c_str());

   TChain friendChain1{};
   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      friendChain1.Add((friendFileNames[i] + "?#" + friendTreeNames[i]).c_str());
   }

   // Same files, but in reverse order
   TChain friendChain2{};
   for (std::size_t i = 4ul; i > 0ul; i--) {
      friendChain2.Add((friendFileNames[i - 1] + "?#" + friendTreeNames[i - 1]).c_str());
   }

   mainChain.AddFriend(&friendChain1);
   mainChain.AddFriend(&friendChain2);

   auto retrieveEntries = GetParam();
   auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(mainChain, retrieveEntries);
   const auto nFriends = friendInfo.fFriendFileNames.size();
   ASSERT_EQ(nFriends, 2);
   EXPECT_EQ(nFriends, friendInfo.fFriendChainSubNames.size());
   EXPECT_EQ(nFriends, friendInfo.fNEntriesPerTreePerFriend.size());
   for (std::size_t i = 0ul; i < nFriends; i++) {
      const auto nFilesInThisFriend = friendInfo.fFriendFileNames[i].size();
      ASSERT_EQ(nFilesInThisFriend, 4);
      EXPECT_EQ(nFilesInThisFriend, friendInfo.fFriendChainSubNames[i].size());
      EXPECT_EQ(nFilesInThisFriend, friendInfo.fNEntriesPerTreePerFriend[i].size());
   }

   // File names and tree names were stored in the correct order
   // First friend: normal order
   const auto &fileNamesInFriend1 = friendInfo.fFriendFileNames[0];
   const auto &treeNamesInFriend1 = friendInfo.fFriendChainSubNames[0];
   EXPECT_VEC_EQ(fileNamesInFriend1, friendFileNames);
   EXPECT_VEC_EQ(treeNamesInFriend1, friendTreeNames);
   // Second friend: reversed order
   const auto &fileNamesInFriend2 = friendInfo.fFriendFileNames[1];
   const auto &treeNamesInFriend2 = friendInfo.fFriendChainSubNames[1];
   auto reversedFriendFileNames = friendFileNames;
   auto reversedFriendTreeNames = friendTreeNames;
   std::reverse(std::begin(reversedFriendFileNames), std::end(reversedFriendFileNames));
   std::reverse(std::begin(reversedFriendTreeNames), std::end(reversedFriendTreeNames));
   EXPECT_VEC_EQ(fileNamesInFriend2, reversedFriendFileNames);
   EXPECT_VEC_EQ(treeNamesInFriend2, reversedFriendTreeNames);
}

TEST_P(RFriendInfoTest, GetFriendInfoMainTreeFriendTree)
{

   TFile mainFile{mainFileName.c_str()};
   TFile friendFile{friendFileNames[0].c_str()};

   std::unique_ptr<TTree> mainTree{mainFile.Get<TTree>(mainTreeName.c_str())};
   std::unique_ptr<TTree> friendTree{friendFile.Get<TTree>(friendTreeNames[0].c_str())};

   mainTree->AddFriend(friendTree.get());

   auto retrieveEntries = GetParam();
   auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(*mainTree, retrieveEntries);
   const auto nFriends = friendInfo.fFriendFileNames.size();
   ASSERT_EQ(nFriends, 1);
   EXPECT_EQ(nFriends, friendInfo.fFriendChainSubNames.size());
   EXPECT_EQ(nFriends, friendInfo.fNEntriesPerTreePerFriend.size());
   for (std::size_t i = 0ul; i < nFriends; i++) {
      const auto nFilesInThisFriend = friendInfo.fFriendFileNames[i].size();
      ASSERT_EQ(nFilesInThisFriend, 1);
      EXPECT_EQ(nFilesInThisFriend, friendInfo.fNEntriesPerTreePerFriend[i].size());
   }
}

TEST_P(RFriendInfoTest, MakeFriendsMainChainFriendChain)
{

   TChain mainChain{mainTreeName.c_str()};
   mainChain.Add(mainFileName.c_str());

   TChain inFriendChain{};
   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      inFriendChain.Add((friendFileNames[i] + "?#" + friendTreeNames[i]).c_str());
   }

   mainChain.AddFriend(&inFriendChain);

   auto retrieveEntries = GetParam();
   auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(mainChain, retrieveEntries);
   auto friends = ROOT::Internal::TreeUtils::MakeFriends(friendInfo);

   const auto nFriends = friendInfo.fFriendNames.size();
   ASSERT_EQ(nFriends, 1);
   EXPECT_EQ(friends.size(), nFriends);

   const auto &outFriendChain = friends[0];
   EXPECT_EQ(outFriendChain->GetEntriesFast(), retrieveEntries ? nEntriesInMainTree : TTree::kMaxEntries);

   const auto *outFriendFiles = outFriendChain->GetListOfFiles();
   ASSERT_TRUE(outFriendFiles);
   EXPECT_EQ(outFriendFiles->GetEntries(), 4);

   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      const auto *curFile = outFriendFiles->At(i);
      EXPECT_STREQ(curFile->GetName(), friendTreeNames[i].c_str());
      EXPECT_STREQ(curFile->GetTitle(), friendFileNames[i].c_str());
   }
}

TEST_P(RFriendInfoTest, MakeFriendsFromAddFriendOverload1)
{
   ROOT::TreeUtils::RFriendInfo friendInfo;
   auto retrieveEntries = GetParam();
   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      if (retrieveEntries) {
         friendInfo.AddFriend(friendTreeNames[i], friendFileNames[i], /*alias*/ "", nEntriesInFriendTree);
      } else {
         friendInfo.AddFriend(friendTreeNames[i], friendFileNames[i]);
      }
   }

   auto friends = ROOT::Internal::TreeUtils::MakeFriends(friendInfo);
   const auto nFriends = friendInfo.fFriendNames.size();
   EXPECT_EQ(friends.size(), nFriends);

   // In this test, each friend tree is held in a separate `std::unique_ptr<TChain>`
   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      const auto &currentFriend = friends[i];
      EXPECT_EQ(currentFriend->GetEntriesFast(), retrieveEntries ? nEntriesInFriendTree : TTree::kMaxEntries);
      const auto *currentFriendFiles = currentFriend->GetListOfFiles();
      ASSERT_TRUE(currentFriendFiles);
      EXPECT_EQ(currentFriendFiles->GetEntries(), 1);
      const auto *curFile = currentFriendFiles->At(0);
      EXPECT_STREQ(curFile->GetName(), friendTreeNames[i].c_str());
      EXPECT_STREQ(curFile->GetTitle(), friendFileNames[i].c_str());
   }
}

TEST_P(RFriendInfoTest, MakeFriendsFromAddFriendOverload3)
{
   ROOT::TreeUtils::RFriendInfo friendInfo;
   auto retrieveEntries = GetParam();
   std::vector<std::pair<std::string, std::string>> treeAndFileNames;
   treeAndFileNames.reserve(friendTreeNames.size());
   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      treeAndFileNames.emplace_back(friendTreeNames[i], friendFileNames[i]);
   }
   if (retrieveEntries) {
      friendInfo.AddFriend(treeAndFileNames, /*alias*/ "",
                           std::vector<Long64_t>(friendTreeNames.size(), nEntriesInFriendTree));
   } else {
      friendInfo.AddFriend(treeAndFileNames);
   }

   auto friends = ROOT::Internal::TreeUtils::MakeFriends(friendInfo);
   // In this test, all friend trees are in the same TChain
   const auto nFriends = friendInfo.fFriendNames.size();
   EXPECT_EQ(friends.size(), nFriends);

   const auto &friendChain = friends[0];
   EXPECT_EQ(friendChain->GetEntriesFast(), retrieveEntries ? nEntriesInMainTree : TTree::kMaxEntries);

   const auto *friendFiles = friendChain->GetListOfFiles();
   ASSERT_TRUE(friendFiles);
   EXPECT_EQ(friendFiles->GetEntries(), 4);

   for (std::size_t i = 0ul; i < friendTreeNames.size(); i++) {
      const auto *curFile = friendFiles->At(i);
      EXPECT_STREQ(curFile->GetName(), friendTreeNames[i].c_str());
      EXPECT_STREQ(curFile->GetTitle(), friendFileNames[i].c_str());
   }
}

INSTANTIATE_TEST_SUITE_P(FriendInfoTests, RFriendInfoTest, ::testing::Values(true, false));

} // namespace
