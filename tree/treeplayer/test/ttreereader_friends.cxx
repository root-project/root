#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "ROOT/TestSupport.hxx"
#include <string>

struct TTreeReaderFriends : public ::testing::Test {
   constexpr static auto fMainTreeName{"tree_10entries"};
   constexpr static auto fMainFileName{"tree_10entries.root"};
   constexpr static auto fFriendTreeName{"tree_2entries"};
   constexpr static auto fFriendFileName{"tree_2entries.root"};
   constexpr static auto fExtraTreeName{"tree_5entries"};
   constexpr static auto fExtraFileName{"tree_5entries.root"};
   constexpr static auto fMainBranch{"x"};
   constexpr static auto fFriendBranch{"y"};
   constexpr static auto fExtraBranch{"e"};
   constexpr static auto fMainEntries{10};
   constexpr static auto fFriendEntries{2};
   constexpr static auto fExtraEntries{5};

   static void SetUpTestCase()
   {
      TFile fMain(fMainFileName, "RECREATE");
      TTree tMain(fMainTreeName, fMainTreeName);
      int x{};
      tMain.Branch("x", &x, "x/I");
      for (int i = 0; i < fMainEntries; i++) {
         x = i;
         tMain.Fill();
      }
      fMain.WriteObject(&tMain, fMainTreeName);

      TFile fFriend(fFriendFileName, "RECREATE");
      TTree tFriend(fFriendTreeName, fFriendTreeName);
      int y{};
      tFriend.Branch("y", &y, "y/I");
      for (int i = 0; i < fFriendEntries; i++) {
         y = i;
         tFriend.Fill();
      }
      fFriend.WriteObject(&tFriend, fFriendTreeName);

      TFile fExtra(fExtraFileName, "RECREATE");
      TTree tExtra(fExtraTreeName, fExtraTreeName);
      int e{};
      tExtra.Branch("e", &e, "e/I");
      for (int i = 0; i < fExtraEntries; i++) {
         e = i;
         tExtra.Fill();
      }
      fExtra.WriteObject(&tExtra, fExtraTreeName);
   }

   static void TearDownTestCase()
   {
      std::remove(fMainFileName);
      std::remove(fFriendFileName);
      std::remove(fExtraFileName);
   }
};

TEST_F(TTreeReaderFriends, MainTreeLongerTTree)
{
   std::unique_ptr<TFile> fMain{TFile::Open(fMainFileName)};
   std::unique_ptr<TTree> tMain{fMain->Get<TTree>(fMainTreeName)};
   std::unique_ptr<TFile> fFriend{TFile::Open(fFriendFileName)};
   std::unique_ptr<TTree> tFriend{fFriend->Get<TTree>(fFriendTreeName)};
   tMain->AddFriend(tFriend.get());

   TTreeReader r(tMain.get());
   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");

   try {
      // One entry beyond end of friend tree
      r.SetEntry(2);
   } catch (const std::runtime_error &err) {
      const std::string expected = "Cannot read entry 2 from friend tree '" + std::string(fFriendTreeName) +
                                   "'. The friend tree has less entries than the main tree. "
                                   "Make sure all trees of the dataset have the same number of entries.";
      EXPECT_STREQ(err.what(), expected.c_str());
   }
}

TEST_F(TTreeReaderFriends, MainTreeLongerTChain)
{
   TChain ch(fMainTreeName);
   ch.Add(fMainFileName);
   ch.AddFriend(fFriendTreeName, fFriendFileName);
   TTreeReader r(&ch);

   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");
   try {
      // One entry beyond end of friend tree
      r.SetEntry(2);
   } catch (const std::runtime_error &err) {
      const std::string expected = "Cannot read entry 2 from friend tree '" + std::string(fFriendTreeName) +
                                   "'. The friend tree has less entries than the main tree. "
                                   "Make sure all trees of the dataset have the same number of entries.";
      EXPECT_STREQ(err.what(), expected.c_str());
   }
}

TEST_F(TTreeReaderFriends, MainTreeLongerTChainExtraFriend)
{
   TChain ch(fMainTreeName);
   ch.Add(fMainFileName);
   ch.AddFriend(fFriendTreeName, fFriendFileName);
   ch.AddFriend(fExtraTreeName, fExtraFileName);
   TTreeReader r(&ch);

   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");
   TTreeReaderValue<int> e(r, "e");

   try {
      // One entry beyond end of shortest friend tree
      r.SetEntry(2);
   } catch (const std::runtime_error &err) {
      const std::string expected = "Cannot read entry 2 from friend tree '" + std::string(fFriendTreeName) +
                                   "'. The friend tree has less entries than the main tree. "
                                   "Make sure all trees of the dataset have the same number of entries.";
      EXPECT_STREQ(err.what(), expected.c_str());
   }
}

TEST_F(TTreeReaderFriends, MainTreeShorterTTree)
{
   std::unique_ptr<TFile> fMain{TFile::Open(fMainFileName)};
   std::unique_ptr<TTree> tMain{fMain->Get<TTree>(fMainTreeName)};
   std::unique_ptr<TFile> fFriend{TFile::Open(fFriendFileName)};
   std::unique_ptr<TTree> tFriend{fFriend->Get<TTree>(fFriendTreeName)};
   tFriend->AddFriend(tMain.get());

   TTreeReader r(tFriend.get());
   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");

   ROOT_EXPECT_WARNING(r.SetEntry(2), "SetEntryBase",
                       "Last entry available from main tree '" + std::string(fFriendTreeName) +
                          "' was 1 but friend tree '" + std::string(fMainTreeName) +
                          "' has more entries beyond the end of the main tree.");
}

TEST_F(TTreeReaderFriends, MainTreeShorterTChain)
{
   TChain ch(fFriendTreeName);
   ch.Add(fFriendFileName);
   ch.AddFriend(fMainTreeName, fMainFileName);

   TTreeReader r(&ch);
   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");
   ROOT_EXPECT_WARNING(r.SetEntry(2), "SetEntryBase",
                       "Last entry available from main tree '" + std::string(fFriendTreeName) +
                          "' was 1 but friend tree '" + std::string(fMainTreeName) +
                          "' has more entries beyond the end of the main tree.");
}

TEST_F(TTreeReaderFriends, MainTreeShorterTTreeExtraFriend)
{
   std::unique_ptr<TFile> fMain{TFile::Open(fMainFileName)};
   std::unique_ptr<TTree> tMain{fMain->Get<TTree>(fMainTreeName)};
   std::unique_ptr<TFile> fFriend{TFile::Open(fFriendFileName)};
   std::unique_ptr<TTree> tFriend{fFriend->Get<TTree>(fFriendTreeName)};
   std::unique_ptr<TFile> fExtra{TFile::Open(fExtraFileName)};
   std::unique_ptr<TTree> tExtra{fExtra->Get<TTree>(fExtraTreeName)};
   tFriend->AddFriend(tMain.get());
   tFriend->AddFriend(tExtra.get());

   TTreeReader r(tFriend.get());
   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");
   TTreeReaderValue<int> e(r, "e");

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "SetEntryBase",
                      "Last entry available from main tree '" + std::string(fFriendTreeName) +
                         "' was 1 but friend tree '" + std::string(fMainTreeName) +
                         "' has more entries beyond the end of the main tree.",
                      /*matchFullMessage*/ true);
   diags.requiredDiag(kWarning, "SetEntryBase",
                      "Last entry available from main tree '" + std::string(fFriendTreeName) +
                         "' was 1 but friend tree '" + std::string(fExtraTreeName) +
                         "' has more entries beyond the end of the main tree.",
                      /*matchFullMessage*/ true);
   r.SetEntry(2);
}

TEST_F(TTreeReaderFriends, MainTreeShorterTChainExtraFriend)
{
   TChain ch(fFriendTreeName);
   ch.Add(fFriendFileName);
   ch.AddFriend(fMainTreeName, fMainFileName);
   ch.AddFriend(fExtraTreeName, fExtraFileName);

   TTreeReader r(&ch);
   TTreeReaderValue<int> x(r, "x");
   TTreeReaderValue<int> y(r, "y");
   TTreeReaderValue<int> e(r, "e");

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "SetEntryBase",
                      "Last entry available from main tree '" + std::string(fFriendTreeName) +
                         "' was 1 but friend tree '" + std::string(fMainTreeName) +
                         "' has more entries beyond the end of the main tree.",
                      /*matchFullMessage*/ true);
   diags.requiredDiag(kWarning, "SetEntryBase",
                      "Last entry available from main tree '" + std::string(fFriendTreeName) +
                         "' was 1 but friend tree '" + std::string(fExtraTreeName) +
                         "' has more entries beyond the end of the main tree.",
                      /*matchFullMessage*/ true);
   r.SetEntry(2);
}

// ROOT-9886 "Reader: Unexpected value in friend tree: expected 13, read 3"
TEST_F(TTreeReaderFriends, EntryChainFriend)
{
   constexpr const int kEntriesPerTree = 10;
   for (int i = 0; i < 3; ++i) {
      TFile file(std::string("maintree9886_" + std::to_string(i) + ".root").c_str(), "RECREATE");
      TTree tree("tree", "Main tree");
      int val;
      tree.Branch("val", &val);
      for (int j = 0; j < kEntriesPerTree; ++j) {
         val = -(j + i * kEntriesPerTree);
         tree.Fill();
      }
      tree.Write();
   }
   for (int i = 0; i < 1; ++i) {
      TFile file(std::string("friendtree9886_" + std::to_string(i) + ".root").c_str(), "RECREATE");
      TTree tree("friend", "Friend tree");
      int entry;
      tree.Branch("entry", &entry);
      for (int j = 0; j < 3 * kEntriesPerTree; ++j) {
         entry = j + i * kEntriesPerTree;
         tree.Fill();
      }
      tree.Write();
   }

   std::unique_ptr<TChain> chainFriend{new TChain("friend")};
   chainFriend->Add("friendtree9886_*.root");
   std::unique_ptr<TChain> chainMain{new TChain("tree")};
   chainMain->Add("maintree9886_*.root");
   chainMain->AddFriend(chainFriend.get());
   TChain *chain = chainMain.get();
   
   // Test using traditional SetBranchAddress
   {
      EXPECT_EQ(chain->GetEntries(), 30);
      EXPECT_EQ(chainFriend->GetEntries(), 30);

      int mainVal;
      int friendVal;
      chain->SetBranchAddress("val", &mainVal);
      chain->SetBranchAddress("entry", &friendVal);
      int entry = -1;
      while (chain->GetEntry(++entry)) {
         EXPECT_EQ(mainVal, -entry);
         EXPECT_EQ(friendVal, entry);
      }
      EXPECT_EQ(entry, 30);
      chain->ResetBranchAddresses();
   }

   // Equivalent test using TTreeReader
   {
      EXPECT_EQ(chain->GetEntries(), 30);
      EXPECT_EQ(chainFriend->GetEntries(), 30);

      TTreeReader reader(chain);
      TTreeReaderValue<int> mainVal(reader, "val");
      TTreeReaderValue<int> friendVal(reader, "entry");
      int entry = 0;
      while (reader.Next()) {
         EXPECT_EQ(*mainVal, -entry);
         EXPECT_EQ(*friendVal, entry);
         ++entry;
      }
      EXPECT_EQ(entry, 30);
   }
}
