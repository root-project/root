#include "TBranch.h"
#include "TChain.h"
#include "TChainIndex.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeIndex.h"

#include "gtest/gtest.h"

#include <string>

template <typename T>
void expect_vec_eq(const std::vector<T> &v1, const std::vector<T> &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

void FillIndexedFriend(const char *mainfile, const char *auxfile)
{
   // Start by creating main Tree
   TFile f(mainfile, "RECREATE");
   TTree mainTree("mainTree", "mainTree");
   int idx;
   mainTree.Branch("idx", &idx);
   int x;
   mainTree.Branch("x", &x);

   idx = 1;
   x = 1;
   mainTree.Fill();
   idx = 1;
   x = 2;
   mainTree.Fill();
   idx = 1;
   x = 3;
   mainTree.Fill();
   idx = 2;
   x = 4;
   mainTree.Fill();
   idx = 2;
   x = 5;
   mainTree.Fill();
   mainTree.Write();
   f.Close();

   // And aux tree
   TFile f2(auxfile, "RECREATE");
   TTree auxTree("auxTree", "auxTree");
   auxTree.Branch("idx", &idx);
   int y;
   auxTree.Branch("y", &y);
   idx = 2;
   y = 5;
   auxTree.Fill();
   idx = 1;
   y = 7;
   auxTree.Fill();
   auxTree.Write();
   f2.Close();
}

TEST(TTreeIndexClone, TChainIndex)
{
   auto mainFile = "IndexedFriendChain_main.root";
   auto auxFile = "IndexedFriendChain_aux.root";
   FillIndexedFriend(mainFile, auxFile);

   TChain mainChain("mainTree", "mainTree");
   mainChain.Add(mainFile);
   TChain auxChain("auxTree", "auxTree");
   auxChain.Add(auxFile);

   auxChain.BuildIndex("idx");
   mainChain.AddFriend(&auxChain);

   {
      // First check the index is reporting entry values as expected
      int x;
      int y;
      int idx_main;
      mainChain.SetBranchAddress("idx", &idx_main);
      mainChain.SetBranchAddress("x", &x);
      auxChain.SetBranchAddress("y", &y);

      std::vector<int> vecx;
      std::vector<int> vecy;

      for (auto i = 0; i < mainChain.GetEntries(); i++) {
         mainChain.GetEntry(i);
         auxChain.GetEntryWithIndex(idx_main);
         vecx.push_back(x);
         vecy.push_back(y);
      }
      std::vector<int> refx{{1, 2, 3, 4, 5}};
      expect_vec_eq(vecx, refx);
      std::vector<int> refy{{7, 7, 7, 5, 5}};
      expect_vec_eq(vecy, refy);
   }

   {
      // Copy the index and check entry values again
      auto *oldIndex = auxChain.GetTreeIndex();
      auto *newIndex{dynamic_cast<TChainIndex *>(oldIndex->Clone())};
      EXPECT_NE(newIndex, nullptr);

      newIndex->SetTree(&auxChain);
      auxChain.SetTreeIndex(newIndex);

      delete oldIndex;
      oldIndex = nullptr;

      int x;
      int y;
      int idx_main;
      mainChain.SetBranchAddress("idx", &idx_main);
      mainChain.SetBranchAddress("x", &x);
      auxChain.SetBranchAddress("y", &y);

      std::vector<int> vecx;
      std::vector<int> vecy;

      for (auto i = 0; i < mainChain.GetEntries(); i++) {
         mainChain.GetEntry(i);
         auxChain.GetEntryWithIndex(idx_main);
         vecx.push_back(x);
         vecy.push_back(y);
      }
      std::vector<int> refx{{1, 2, 3, 4, 5}};
      expect_vec_eq(vecx, refx);
      std::vector<int> refy{{7, 7, 7, 5, 5}};
      expect_vec_eq(vecy, refy);
   }

   gSystem->Unlink(mainFile);
   gSystem->Unlink(auxFile);
}

TEST(TTreeIndexClone, TTreeIndex)
{
   auto mainFile = "IndexedFriendTree_main.root";
   auto auxFile = "IndexedFriendTree_aux.root";
   FillIndexedFriend(mainFile, auxFile);

   TFile mainF(mainFile);
   auto *mainTree = mainF.Get<TTree>("mainTree");
   EXPECT_NE(mainTree, nullptr);

   TFile auxF(auxFile);
   auto *auxTree = auxF.Get<TTree>("auxTree");
   EXPECT_NE(auxTree, nullptr);

   auxTree->BuildIndex("idx");
   mainTree->AddFriend(auxTree);

   {
      // First check the index is reporting entry values as expected
      int x;
      int y;
      int idx_main;
      mainTree->SetBranchAddress("idx", &idx_main);
      mainTree->SetBranchAddress("x", &x);
      auxTree->SetBranchAddress("y", &y);

      std::vector<int> vecx;
      std::vector<int> vecy;

      for (auto i = 0; i < mainTree->GetEntries(); i++) {
         mainTree->GetEntry(i);
         auxTree->GetEntryWithIndex(idx_main);
         vecx.push_back(x);
         vecy.push_back(y);
      }
      std::vector<int> refx{{1, 2, 3, 4, 5}};
      expect_vec_eq(vecx, refx);
      std::vector<int> refy{{7, 7, 7, 5, 5}};
      expect_vec_eq(vecy, refy);
   }

   {
      // Copy the index and check entry values again
      auto *oldIndex = auxTree->GetTreeIndex();
      auto *newIndex{dynamic_cast<TTreeIndex *>(oldIndex->Clone())};
      EXPECT_NE(newIndex, nullptr);

      newIndex->SetTree(auxTree);
      auxTree->SetTreeIndex(newIndex);

      delete oldIndex;
      oldIndex = nullptr;

      int x;
      int y;
      int idx_main;
      mainTree->SetBranchAddress("idx", &idx_main);
      mainTree->SetBranchAddress("x", &x);
      auxTree->SetBranchAddress("y", &y);

      std::vector<int> vecx;
      std::vector<int> vecy;

      for (auto i = 0; i < mainTree->GetEntries(); i++) {
         mainTree->GetEntry(i);
         auxTree->GetEntryWithIndex(idx_main);
         vecx.push_back(x);
         vecy.push_back(y);
      }
      std::vector<int> refx{{1, 2, 3, 4, 5}};
      expect_vec_eq(vecx, refx);
      std::vector<int> refy{{7, 7, 7, 5, 5}};
      expect_vec_eq(vecy, refy);
   }

   gSystem->Unlink(mainFile);
   gSystem->Unlink(auxFile);
}
