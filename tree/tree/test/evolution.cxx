#include "gtest/gtest.h"

#include "EvolutionStruct.hxx"

#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <memory>
#include <vector>

TEST(TTree, RenameSplitCollection)
{
   constexpr auto fileName{"test_ttree_rename_split_collection.root"};
   struct FileGuard {
      ~FileGuard() { std::remove("test_ttree_rename_split_collection.root"); }
   } _;

   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "RECREATE"));
      TTree tree = TTree("t", "");

      auto vec = new std::vector<EvolutionStruct_V2>({EvolutionStruct_V2{1.0}});
      tree.Branch("vec", "std::vector<EvolutionStruct_V2>", &vec, 32000, 99);

      tree.Fill();
      tree.Write();

      file->Close();
   }

   auto file = std::unique_ptr<TFile>(TFile::Open(fileName));
   auto tree = file->Get<TTree>("t");

   std::vector<EvolutionStruct_V3> *vec = nullptr;
   tree->SetBranchAddress("vec", &vec);

   tree->GetEntry(0);
   ASSERT_EQ(1u, vec->size());
   EXPECT_FLOAT_EQ(1.0, vec->at(0).fNewMember);
}

// Covers GitHub issue #22725
TEST(TTree, ClonesArrayIoRuleGH22725)
{
   constexpr auto fileName{"test_ttree_clonesarray_iorule_gh22725.root"};
   struct FileGuard {
      ~FileGuard() { std::remove("test_ttree_clonesarray_iorule_gh22725.root"); }
   } _;

   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "RECREATE"));
      TTree tree("t", "test tree");

      EvolutionClonesHolder_V2 holder;
      tree.Branch("holder", &holder);

      new ((*holder.fItems)[0]) EvolutionClonesItem_V2(1.0f, 2.0f, 3.0f);
      tree.Fill();
      tree.Write();
      file->Close();
   }

   auto file = std::unique_ptr<TFile>(TFile::Open(fileName));
   auto tree = file->Get<TTree>("t");

   EvolutionClonesHolder_V3 *holder = nullptr;
   tree->SetBranchAddress("holder", &holder);

   tree->GetEntry(0);
   ASSERT_NE(holder, nullptr);
   ASSERT_NE(holder->fItems, nullptr);
   ASSERT_EQ(holder->fItems->GetEntries(), 1);
   auto item = static_cast<EvolutionClonesItem_V3 *>(holder->fItems->At(0));
   ASSERT_NE(item, nullptr);
   EXPECT_EQ(item->fCanary1, 33);
   EXPECT_EQ(item->fCanary2, 66);
   EXPECT_EQ(item->fCombined, 203);
   delete holder;
}
