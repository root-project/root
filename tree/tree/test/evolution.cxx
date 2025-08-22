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
