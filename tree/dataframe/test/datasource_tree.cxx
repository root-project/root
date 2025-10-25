/* TTree-specific tests for RDataFrame */
#include <cstdio>

#include <TTree.h>
#include <TFile.h>
#include <TChain.h>

#include <ROOT/RDataFrame.hxx>

#include <gtest/gtest.h>

#include "ClassWithNestedSameName.hxx"

struct InputTreeRAII {
   const char *fTreeName{"nestedsamename"};
   const char *fFileName{"nestedsamename.root"};

   InputTreeRAII()
   {
      std::unique_ptr<TFile> ofile{TFile::Open(fFileName, "recreate")};

      std::unique_ptr<TTree> myTree = std::make_unique<TTree>(fTreeName, fTreeName);

      TopLevel obj{};

      myTree->Branch("toplevel", &obj);
      myTree->Fill();
      ofile->Write();
   }

   ~InputTreeRAII() { std::remove(fFileName); }
};

void expect_vec_eq(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST(RTTreeDS, BranchWithNestedSameName)
{
   // Test that a toplevel branch with data member "fInner" with in turn another data member with the same name is
   // properly retrieved in the list of branch names
   InputTreeRAII dataset{};

   ROOT::RDataFrame df{dataset.fTreeName, dataset.fFileName};
   auto branchNames = df.GetColumnNames();

   std::vector<std::string> expectedBranchNames{"toplevel", "toplevel.fInner", "toplevel.fInner.fInner.fPt", "fInner",
                                                "fInner.fInner.fPt"};

   std::sort(expectedBranchNames.begin(), expectedBranchNames.end());
   std::sort(branchNames.begin(), branchNames.end());

   expect_vec_eq(branchNames, expectedBranchNames);
}

#ifdef R__USE_IMT
struct Dataset20164RAIII {
   const char *fTreeName{"tree_20164"};
   const char *fFileName{"tree_20164.root"};

   Dataset20164RAIII()
   {
      std::unique_ptr<TFile> ofile{TFile::Open(fFileName, "recreate")};

      std::unique_ptr<TTree> myTree = std::make_unique<TTree>(fTreeName, fTreeName);

      int v1{42};

      myTree->Branch("v1", &v1);
      myTree->Fill();
      ofile->Write();

      // It's important to use 1 as the logic in RTTreeDS was faulty only for the case fNSlots==1
      ROOT::EnableImplicitMT(1);
   }

   ~Dataset20164RAIII()
   {
      ROOT::DisableImplicitMT();
      std::remove(fFileName);
   }
};

TEST(RTTreeDS, Regression20164)
{
   // Test that TTree thread tasks coherently use the TTreeReader created for the specific task and do not access
   // a TTreeReader from the RTTreeDS.
   Dataset20164RAIII dataset{};

   auto chain = std::make_unique<TChain>(dataset.fTreeName);
   chain->Add(dataset.fFileName);

   auto val = ROOT::RDataFrame(*chain).Sum<int>("v1").GetValue();

   ASSERT_EQ(val, 42);
}

#endif
