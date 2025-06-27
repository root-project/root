#include "ROOT/RDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TTree.h"
#include "gtest/gtest.h"

#include <memory>
#include <TClonesArray.h>
#include <TLorentzVector.h>
#include <ROOT/RVec.hxx>
#include <ROOT/TestSupport.hxx>

#include <TFile.h>

using namespace ROOT::VecOps;

template <typename T0, typename T1>
void expect_vec_float_eq(const T0 &v1, const T1 &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_FLOAT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

struct BrVal {
   Float_t a;
   Int_t i;
};

TEST(RDFLeaves, ReadIndividualLeaves)
{
   // Taken directly from ROOT-9142
   const auto nEntries = 8;
   TTree t("t", "t");
   BrVal val;
   t.Branch("b", &val, "a/F:i/I");
   for (auto i : ROOT::TSeqI(nEntries)) {
      val.a = val.i = i;
      t.Fill();
   }

   auto histEntries = 0;
   auto op = [&](){
      ROOT::RDataFrame df(t);
      df.Histo1D<float>("b.a")->Draw();
      auto h = df.Define("aa", [](Float_t bv) { return bv; }, {"b.a"}).Histo1D<float>("aa");
      histEntries = h->GetEntries();
   };
   EXPECT_NO_THROW(op());
   EXPECT_EQ(nEntries, histEntries);
}

struct S {
  int a, b;
};

TEST(RDFLeaves, LeavesWithDotNameClass)
{
   TTree t("t", "t");
   S s{40, 41};
   t.Branch("s", &s, "a/I:b/I");
   int s_a = 2;
   t.Branch("s_a", &s_a);
   t.Fill();
   t.ResetBranchAddresses();
   ROOT::RDataFrame d(t);
   auto four = d.Define("res1", "s_a + s_a").Min<int>("res1");
   auto ans = d.Define("res2", "s.a + s_a").Min<int>("res2");
  
   EXPECT_EQ(*four, 4);
   EXPECT_EQ(*ans, 42);
}

struct S2 {
  int a;
};

TEST(RDFLeaves, OneLeafWithDotNameClass)
{
   TTree t("t", "t");
   S2 s{40};
   t.Branch("s", &s, "a/I");
   t.Fill();
   t.ResetBranchAddresses();
   ROOT::RDataFrame d(t);
   auto res = d.Define("res", "s.a").Min<int>("res");

   EXPECT_EQ(*res, 40);
}

TEST(RDFLeaves, LeafFromTClonesArray)
{
   // Regression test for https://github.com/root-project/root/issues/19104
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kWarning, "TTree::Bronch",
                         "Using split mode on a class: TLorentzVector with a custom Streamer");
#ifndef NDEBUG
   diagRAII.requiredDiag(
      kWarning, "RTreeColumnReader::Get",
      "Branch ca.fE hangs from a non-split branch. A copy is being performed in order to properly read the content.");
#endif
   constexpr static auto fName{"leaffromtclonesarray.root"};
   struct Dataset {
      Dataset()
      {
         auto f = std::make_unique<TFile>(fName, "recreate");
         auto t = std::make_unique<TTree>("t", "t");
         auto ca = std::make_unique<TClonesArray>("TLorentzVector");
         auto branchData = ca.get();
         auto &caRef = *ca;
         t->Branch("ca", &branchData);
         // Fill the array with two elements. The issue arose from the fact
         // we were not computing the correct size of the correction value
         // type, so the wrong offset was used when reading back next elements
         // of the array
         new (caRef[0]) TLorentzVector(0, 0, 0, 42.42);
         new (caRef[1]) TLorentzVector(0, 0, 0, 84.84);
         t->Fill();
         f->Write();
      }

      ~Dataset() { std::remove(fName); }
   } _;

   std::vector expected{42.42, 84.84};

   ROOT::RDataFrame df("t", fName);
   auto resultptr = df.Take<ROOT::RVecD>("ca.fE");
   ASSERT_EQ(resultptr->size(), 1);
   expect_vec_float_eq(expected, resultptr->at(0));
}
