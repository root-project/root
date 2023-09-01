#include "ROOT/RDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TTree.h"
#include "gtest/gtest.h"


using namespace ROOT::VecOps;

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

