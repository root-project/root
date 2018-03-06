#include "ROOT/TDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TTree.h"
#include "gtest/gtest.h"

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::VecOps;

struct BrVal {
   Float_t a;
   Int_t i;
};

TEST(TDFLeaves, ReadIndividualLeaves)
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

   auto res = 0;
   auto histEntries = 0;
   try {
      ROOT::Experimental::TDataFrame df(t);
      df.Histo1D("b.a")->Draw();
      auto h = df.Define("aa", [](Float_t bv) { return bv; }, {"b.a"}).Histo1D("aa");
      histEntries = h->GetEntries();
   } catch (std::runtime_error &e) {
      std::cout << e.what() << '\n';
      res = 1;
   }
   EXPECT_EQ(0, res);
   EXPECT_EQ(nEntries, histEntries);
}
