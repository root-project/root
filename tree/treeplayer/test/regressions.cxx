#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

#include "gtest/gtest.h"


// ROOT-10702
TEST(TTreeReaderRegressions, CompositeTypeWithNameClash)
{
   struct Int { int x; };
	gInterpreter->Declare("struct Int { int x; };");

   const auto fname = "ttreereader_compositetypewithnameclash.root";

   {
      TFile f(fname, "recreate");
      Int i{-1};
      int x = 1;
      TTree t("t", "t");
      const auto toJit = "((TTree*)" + std::to_string(reinterpret_cast<std::size_t>(&t)) + ")->Branch(\"i\", (Int*)" +
                         std::to_string(reinterpret_cast<std::size_t>(&i)) + ");";
      gInterpreter->ProcessLine(toJit.c_str());
      t.Branch("x", &x);
      t.Fill();
      t.Write();
      f.Close();
   }

   TFile f(fname);
   TTreeReader r("t", &f);
   TTreeReaderValue<int> iv(r, "i.x");
   TTreeReaderValue<int> xv(r, "x");
   r.Next();
   EXPECT_EQ(xv.GetSetupStatus(), 0);
   if (xv.GetSetupStatus() == 0) {
      EXPECT_EQ(*xv, 1);
   }
   EXPECT_EQ(iv.GetSetupStatus(), 0);
   if (iv.GetSetupStatus() == 0) {
      EXPECT_EQ(*iv, -1);
   }

   gSystem->Unlink(fname);
}
