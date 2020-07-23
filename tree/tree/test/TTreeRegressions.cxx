#include "TMemFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

// ROOT-10702
TEST(TTreeRegressions, CompositeTypeWithNameClash)
{
   struct Int {
      int x;
   };
   gInterpreter->Declare("struct Int { int x; };");

   TMemFile f("tree_compositetypewithnameclash.root", "recreate");
   {
      Int i{-1};
      int x = 1;
      TTree t("t", "t");
      const auto toJit = "((TTree*)" + std::to_string(reinterpret_cast<std::size_t>(&t)) + ")->Branch(\"i\", (Int*)" +
                         std::to_string(reinterpret_cast<std::size_t>(&i)) + ");";
      gInterpreter->ProcessLine(toJit.c_str());
      t.Branch("x", &x);
      t.Fill();
      t.Write();
   }

   auto &t = *f.Get<TTree>("t");
   int x = 123;

   const auto ret = t.SetBranchAddress("x", &x);
   EXPECT_EQ(ret, 0);

   t.GetEntry(0);
   EXPECT_EQ(x, 1);

   int ix;
   const auto toJit2 = "((TTree*)" + std::to_string(reinterpret_cast<std::size_t>(&t)) +
                       ")->SetBranchAddress(\"i.x\", (int*)" + std::to_string(reinterpret_cast<std::size_t>(&ix)) +
                       ");";
   gInterpreter->ProcessLine(toJit2.c_str());
   t.GetEntry(0);
   EXPECT_EQ(ix, -1);
}

// ROOT-10942
struct SimpleStruct {
   double a;
   double b;
};

TEST(TTreeRegressions, GetLeafByFullName)
{
   gInterpreter->Declare("struct SimpleStruct { double a; double b; };");
   SimpleStruct c;
   TTree t("t1", "t1");
   t.Branch("c", &c);
   t.Fill();

   EXPECT_TRUE(t.GetLeaf("a") != nullptr);
   EXPECT_TRUE(t.GetLeaf("c.a") != nullptr);
   EXPECT_TRUE(t.GetLeaf("c", "a") != nullptr);
   EXPECT_TRUE(t.GetLeaf(t.GetLeaf("a")->GetFullName()) != nullptr);
}
