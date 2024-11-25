#include "TMemFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TInterpreter.h"
#include "TSystem.h"

#include "gtest/gtest.h"

#include <vector>

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

// Issue #6527
TEST(TTreeRegressions, ChangeFileWithTFileOnStack)
{
   TFile f("ChangeFileWithTFileOnStack.root", "recreate");
   EXPECT_FALSE(f.IsOnHeap());
   TTree t("t", "SetMaxTreeSize(1000)", 99, &f);
   int x;
   auto nentries = 20000;

   t.Branch("x", &x, "x/I");
   t.SetMaxTreeSize(1000);

   for (auto i = 0; i < nentries; i++){
      x = i;
      t.Fill();
   }

   auto cf = t.GetCurrentFile();
   cf->Write();
   cf->Close();

   gSystem->Unlink("ChangeFileWithTFileOnStack.root");
   gSystem->Unlink("ChangeFileWithTFileOnStack_1.root");
   gSystem->Unlink("ChangeFileWithTFileOnStack_2.root");
}

// Issue #6964
TEST(TTreeRegressions, GetLeafAndFriends)
{
   TTree t("t", "t");
   int x = 42;
   std::vector<int> v(1, 42);
   t.Branch("x", &x);
   t.Branch("vec", &v);
   t.Fill();

   TTree t2("t2", "t2");
   t2.Branch("x", &x);
   t2.Branch("vec", &v);
   t2.Fill();

   EXPECT_EQ(t.GetLeaf("asdklj", "x"), nullptr);
   EXPECT_EQ(t.GetLeaf("asdklj", "vec"), nullptr);

   t.AddFriend(&t2);
   EXPECT_EQ(t.GetLeaf("asdklj", "x"), nullptr);
   EXPECT_EQ(t.GetLeaf("asdklj", "vec"), nullptr);
}

// PR #14887
TEST(TTreeRegressions, LeafLongString)
{
   TTree t("t", "t");
   char s[1000];
   memset(s, 'a', 999);
   s[999] = 0;
   t.Branch("s", &s, "s/C");

   s[254] = 0;
   t.Fill();
   s[254] = 'a';
   s[255] = 0;
   t.Fill();
   s[255] = 'a';
   t.Fill();

   s[0] = 0;
   t.GetEntry(0);
   EXPECT_EQ(strlen(s), 254);

   s[0] = 0;
   t.GetEntry(1);
   EXPECT_EQ(strlen(s), 255);

   s[0] = 0;
   t.GetEntry(2);
   EXPECT_EQ(strlen(s), 999);
}

// Issue ROOT-9961
TEST(TTreeRegressions, PrintTopOnly)
{
   TTree tree("newtree", "");
   tree.Branch("brancha", 0, "brancha/I");
   tree.Branch("branchb", 0, "branchb/I");

   testing::internal::CaptureStdout();

   tree.Print("toponly");

   const std::string output = testing::internal::GetCapturedStdout();
   const auto ref = "******************************************************************************\n"
                    "*Tree    :newtree   :                                                        *\n"
                    "*Entries :        0 : Total =            1285 bytes  File  Size =          0 *\n"
                    "*        :          : Tree compression factor =   1.00                       *\n"
                    "******************************************************************************\n"
                    "branch: brancha                      0\n"
                    "branch: branchb                      0\n";
   EXPECT_EQ(output, ref);
}
